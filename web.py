"""
genesis/web.py — live web UI for the genesis substrate.

A thin visualization layer wrapped around substrate.World /
PatchyWorld / OasisWorld. The simulation runs in a background
thread; the HTTP server exposes its state as JSON and accepts
control commands. A single-page HTML in web/ renders the grid
on a canvas and mirrors the census.

Start the server:

    python3 web.py --port 8765

Then open http://localhost:8765/ in a browser.

Stdlib only — no fastapi / flask dependency.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import substrate
from substrate import World
from patchy import PatchyWorld, OasisWorld


WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

WORLD_VARIANTS = {
    "uniform": World,
    "patchy": PatchyWorld,
    "oasis": OasisWorld,
}


class Simulation:
    """Owns the World and runs it on a background thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.world: World | None = None
        self.variant: str = "uniform"
        self.running: bool = False
        self.speed: float = 15.0  # ticks per second (was 50.0)
        self.stop_flag = threading.Event()
        self.thread: threading.Thread | None = None
        # Match run.py --gui defaults: uniform food, 64x64 grid, 300 pop.
        # That's what produces the recognizable "moving swarm" picture;
        # patchy world intentionally clusters agents at food patches.
        self.reset(variant="uniform", params={"WORLD_SIZE": 64})

    def reset(self, *, variant: str, params: dict):
        """Instantiate a fresh world. `params` overrides module globals."""
        with self.lock:
            self._stop_thread()
            # Apply module-global overrides before constructing the world.
            # Keys match substrate module names (MUTATION_RATE, FOOD_RATE,
            # INITIAL_POPULATION, INITIAL_ENERGY, WORLD_SIZE, etc.).
            for key, value in params.items():
                if hasattr(substrate, key):
                    setattr(substrate, key, value)

            size = int(params.get("WORLD_SIZE", substrate.WORLD_SIZE))
            cls = WORLD_VARIANTS.get(variant, World)
            self.world = cls(size=size)
            self.variant = variant
            self.world.seed_population(
                count=int(params.get("INITIAL_POPULATION", substrate.INITIAL_POPULATION)),
                luca_fraction=float(params.get("LUCA_FRACTION", 0.3)),
            )
            self.running = False
            self._start_thread()

    def control(self, action: str):
        with self.lock:
            if action == "play":
                self.running = True
            elif action == "pause":
                self.running = False
            elif action == "step":
                if self.world is not None:
                    self.world.tick_step()

    def set_speed(self, speed: float):
        self.speed = max(1.0, min(500.0, float(speed)))

    # ── Thread management ─────────────────────────────────────────

    def _start_thread(self):
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _stop_thread(self):
        self.stop_flag.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def _run_loop(self):
        while not self.stop_flag.is_set():
            if self.running and self.world is not None:
                try:
                    self.world.tick_step()
                except Exception:
                    # Swallow — substrate may raise on edge cases;
                    # keep the server alive so we can reset from UI.
                    pass
                time.sleep(1.0 / self.speed)
            else:
                time.sleep(0.5)  # longer sleep when paused (was 0.05)

    # ── State serialization ───────────────────────────────────────

    def state(self) -> dict:
        with self.lock:
            w = self.world
            if w is None:
                return {"tick": 0, "size": 0, "agents": [], "food": [], "traces": []}

            agents: list[list] = []
            food: list[list] = []
            traces: list[list] = []
            for y, row in enumerate(w.grid):
                for x, cell in enumerate(row):
                    if cell.agent and cell.agent.alive:
                        agent = cell.agent
                        # Lineage proxy: hash first 8 genome bytes.
                        # Similar clades → similar colors, mutations
                        # jump color slowly. Actual lineage tracking
                        # lives in phylogeny.py; this is purely viz.
                        head = bytes(agent.genome[:8])
                        lineage_hue = (sum(head) * 7 + len(agent.genome) * 3) % 360
                        agents.append([x, y, lineage_hue, agent.energy])
                    if cell.food > 0:
                        food.append([x, y, cell.food])
                    if cell.trace > 0:
                        traces.append([x, y, cell.trace])

            # Opcode histogram across all living genomes — for the
            # "what's selection favoring right now" sidebar.
            op_counts = [0] * substrate.NUM_OPCODES
            total_len = 0
            agent_count = 0
            for row in w.grid:
                for cell in row:
                    if cell.agent and cell.agent.alive:
                        agent = cell.agent
                        agent_count += 1
                        for byte in agent.genome:
                            op = byte & 0x1F
                            if op < substrate.NUM_OPCODES:
                                op_counts[op] += 1
                        total_len += len(agent.genome)
            avg_genome_len = total_len / agent_count if agent_count else 0

            return {
                "tick": w.tick,
                "size": w.size,
                "variant": self.variant,
                "running": self.running,
                "speed": self.speed,
                "agents": agents,
                "food": food,
                "traces": traces,
                "census": {
                    "population": agent_count,
                    "avg_genome_len": round(avg_genome_len, 2),
                    "opcodes": op_counts,
                },
                "params": {
                    "MUTATION_RATE": substrate.MUTATION_RATE,
                    "FOOD_RATE": substrate.FOOD_RATE,
                    "INITIAL_POPULATION": substrate.INITIAL_POPULATION,
                    "INITIAL_ENERGY": substrate.INITIAL_ENERGY,
                    "WORLD_SIZE": substrate.WORLD_SIZE,
                },
            }


SIM = Simulation()


# ── HTTP handler ────────────────────────────────────────────────────


def _ensure_tick_step():
    """World's tick method is called `tick` (attribute) AND tick()
    (method that advances state). We need to expose a single call
    that advances — substrate.World defines it as `step()` or `tick()`?
    Probe and alias."""
    if not hasattr(World, "tick_step"):
        if hasattr(World, "step"):
            World.tick_step = World.step  # type: ignore
        else:
            # Fallback: many substrates call the advance method `tick()`,
            # but we can't name-collide with the tick counter attribute.
            # Look for common names.
            for name in ("advance", "update", "run_one_tick"):
                if hasattr(World, name):
                    World.tick_step = getattr(World, name)
                    return
            raise RuntimeError(
                "substrate.World has no tick_step/step/advance method"
            )


_ensure_tick_step()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Silence per-request logs; the UI polls ~10Hz.
        pass

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self._send(status, body, "application/json")

    def _send_file(self, path: str, content_type: str):
        try:
            with open(path, "rb") as f:
                body = f.read()
            self._send(200, body, content_type)
        except FileNotFoundError:
            self._send(404, b"not found", "text/plain")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/state":
            self._send_json(SIM.state())
            return
        if parsed.path == "/" or parsed.path == "/index.html":
            self._send_file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self._send_file(os.path.join(WEB_DIR, "app.js"), "application/javascript")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, status=400)
            return

        parsed = urlparse(self.path)
        if parsed.path == "/control":
            action = body.get("action", "")
            if action == "reset":
                variant = body.get("variant", "patchy")
                params = body.get("params", {})
                SIM.reset(variant=variant, params=params)
                self._send_json({"ok": True})
                return
            if action == "speed":
                SIM.set_speed(body.get("speed", 15.0))
                self._send_json({"ok": True})
                return
            SIM.control(action)
            self._send_json({"ok": True})
            return
        self._send(404, b"not found", "text/plain")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"genesis web ui: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.server_close()


if __name__ == "__main__":
    main()
