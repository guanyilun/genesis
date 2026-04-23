// genesis web UI — single-page client
// Polls /state ~10Hz, renders the grid to <canvas>, and sends
// /control POSTs when the user toggles play / resets / tweaks speed.

const canvas = document.getElementById("world");
const ctx = canvas.getContext("2d");

const CANVAS_PX = 640;

const OPCODE_NAMES = [
  "NOP", "MOVR0", "MOVR1", "MOV01", "MOV10",
  "ADD", "SUB", "LOAD", "STORE", "MOVE",
  "EAT", "SHARE", "FORK", "JMP", "JZ",
  "JNZ", "CMPZ", "RAND", "SENSE", "DIE",
  "SWAP", "SHL", "SHR", "AND", "OR",
  "XOR", "SEND", "RECV", "op28", "op29",
];

let latestState = null;
let overlays = { food: true, traces: true, agents: true };
// Coloring mode: "tui" matches run.py --gui (all agents same hue, energy → brightness).
// "lineage" hashes genome head bytes to HSL so clades are visible as color regions.
let colorMode = "tui";

// ── Rendering ──────────────────────────────────────────────────────

function render() {
  if (!latestState || !latestState.size) return;
  const size = latestState.size;
  const cellPx = CANVAS_PX / size;

  // Clear
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, CANVAS_PX, CANVAS_PX);

  // Traces (stigmergy) — faint amber heatmap, dim background
  if (overlays.traces && latestState.traces) {
    for (const [x, y, value] of latestState.traces) {
      const alpha = Math.min(0.6, 0.12 + (value / 255) * 0.5);
      ctx.fillStyle = `rgba(212, 144, 70, ${alpha})`;
      ctx.fillRect(x * cellPx, y * cellPx, cellPx, cellPx);
    }
  }

  // Food — green dots, energy → brightness
  if (overlays.food && latestState.food) {
    for (const [x, y, value] of latestState.food) {
      const bright = Math.min(1, 0.3 + value / 200);
      ctx.fillStyle = `rgba(90, 180, 110, ${bright})`;
      ctx.fillRect(x * cellPx + cellPx * 0.2, y * cellPx + cellPx * 0.2, cellPx * 0.6, cellPx * 0.6);
    }
  }

  // Agents — TUI mode: single cyan-ish hue, brightness → energy (matches
  // run.py where bold ≈ high energy). Lineage mode: hash-derived hue so
  // clades show as color regions (useful once selection has run a while).
  if (overlays.agents && latestState.agents) {
    for (const [x, y, hue, energy] of latestState.agents) {
      const light = Math.max(35, Math.min(75, 35 + energy / 10));
      if (colorMode === "tui") {
        ctx.fillStyle = `hsl(180, 65%, ${light}%)`;
      } else {
        ctx.fillStyle = `hsl(${hue}, 80%, ${light}%)`;
      }
      ctx.fillRect(x * cellPx, y * cellPx, cellPx, cellPx);
    }
  }
}

// ── Sidebar ────────────────────────────────────────────────────────

function updateSidebar() {
  if (!latestState) return;
  document.getElementById("s-tick").textContent = latestState.tick.toLocaleString();
  document.getElementById("s-pop").textContent = latestState.census.population;
  document.getElementById("s-genome").textContent = latestState.census.avg_genome_len;
  document.getElementById("s-food").textContent = latestState.food?.length ?? 0;
  document.getElementById("s-traces").textContent = latestState.traces?.length ?? 0;

  const opcodes = latestState.census.opcodes || [];
  const total = opcodes.reduce((s, n) => s + n, 0) || 1;
  const pairs = opcodes.map((count, i) => [i, count])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  const rootEl = document.getElementById("opcodes");
  rootEl.innerHTML = "";
  const maxCount = pairs[0]?.[1] || 1;
  for (const [op, count] of pairs) {
    if (count === 0) continue;
    const row = document.createElement("div");
    row.className = "opcode-bar";
    const pct = ((count / total) * 100).toFixed(1);
    const widthPct = (count / maxCount) * 100;
    row.innerHTML = `
      <span class="name">${OPCODE_NAMES[op] || "op" + op}</span>
      <span class="bar"><span class="fill" style="width:${widthPct}%"></span></span>
      <span class="pct">${pct}%</span>
    `;
    rootEl.appendChild(row);
  }
}

// ── Polling ────────────────────────────────────────────────────────

async function pollState() {
  try {
    const resp = await fetch("/state");
    latestState = await resp.json();
    render();
    updateSidebar();
  } catch (e) {
    // Swallow — server may be restarting; next poll will recover.
  }
}

setInterval(pollState, 100);
pollState();

// ── Controls ───────────────────────────────────────────────────────

function post(action, extra = {}) {
  return fetch("/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, ...extra }),
  });
}

document.getElementById("btn-play").onclick = () => post("play");
document.getElementById("btn-pause").onclick = () => post("pause");
document.getElementById("btn-step").onclick = () => post("step");

document.getElementById("speed").oninput = (e) => {
  const v = parseInt(e.target.value, 10);
  document.getElementById("speed-val").textContent = v + "/s";
  post("speed", { speed: v });
};

// Param sliders live-update their labels; values only apply on Reset.
function bindSlider(id, labelId, fmt) {
  const el = document.getElementById(id);
  const out = document.getElementById(labelId);
  el.oninput = () => { out.textContent = fmt(el.value); };
  out.textContent = fmt(el.value);
}
bindSlider("p-luca", "p-luca-val", v => Number(v).toFixed(2));
bindSlider("p-mut", "p-mut-val", v => Number(v).toFixed(3));
bindSlider("p-food", "p-food-val", v => Number(v).toFixed(3));

document.getElementById("btn-reset").onclick = () => {
  const variant = document.getElementById("variant").value;
  const params = {
    WORLD_SIZE: parseInt(document.getElementById("p-size").value, 10),
    INITIAL_POPULATION: parseInt(document.getElementById("p-pop").value, 10),
    LUCA_FRACTION: parseFloat(document.getElementById("p-luca").value),
    MUTATION_RATE: parseFloat(document.getElementById("p-mut").value),
    FOOD_RATE: parseFloat(document.getElementById("p-food").value),
  };
  post("reset", { variant, params });
};

// Overlay toggles — render side only, no server roundtrip.
function bindOverlay(id, key) {
  document.getElementById(id).onchange = (e) => {
    overlays[key] = e.target.checked;
    render();
  };
}
bindOverlay("ov-food", "food");
bindOverlay("ov-traces", "traces");
bindOverlay("ov-agents", "agents");

document.getElementById("color-mode").onchange = (e) => {
  colorMode = e.target.value;
  render();
};
