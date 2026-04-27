"""
Controlled experiments for the genesis substrate.

What's new here (vs long_experiment.py which just observed):
  1. Fixed RNG seeding — reproducible runs
  2. Three falsification controls:
     - Scrambled-trace: randomize all traces each tick
     - Trace-knockout: STORE/LOAD become NOP
     - Opcode-frequency null: compare observed vs random baseline
  3. Mutual information I(trace_value; next_action) — the publishable metric
  4. Multi-seed replicate runner

Usage:
    python controlled_experiment.py single oasis 42              # one run, seed 42
    python controlled_experiment.py control scramble oasis 42     # scrambled control
    python controlled_experiment.py control knockout oasis 42     # knockout control
    python controlled_experiment.py replicates oasis 30 200000    # 30 seeds
    python controlled_experiment.py mi oasis 42                   # mutual info measurement
"""

import sys
import os
import json
import time
import math
import random
from collections import Counter, defaultdict
from itertools import product

from substrate import (
    World, Agent, Cell, DIRECTIONS, NUM_OPCODES, HAS_OPERAND,
    WORLD_SIZE, MAX_GENOME, MIN_GENOME, INITIAL_ENERGY,
    TICK_ENERGY_COST, MOVE_ENERGY_COST, FORK_ENERGY_COST,
    EAT_YIELD, MAX_AGE, MUTATION_RATE, INITIAL_POPULATION,
    NOP, LOAD, STORE, MOVE, EAT, FORK, JMP, JZ, JNZ,
    SENSE, SEND, RECV, SHARE, DIE,
    INC_R0, INC_R1, DEC_R0, DEC_R1, ADD_RR, SUB_RR,
    SHL, SHR, AND_RR, OR_RR, XOR_RR, RAND, SWAP, SET_R0, SET_R1,
)
from patchy import PatchyWorld, OasisWorld
from long_experiment import deep_census


# ─── Control 1: Scrambled-trace world ────────────────────────────────

class ScrambledTraceWorld(type('mixin', (), {})):
    """
    Mixin that scrambles all trace values to random bytes each tick.
    If organisms still survive and LOAD→MOVE still emerges, the traces
    weren't carrying meaningful information.
    """
    def step(self):
        super().step()
        # Overwrite every cell's trace with a random value
        for r in range(self.size):
            for c in range(self.size):
                self.grid[r][c].trace = random.randint(0, 255)


class ScrambledPatchyWorld(ScrambledTraceWorld, PatchyWorld):
    pass

class ScrambledOasisWorld(ScrambledTraceWorld, OasisWorld):
    pass

class ScrambledUniformWorld(ScrambledTraceWorld, World):
    pass


# ─── Control 2: Trace-knockout world ────────────────────────────────

class KnockoutWorld(World):
    """LOAD and STORE compile to NOP. The stigmergy channel is dead."""
    def _execute_agent(self, r, c, agent):
        # Intercept before parent execution: if next opcode is LOAD/STORE, skip it
        if not agent.alive:
            return
        ip = agent.ip % len(agent.genome)
        opcode = agent.genome[ip] % NUM_OPCODES
        # Replace LOAD/STORE with NOP, consume operand if needed
        if opcode in (LOAD, STORE):
            agent.genome_temp = agent.genome[ip]  # save for metrics
            agent.ip += 1
            if opcode in HAS_OPERAND:
                agent.ip += 1
            agent.age += 1
            agent.energy -= TICK_ENERGY_COST
            if agent.energy <= 0 or agent.age >= MAX_AGE:
                self.remove_agent(r, c)
            return
        super()._execute_agent(r, c, agent)


class KnockoutPatchyWorld(KnockoutWorld, PatchyWorld):
    """Patchy world with stigmergy channel disabled."""
    def step(self):
        PatchyWorld.step(self)
        # KnockoutWorld._execute_agent will handle the NOP substitution

    def _execute_agent(self, r, c, agent):
        KnockoutWorld._execute_agent(self, r, c, agent)


class KnockoutOasisWorld(KnockoutWorld, OasisWorld):
    def step(self):
        OasisWorld.step(self)

    def _execute_agent(self, r, c, agent):
        KnockoutWorld._execute_agent(self, r, c, agent)


# ─── Mutual Information ──────────────────────────────────────────────

def compute_mutual_information(world):
    """
    Compute I(trace_value_at_cell; action_of_next_agent_to_enter).

    We record (trace_before, action_after) pairs for every cell transition
    where an agent moves into a cell. Actions are discretized:
      0 = EAT, 1 = MOVE, 2 = FORK, 3 = other.

    Uses the empirical joint distribution and the plug-in MI estimator.
    """
    trace_bins = 8    # quantize trace to 8 bins
    action_bins = 4

    # Collect joint counts from the current world state
    # Strategy: scan grid, for each agent that recently moved into a cell
    # (identified by direction != default), record the trace it would have read
    #
    # Simpler approach: for every living agent, record what trace it sees
    # in its neighborhood vs what action it takes next

    # We'll do a prospective measurement: for each agent, record the trace
    # value at its current position and its next opcode
    trace_action_pairs = []

    for r in range(world.size):
        for c in range(world.size):
            agent = world.grid[r][c].agent
            if agent is None:
                continue

            # The trace the agent is currently standing on
            local_trace = world.grid[r][c].trace
            trace_bin = min(local_trace * trace_bins // 256, trace_bins - 1)

            # The agent's next instruction
            ip = agent.ip % len(agent.genome)
            opcode = agent.genome[ip] % NUM_OPCODES

            if opcode == EAT:
                action = 0
            elif opcode == MOVE:
                action = 1
            elif opcode == FORK:
                action = 2
            elif opcode == LOAD:
                # LOAD → check if followed by MOVE (stigmergic navigation)
                for j in range(ip + 2, min(ip + 6, len(agent.genome)), 2):
                    if agent.genome[j] % NUM_OPCODES == MOVE:
                        action = 1  # count LOAD→MOVE as MOVE (navigation)
                        break
                else:
                    action = 3
            else:
                action = 3

            trace_action_pairs.append((trace_bin, action))

    if len(trace_action_pairs) < 10:
        return 0.0

    n = len(trace_action_pairs)

    # Joint distribution
    joint = Counter(trace_action_pairs)
    p_trace = Counter(t for t, a in trace_action_pairs)
    p_action = Counter(a for t, a in trace_action_pairs)

    # MI = sum p(x,y) * log2(p(x,y) / (p(x) * p(y)))
    mi = 0.0
    for (t, a), count in joint.items():
        p_xy = count / n
        p_x = p_trace[t] / n
        p_y = p_action[a] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))

    return mi


def compute_mutual_information_bias_corrected(world, n_shuffles=100):
    """
    Bootstrap bias-corrected MI.
    
    Compute MI_observed, then shuffle action labels n_shuffles times,
    compute MI for each shuffle, and return MI_observed - mean(MI_shuffled).
    
    If the result is near zero, the observed MI is entirely bias.
    """
    # Save RNG state to avoid contaminating simulation with bootstrap shuffles
    rng_state = random.getstate()
    
    trace_bins = 8
    action_bins = 4

    trace_action_pairs = []

    for r in range(world.size):
        for c in range(world.size):
            agent = world.grid[r][c].agent
            if agent is None:
                continue

            local_trace = world.grid[r][c].trace
            trace_bin = min(local_trace * trace_bins // 256, trace_bins - 1)

            ip = agent.ip % len(agent.genome)
            opcode = agent.genome[ip] % NUM_OPCODES

            if opcode == EAT:
                action = 0
            elif opcode == MOVE:
                action = 1
            elif opcode == FORK:
                action = 2
            elif opcode == LOAD:
                for j in range(ip + 2, min(ip + 6, len(agent.genome)), 2):
                    if agent.genome[j] % NUM_OPCODES == MOVE:
                        action = 1
                        break
                else:
                    action = 3
            else:
                action = 3

            trace_action_pairs.append((trace_bin, action))

    if len(trace_action_pairs) < 10:
        return 0.0, 0.0, 0.0

    traces = [t for t, a in trace_action_pairs]
    actions = [a for t, a in trace_action_pairs]
    n = len(traces)

    def _plugin_mi(t_list, a_list):
        joint = Counter(zip(t_list, a_list))
        p_t = Counter(t_list)
        p_a = Counter(a_list)
        nn = len(t_list)
        mi = 0.0
        for (t, a), count in joint.items():
            p_xy = count / nn
            p_x = p_t[t] / nn
            p_y = p_a[a] / nn
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
        return mi

    mi_observed = _plugin_mi(traces, actions)

    # Bootstrap: shuffle actions to break any real dependence
    shuffled_mis = []
    for _ in range(n_shuffles):
        shuffled_a = actions.copy()
        random.shuffle(shuffled_a)
        shuffled_mis.append(_plugin_mi(traces, shuffled_a))

    bias = sum(shuffled_mis) / len(shuffled_mis)
    mi_corrected = mi_observed - bias

    # Restore RNG state
    random.setstate(rng_state)
    return mi_corrected, mi_observed, bias


def mi_census(world):
    """Run deep_census plus mutual information measurement (with bias correction)."""
    c = deep_census(world)
    mi_corr, mi_raw, mi_bias = compute_mutual_information_bias_corrected(world)
    c['mutual_information'] = mi_corr       # bias-corrected (the one to use)
    c['mutual_information_raw'] = mi_raw    # uncorrected
    c['mutual_information_bias'] = mi_bias  # estimated bias
    return c


# ─── Seeded run ──────────────────────────────────────────────────────

WORLD_MAP = {
    'uniform': World,
    'patchy': PatchyWorld,
    'oasis': OasisWorld,
}

WORLD_KWARGS = {
    'uniform': dict(),
    'patchy': dict(patch_interval=3, food_per_patch=400, patch_radius=10),
    'oasis': dict(num_oases=5, oasis_radius=6, refill_interval=100, refill_amount=800),
}

SCRAMBLED_MAP = {
    'uniform': ScrambledUniformWorld,
    'patchy': ScrambledPatchyWorld,
    'oasis': ScrambledOasisWorld,
}

KNOCKOUT_MAP = {
    'uniform': KnockoutPatchyWorld,  # reuses KnockoutWorld logic
    'patchy': KnockoutPatchyWorld,
    'oasis': KnockoutOasisWorld,
}


def seeded_run(world_class, world_kwargs, seed, ticks=200000, census_interval=10000,
               label="run", measure_mi=True):
    """Run a single experiment with a fixed RNG seed."""
    random.seed(seed)

    # For oasis, we need smaller grid
    world = world_class(**world_kwargs)
    world.seed_population()

    data = []
    start = time.time()

    for i in range(ticks):
        world.step()

        if i % census_interval == 0:
            elapsed = time.time() - start
            tps = (i + 1) / elapsed if elapsed > 0 else 0

            if measure_mi:
                c = mi_census(world)
            else:
                c = deep_census(world)

            c['tick'] = world.tick
            c['elapsed_s'] = elapsed
            c['tps'] = tps
            c['seed'] = seed
            data.append(c)

            print(f"TICK {world.tick:>7d}  ({elapsed:>5.0f}s, {tps:>4.0f} tps)  "
                  f"pop={c['population']:>5d}  "
                  f"load→move={c['load_before_move_frac']:.3f}  "
                  f"MI={c.get('mutual_information', -1):.4f}  "
                  f"trace_util={c['trace_utilization']:.3f}",
                  flush=True)

        if world.population == 0:
            print(f"  EXTINCTION at tick {world.tick}", flush=True)
            break

    elapsed = time.time() - start
    result = {
        'seed': seed,
        'label': label,
        'ticks_requested': ticks,
        'ticks_completed': world.tick,
        'elapsed_s': elapsed,
        'census_data': data,
    }
    return result


def run_replicates(env, n_replicates, ticks=200000, census_interval=10000):
    """Run N replicates of a given environment with different seeds."""
    results = []
    for i in range(n_replicates):
        seed = 1000 + i  # deterministic seed sequence
        label = f"{env}_seed{seed}"
        print(f"\n{'='*70}", flush=True)
        print(f"  REPLICATE {i+1}/{n_replicates}: {label}", flush=True)
        print(f"{'='*70}", flush=True)

        r = seeded_run(
            WORLD_MAP[env], WORLD_KWARGS[env],
            seed=seed, ticks=ticks, census_interval=census_interval,
            label=label
        )
        results.append(r)

    # Summary stats across replicates
    print(f"\n{'='*70}", flush=True)
    print(f"  REPLICATE SUMMARY: {env}, {n_replicates} runs", flush=True)
    print(f"{'='*70}", flush=True)

    final_pops = []
    final_load_move = []
    final_mi = []
    for r in results:
        if r['census_data']:
            last = r['census_data'][-1]
            final_pops.append(last.get('population', 0))
            final_load_move.append(last.get('load_before_move_frac', 0))
            final_mi.append(last.get('mutual_information', 0))

    def stats(vals):
        if not vals:
            return "N/A"
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((v - m)**2 for v in vals) / len(vals)) if len(vals) > 1 else 0
        return f"{m:.3f} ± {s:.3f}"

    print(f"  Final population: {stats(final_pops)}", flush=True)
    print(f"  Final LOAD→MOVE: {stats(final_load_move)}", flush=True)
    print(f"  Final MI: {stats(final_mi)}", flush=True)

    return results


def run_control_comparison(env, seed, ticks=200000, census_interval=10000):
    """Run normal + scrambled + knockout with same seed, compare."""
    print(f"\n{'#'*70}", flush=True)
    print(f"  CONTROL COMPARISON: {env}, seed={seed}", flush=True)
    print(f"{'#'*70}", flush=True)

    # Normal run
    print(f"\n--- NORMAL ---", flush=True)
    normal = seeded_run(WORLD_MAP[env], WORLD_KWARGS[env], seed, ticks, census_interval,
                        label=f"{env}_normal")

    # Scrambled control
    print(f"\n--- SCRAMBLED TRACE ---", flush=True)
    random.seed(seed)  # reset before creating world
    scrambled = seeded_run(SCRAMBLED_MAP[env], WORLD_KWARGS[env], seed, ticks, census_interval,
                           label=f"{env}_scrambled")

    # Knockout control
    print(f"\n--- TRACE KNOCKOUT ---", flush=True)
    random.seed(seed)
    knockout = seeded_run(KNOCKOUT_MAP[env], WORLD_KWARGS[env], seed, ticks, census_interval,
                          label=f"{env}_knockout")

    # Comparison
    print(f"\n{'='*70}", flush=True)
    print(f"  CONTROL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    for name, result in [("Normal", normal), ("Scrambled", scrambled), ("Knockout", knockout)]:
        if result['census_data']:
            last = result['census_data'][-1]
            print(f"  {name:12s}: pop={last.get('population',0):>5d}  "
                  f"LOAD→MOVE={last.get('load_before_move_frac',0):.3f}  "
                  f"MI={last.get('mutual_information',0):.4f}  "
                  f"tick={result['ticks_completed']}", flush=True)
        else:
            print(f"  {name:12s}: no data", flush=True)

    return {'normal': normal, 'scrambled': scrambled, 'knockout': knockout}


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else "help"

    if mode == "single":
        env = sys.argv[2] if len(sys.argv) > 2 else "oasis"
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
        ticks = int(sys.argv[4]) if len(sys.argv) > 4 else 200000
        result = seeded_run(WORLD_MAP[env], WORLD_KWARGS[env], seed, ticks,
                            label=f"{env}_seed{seed}")
        out = f"runs/controlled_{env}_seed{seed}_{int(time.time())}.json"
        with open(out, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to {out}")

    elif mode == "control":
        ctrl_type = sys.argv[2]  # scramble or knockout
        env = sys.argv[3] if len(sys.argv) > 3 else "oasis"
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
        ticks = int(sys.argv[5]) if len(sys.argv) > 5 else 200000

        if ctrl_type == "compare":
            result = run_control_comparison(env, seed, ticks)
        elif ctrl_type == "scramble":
            result = seeded_run(SCRAMBLED_MAP[env], WORLD_KWARGS[env], seed, ticks,
                                label=f"{env}_scrambled")
        elif ctrl_type == "knockout":
            result = seeded_run(KNOCKOUT_MAP[env], WORLD_KWARGS[env], seed, ticks,
                                label=f"{env}_knockout")
        else:
            print(f"Unknown control type: {ctrl_type}")
            sys.exit(1)

        out = f"runs/controlled_{ctrl_type}_{env}_seed{seed}_{int(time.time())}.json"
        with open(out, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to {out}")

    elif mode == "replicates":
        env = sys.argv[2] if len(sys.argv) > 2 else "oasis"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        ticks = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
        results = run_replicates(env, n, ticks)
        out = f"runs/replicates_{env}_n{n}_{int(time.time())}.json"
        with open(out, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to {out}")

    elif mode == "mi":
        env = sys.argv[2] if len(sys.argv) > 2 else "oasis"
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
        ticks = int(sys.argv[4]) if len(sys.argv) > 4 else 200000
        result = seeded_run(WORLD_MAP[env], WORLD_KWARGS[env], seed, ticks,
                            label=f"{env}_mi", measure_mi=True)
        out = f"runs/mi_{env}_seed{seed}_{int(time.time())}.json"
        with open(out, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to {out}")

    else:
        print(__doc__)
