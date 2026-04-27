"""
Early-window stigmergy analysis.

Measures trace field spatial statistics and bias-corrected MI
at fine granularity (every 500 ticks) during the critical
window when stigmergy emerges (ticks 0–5000).

Also measures at coarse granularity (every 5k ticks) through
tick 50k to capture the full behavioral arc.

Usage:
    python3 early_window.py patchy 42       # single run
    python3 early_window.py patchy 42 50    # 50 runs for statistics
"""

import sys, os, json, time, math, random
from collections import Counter

from substrate import EAT, MOVE, FORK, LOAD, NUM_OPCODES
from patchy import PatchyWorld

def fine_census(world):
    """Census with trace spatial stats and bias-corrected MI."""
    # Save and restore RNG state to avoid contaminating simulation
    rng_state = random.getstate()
    
    c = {}

    # Population and genome stats
    pop = 0
    genome_lengths = []
    lbm_count = 0
    trace_action_pairs = []
    trace_values = []
    has_load_count = 0

    for r in range(world.size):
        for col in range(world.size):
            agent = world.grid[r][col].agent
            trace_values.append(world.grid[r][col].trace)

            if agent is None:
                continue
            pop += 1
            genome_lengths.append(len(agent.genome))

            ip = agent.ip % len(agent.genome)
            opcode = agent.genome[ip] % NUM_OPCODES

            # Check LOAD in genome
            if LOAD in [op % NUM_OPCODES for op in agent.genome]:
                has_load_count += 1

            # Action classification
            if opcode == EAT:
                action = 0
            elif opcode == MOVE:
                action = 1
            elif opcode == FORK:
                action = 2
            elif opcode == LOAD:
                is_lm = False
                for j in range(ip + 2, min(ip + 6, len(agent.genome)), 2):
                    if agent.genome[j] % NUM_OPCODES == MOVE:
                        is_lm = True
                        break
                action = 1 if is_lm else 3
                if is_lm:
                    lbm_count += 1
            else:
                action = 3

            # Trace-action pair for MI
            local_trace = world.grid[r][col].trace
            trace_bin = min(local_trace * 8 // 256, 7)
            trace_action_pairs.append((trace_bin, action))

    c['tick'] = world.tick
    c['population'] = pop
    c['avg_genome_length'] = sum(genome_lengths) / len(genome_lengths) if genome_lengths else 0
    c['load_before_move_frac'] = lbm_count / pop if pop > 0 else 0
    c['load_gene_freq'] = has_load_count / pop if pop > 0 else 0

    # Trace field stats
    n_cells = len(trace_values)
    mean_trace = sum(trace_values) / n_cells
    variance = sum((t - mean_trace) ** 2 for t in trace_values) / n_cells
    c['trace_mean'] = mean_trace
    c['trace_std'] = math.sqrt(variance)
    c['trace_cv'] = math.sqrt(variance) / mean_trace if mean_trace > 0 else 0
    c['trace_coverage'] = sum(1 for t in trace_values if t > 0) / n_cells
    c['trace_zeros'] = sum(1 for t in trace_values if t == 0)

    # Spatial heterogeneity: fraction of cells where trace > mean + 1 std
    threshold = mean_trace + math.sqrt(variance)
    c['trace_hotspot_frac'] = sum(1 for t in trace_values if t > threshold) / n_cells

    # Bias-corrected MI
    if len(trace_action_pairs) >= 20:
        traces = [t for t, a in trace_action_pairs]
        actions = [a for t, a in trace_action_pairs]

        def _plugin_mi(t_list, a_list):
            joint = Counter(zip(t_list, a_list))
            p_t = Counter(t_list)
            p_a = Counter(a_list)
            n = len(t_list)
            mi = 0.0
            for (t, a), count in joint.items():
                p_xy = count / n
                p_x = p_t[t] / n
                p_y = p_a[a] / n
                if p_xy > 0 and p_x > 0 and p_y > 0:
                    mi += p_xy * math.log2(p_xy / (p_x * p_y))
            return mi

        mi_raw = _plugin_mi(traces, actions)
        shuffled_mis = []
        for _ in range(100):
            sa = actions.copy()
            random.shuffle(sa)
            shuffled_mis.append(_plugin_mi(traces, sa))
        bias = sum(shuffled_mis) / len(shuffled_mis)

        c['mi_corrected'] = mi_raw - bias
        c['mi_raw'] = mi_raw
        c['mi_bias'] = bias
    else:
        c['mi_corrected'] = 0.0
        c['mi_raw'] = 0.0
        c['mi_bias'] = 0.0

    # Restore RNG state so bootstrap shuffles don't contaminate simulation
    random.setstate(rng_state)
    return c


def run_early_window(env='patchy', seed=42):
    """Run with fine-grained early census."""
    random.seed(seed)

    if env == 'patchy':
        world = PatchyWorld(patch_interval=3, food_per_patch=400, patch_radius=10)
    else:
        from substrate import World
        world = World()

    world.seed_population()

    data = []
    start = time.time()

    # Phase 1: every 500 ticks from 0 to 5000 (11 censuses)
    for i in range(5001):
        world.step()
        if i % 500 == 0:
            c = fine_census(world)
            elapsed = time.time() - start
            c['tps'] = (i + 1) / elapsed if elapsed > 0 else 0
            data.append(c)
            print(f"  tick={c['tick']:>6d}  pop={c['population']:>4d}  "
                  f"L→M={c['load_before_move_frac']:.3f}  "
                  f"trace_cov={c['trace_coverage']:.3f}  "
                  f"trace_cv={c['trace_cv']:.3f}  "
                  f"MI_corr={c['mi_corrected']:.4f}")

    # Phase 2: every 5000 ticks from 5k to 50k (9 censuses)
    for i in range(5001, 50001):
        world.step()
        if i % 5000 == 0:
            c = fine_census(world)
            elapsed = time.time() - start
            c['tps'] = (i + 1) / elapsed if elapsed > 0 else 0
            data.append(c)
            print(f"  tick={c['tick']:>6d}  pop={c['population']:>4d}  "
                  f"L→M={c['load_before_move_frac']:.3f}  "
                  f"trace_cov={c['trace_coverage']:.3f}  "
                  f"trace_cv={c['trace_cv']:.3f}  "
                  f"MI_corr={c['mi_corrected']:.4f}")

    elapsed = time.time() - start
    return {'seed': seed, 'env': env, 'census_data': data, 'elapsed_s': elapsed}


if __name__ == '__main__':
    env = sys.argv[1] if len(sys.argv) > 1 else 'patchy'
    seed_start = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    all_runs = []
    for s in range(n_seeds):
        seed = seed_start + s
        print(f"\n{'='*70}")
        print(f"  SEED {seed} ({s+1}/{n_seeds}), env={env}")
        print(f"{'='*70}")
        result = run_early_window(env, seed)
        all_runs.append(result)

    timestamp = int(time.time())
    outfile = f"early_window_{env}_n{n_seeds}_{timestamp}.json"
    outpath = os.path.join(os.path.dirname(__file__), 'runs', outfile)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(all_runs, f, indent=2)

    print(f"\nResults saved to {outpath}")
