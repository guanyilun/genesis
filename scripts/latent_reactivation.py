#!/usr/bin/env python3
"""
Latent Reactivation Experiment
================================

Hypothesis: Organisms from late-stage worlds (tick 150k+) retain LOAD→MOVE
patterns in their genomes even though they've stopped executing them (genome-vs-
execution decoupling). If transplanted into a fresh world with high trace
information, they should re-activate stigmergic behavior faster than organisms
starting from scratch.

Design:
  Phase 1: Run donor worlds to tick 150k (or configurable) — collect genomes
  Phase 2: Create transplant worlds seeded with those genomes into fresh worlds
  Phase 3: Create control worlds seeded with same number of random+LUCA agents
  Phase 4: Run both for 100k ticks, compare L→M emergence timing

The key prediction: transplant worlds reach L→M>0.1 earlier than controls.

Usage:
    python latent_reactivation.py run                    # Full experiment (slow)
    python latent_reactivation.py run --ticks-donor 100000 --ticks-transplant 80000 --n-replicates 5
    python latent_reactivation.py analyze results.json   # Analyze existing results

Based on ash 9c40a3's decoupling discovery:
  In the Normal condition, LOAD→MOVE genomic adjacency peaks at 0.78 (tick 30k)
  and collapses to 0.01 by tick 190k. But the *executed* LOAD opcode frequency
  drops to zero by tick 50k — while the genomic pattern is still at 0.38.
"""

import sys
import os
import json
import time
import random
import math
import argparse
from collections import Counter

# Add parent dir to path so we can import from genesis root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
from controlled_experiment import (
    compute_mutual_information_bias_corrected,
    WORLD_MAP, WORLD_KWARGS,
)


def mi_census(world):
    """Run deep_census plus mutual information measurement (with bias correction)."""
    c = deep_census(world)
    mi_corr, mi_raw, mi_bias = compute_mutual_information_bias_corrected(world)
    c['mutual_information'] = mi_corr
    c['mutual_information_raw'] = mi_raw
    c['mutual_information_bias'] = mi_bias
    return c


def extract_genomes(world):
    """Extract all living genomes from a world."""
    genomes = []
    for r in range(world.size):
        for c in range(world.size):
            agent = world.grid[r][c].agent
            if agent is not None and agent.alive:
                genomes.append(list(agent.genome))
    return genomes


def genome_load_move_fraction(genomes):
    """What fraction of genomes have LOAD within 3 positions before a MOVE?"""
    if not genomes:
        return 0.0
    count = 0
    for g in genomes:
        has_pattern = False
        for i in range(len(g)):
            if g[i] % NUM_OPCODES == MOVE:
                # Check if LOAD appears in prior 3 positions
                for j in range(max(0, i - 3), i):
                    if g[j] % NUM_OPCODES == LOAD:
                        has_pattern = True
                        break
                if has_pattern:
                    break
        if has_pattern:
            count += 1
    return count / len(genomes)


def seed_from_genomes(world, genomes, energy=INITIAL_ENERGY):
    """Place organisms with given genomes into a world at random positions."""
    random.shuffle(genomes)
    placed = 0
    for genome in genomes:
        agent = Agent(genome=list(genome), energy=energy)
        for _ in range(100):
            r = random.randint(0, world.size - 1)
            c = random.randint(0, world.size - 1)
            if world.grid[r][c].agent is None:
                world.place_agent(r, c, agent)
                placed += 1
                break
    return placed


def run_donor_phase(env='patchy', seed=42, ticks=150000, census_interval=10000):
    """Phase 1: Run a world to harvest late-stage genomes."""
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 1: DONOR WORLD — {env}, seed={seed}, ticks={ticks}", flush=True)
    print(f"{'='*70}", flush=True)

    random.seed(seed)
    world_class = WORLD_MAP[env]
    world_kwargs = WORLD_KWARGS[env]
    world = world_class(**world_kwargs)
    world.seed_population()

    census_data = []
    start = time.time()

    for i in range(ticks):
        world.step()

        if i % census_interval == 0:
            c = mi_census(world)
            c['tick'] = world.tick
            c['elapsed_s'] = time.time() - start
            census_data.append(c)
            print(f"  TICK {world.tick:>7d}  pop={c['population']:>5d}  "
                  f"L→M={c['load_before_move_frac']:.3f}  "
                  f"MI={c['mutual_information']:.4f}  "
                  f"trace_util={c['trace_utilization']:.3f}",
                  flush=True)

        if world.population == 0:
            print(f"  EXTINCTION at tick {world.tick}", flush=True)
            break

    # Extract genomes from surviving population
    genomes = extract_genomes(world)
    genomic_lm = genome_load_move_fraction(genomes)

    result = {
        'seed': seed,
        'env': env,
        'ticks_requested': ticks,
        'ticks_completed': world.tick,
        'census_data': census_data,
        'genomes_harvested': len(genomes),
        'genomic_lm_fraction': genomic_lm,
        'avg_genome_length': sum(len(g) for g in genomes) / len(genomes) if genomes else 0,
    }

    print(f"\n  Harvested {len(genomes)} genomes, genomic L→M = {genomic_lm:.3f}", flush=True)
    return result, genomes


def run_transplant_experiment(genomes, env='patchy', seed_offset=0,
                              ticks=100000, census_interval=5000,
                              label='transplant', fresh_traces=True):
    """Phase 2: Seed a fresh world with harvested genomes, run and measure."""
    random.seed(10000 + seed_offset)
    world_class = WORLD_MAP[env]
    world_kwargs = WORLD_KWARGS[env]
    world = world_class(**world_kwargs)

    # Seed world with transplanted genomes
    n_placed = seed_from_genomes(world, genomes)

    print(f"  [{label}] Placed {n_placed} organisms, running {ticks} ticks...", flush=True)

    census_data = []
    start = time.time()

    for i in range(ticks):
        world.step()

        if i % census_interval == 0:
            c = mi_census(world)
            c['tick'] = world.tick
            c['elapsed_s'] = time.time() - start
            census_data.append(c)

            if world.population == 0:
                print(f"  [{label}] EXTINCTION at tick {world.tick}", flush=True)
                break

    elapsed = time.time() - start
    result = {
        'seed_offset': seed_offset,
        'label': label,
        'ticks_requested': ticks,
        'ticks_completed': world.tick,
        'n_transplanted': n_placed,
        'elapsed_s': elapsed,
        'census_data': census_data,
    }
    return result


def run_control_experiment(n_agents, env='patchy', seed_offset=0,
                           ticks=100000, census_interval=5000,
                           label='control', luca_fraction=0.3):
    """Phase 3: Seed a fresh world with random+LUCA (no stigmergic memory)."""
    random.seed(20000 + seed_offset)
    world_class = WORLD_MAP[env]
    world_kwargs = WORLD_KWARGS[env]
    world = world_class(**world_kwargs)
    world.seed_population(count=n_agents, luca_fraction=luca_fraction)

    print(f"  [{label}] Seeded {n_agents} agents, running {ticks} ticks...", flush=True)

    census_data = []
    start = time.time()

    for i in range(ticks):
        world.step()

        if i % census_interval == 0:
            c = mi_census(world)
            c['tick'] = world.tick
            c['elapsed_s'] = time.time() - start
            census_data.append(c)

            if world.population == 0:
                print(f"  [{label}] EXTINCTION at tick {world.tick}", flush=True)
                break

    elapsed = time.time() - start
    result = {
        'seed_offset': seed_offset,
        'label': label,
        'ticks_requested': ticks,
        'ticks_completed': world.tick,
        'n_seeded': n_agents,
        'elapsed_s': elapsed,
        'census_data': census_data,
    }
    return result


def find_lm_emergence(census_data, threshold=0.10):
    """Find the first tick where load_before_move_frac exceeds threshold."""
    for c in census_data:
        if c.get('load_before_move_frac', 0) >= threshold:
            return c['tick']
    return None


def analyze_results(results):
    """Produce a statistical comparison of transplant vs control."""
    print("\n" + "=" * 70)
    print("  LATENT REACTIVATION — ANALYSIS")
    print("=" * 70)

    donor = results.get('donor', {})
    transplant_runs = results.get('transplants', [])
    control_runs = results.get('controls', [])

    print(f"\n  DONOR PHASE:")
    print(f"    Ticks completed: {donor.get('ticks_completed', 'N/A')}")
    print(f"    Genomes harvested: {donor.get('genomes_harvested', 'N/A')}")
    print(f"    Genomic L→M fraction: {donor.get('genomic_lm_fraction', 0):.3f}")
    print(f"    Avg genome length: {donor.get('avg_genome_length', 0):.1f}")

    # Find emergence times
    def extract_emergence(runs, label):
        emergence_times = []
        peak_lms = []
        final_lms = []
        for run in runs:
            cd = run.get('census_data', [])
            if cd:
                # First tick where L→M exceeds 0.10
                t = find_lm_emergence(cd, threshold=0.10)
                emergence_times.append(t)
                # Peak L→M
                lms = [c.get('load_before_move_frac', 0) for c in cd]
                peak_lms.append(max(lms) if lms else 0)
                final_lms.append(lms[-1] if lms else 0)
        return emergence_times, peak_lms, final_lms

    trans_emergence, trans_peaks, trans_finals = extract_emergence(transplant_runs, 'transplant')
    ctrl_emergence, ctrl_peaks, ctrl_finals = extract_emergence(control_runs, 'control')

    print(f"\n  TRANSPLANT (n={len(transplant_runs)}):")
    print(f"    L→M > 0.10 emergence ticks: {[t for t in trans_emergence if t is not None]}")
    print(f"    Emerged in {sum(1 for t in trans_emergence if t is not None)}/{len(trans_emergence)} runs")
    print(f"    Peak L→M: mean={sum(trans_peaks)/len(trans_peaks):.3f}" if trans_peaks else "    No data")
    print(f"    Final L→M: mean={sum(trans_finals)/len(trans_finals):.3f}" if trans_finals else "    No data")

    print(f"\n  CONTROL (n={len(control_runs)}):")
    print(f"    L→M > 0.10 emergence ticks: {[t for t in ctrl_emergence if t is not None]}")
    print(f"    Emerged in {sum(1 for t in ctrl_emergence if t is not None)}/{len(ctrl_emergence)} runs")
    print(f"    Peak L→M: mean={sum(ctrl_peaks)/len(ctrl_peaks):.3f}" if ctrl_peaks else "    No data")
    print(f"    Final L→M: mean={sum(ctrl_finals)/len(ctrl_finals):.3f}" if ctrl_finals else "    No data")

    # Statistical test (Mann-Whitney U) if we have enough data
    valid_trans = [t for t in trans_emergence if t is not None]
    valid_ctrl = [t for t in ctrl_emergence if t is not None]

    if len(valid_trans) >= 2 and len(valid_ctrl) >= 2:
        from scipy import stats
        try:
            u_stat, p_val = stats.mannwhitneyu(valid_trans, valid_ctrl, alternative='two-sided')
            print(f"\n  Mann-Whitney U test (emergence time):")
            print(f"    U={u_stat:.1f}, p={p_val:.4f}")
            if p_val < 0.05:
                if sum(valid_trans) / len(valid_trans) < sum(valid_ctrl) / len(valid_ctrl):
                    print(f"    *** TRANSPLANT emerges significantly EARLIER ***")
                else:
                    print(f"    *** CONTROL emerges significantly EARLIER ***")
            else:
                print(f"    No significant difference (p >= 0.05)")
        except Exception as e:
            print(f"\n  Statistical test failed: {e}")
    elif len(valid_trans) > 0 or len(valid_ctrl) > 0:
        print(f"\n  Too few runs for statistical test (need >= 2 per group)")

    # Also compare peak and final L→M
    if len(trans_peaks) >= 2 and len(ctrl_peaks) >= 2:
        from scipy import stats
        try:
            u_stat, p_val = stats.mannwhitneyu(trans_peaks, ctrl_peaks, alternative='two-sided')
            print(f"\n  Mann-Whitney U test (peak L→M):")
            print(f"    U={u_stat:.1f}, p={p_val:.4f}")
        except:
            pass

    print("\n" + "=" * 70)

    # Return structured data for visualization
    return {
        'transplant_emergence_mean': sum(t for t in trans_emergence if t is not None) / max(len([t for t in trans_emergence if t is not None]), 1),
        'control_emergence_mean': sum(t for t in ctrl_emergence if t is not None) / max(len([t for t in ctrl_emergence if t is not None]), 1),
        'transplant_peak_lm_mean': sum(trans_peaks) / len(trans_peaks) if trans_peaks else 0,
        'control_peak_lm_mean': sum(ctrl_peaks) / len(ctrl_peaks) if ctrl_peaks else 0,
        'transplant_final_lm_mean': sum(trans_finals) / len(trans_finals) if trans_finals else 0,
        'control_final_lm_mean': sum(ctrl_finals) / len(ctrl_finals) if ctrl_finals else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Latent Reactivation Experiment')
    parser.add_argument('mode', choices=['run', 'analyze'], help='Run experiment or analyze results')
    parser.add_argument('--donor-seed', type=int, default=42, help='Seed for donor world')
    parser.add_argument('--env', type=str, default='patchy', choices=['patchy', 'oasis', 'uniform'])
    parser.add_argument('--ticks-donor', type=int, default=150000, help='Ticks for donor phase')
    parser.add_argument('--ticks-transplant', type=int, default=100000, help='Ticks for transplant/control phase')
    parser.add_argument('--census-interval', type=int, default=5000, help='Census interval for transplant/control')
    parser.add_argument('--n-replicates', type=int, default=5, help='Number of transplant + control replicates')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')

    args = parser.parse_args()

    if args.mode == 'analyze':
        path = args.output or sys.argv[2] if len(sys.argv) > 2 else None
        if path is None:
            # Find most recent results file
            import glob
            files = sorted(glob.glob('runs/latent_reactivation_*.json'))
            if not files:
                print("No results files found")
                return
            path = files[-1]
            print(f"Analyzing: {path}")
        with open(path) as f:
            results = json.load(f)
        analyze_results(results)
        return

    output_path = args.output or f'runs/latent_reactivation_{int(time.time())}.json'

    print("=" * 70)
    print("  LATENT REACTIVATION EXPERIMENT")
    print("=" * 70)
    print(f"  Donor: {args.env}, seed={args.donor_seed}, {args.ticks_donor} ticks")
    print(f"  Transplant/Control: {args.n_replicates} replicates, {args.ticks_transplant} ticks each")
    print(f"  Output: {output_path}")
    print()

    # ── Phase 1: Run donor world ──
    donor_result, genomes = run_donor_phase(
        env=args.env, seed=args.donor_seed, ticks=args.ticks_donor
    )

    if not genomes:
        print("\n  *** DONOR WORLD WENT EXTINCT — cannot transplant ***")
        print("  Try a different seed or environment.")
        return

    n_transplant = min(len(genomes), 300)  # Cap at initial population
    # Subsample if we have more genomes than needed
    if len(genomes) > n_transplant:
        random.seed(42)
        genomes = random.sample(genomes, n_transplant)

    # ── Phase 2: Transplant replicates ──
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 2: TRANSPLANT REPLICATES — seeding with {len(genomes)} harvested genomes", flush=True)
    print(f"{'='*70}", flush=True)

    transplant_results = []
    for i in range(args.n_replicates):
        print(f"\n  --- Transplant replicate {i+1}/{args.n_replicates} ---", flush=True)
        r = run_transplant_experiment(
            genomes=genomes, env=args.env, seed_offset=i,
            ticks=args.ticks_transplant, census_interval=args.census_interval,
            label=f'transplant_{i}'
        )
        transplant_results.append(r)

        # Report
        if r['census_data']:
            last = r['census_data'][-1]
            print(f"  Result: pop={last['population']}, L→M={last['load_before_move_frac']:.3f}", flush=True)

    # ── Phase 3: Control replicates ──
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 3: CONTROL REPLICATES — seeding with {n_transplant} random+LUCA agents", flush=True)
    print(f"{'='*70}", flush=True)

    control_results = []
    for i in range(args.n_replicates):
        print(f"\n  --- Control replicate {i+1}/{args.n_replicates} ---", flush=True)
        r = run_control_experiment(
            n_agents=n_transplant, env=args.env, seed_offset=i,
            ticks=args.ticks_transplant, census_interval=args.census_interval,
            label=f'control_{i}'
        )
        control_results.append(r)

        if r['census_data']:
            last = r['census_data'][-1]
            print(f"  Result: pop={last['population']}, L→M={last['load_before_move_frac']:.3f}", flush=True)

    # ── Save results ──
    results = {
        'experiment': 'latent_reactivation',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'env': args.env,
            'donor_seed': args.donor_seed,
            'ticks_donor': args.ticks_donor,
            'ticks_transplant': args.ticks_transplant,
            'n_replicates': args.n_replicates,
            'census_interval': args.census_interval,
        },
        'donor': donor_result,
        'transplants': transplant_results,
        'controls': control_results,
    }

    # Remove genomes from donor_result (they're large lists — save separately if needed)
    # Actually keep them — they're the transplant material and valuable for analysis

    os.makedirs(os.path.dirname(output_path) or 'runs', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    # ── Analysis ──
    summary = analyze_results(results)

    results['summary'] = summary
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Summary updated in: {output_path}")


if __name__ == '__main__':
    main()