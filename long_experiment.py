"""
Long-duration experiment: does proto-stigmergy deepen into actual stigmergy?

Design:
  - Patchy world: 200k ticks, census every 10k
  - Oasis world: 200k ticks, census every 10k
  - Uniform world (control): 200k ticks, census every 10k

Measures beyond f19a29's:
  - Stigmergy gene frequency over time (LOAD, STORE, SEND, RECV)
  - LOAD-before-MOVE patterns (actual stigmergic navigation)
  - Species count and diversity over time
  - Parasite emergence (FORK-heavy, EAT-zero genomes)
  - Trace field utilization across the grid
  - Genome complexity trajectory

The question: at 12k ticks we saw 76.6% write / 14.5% read-branch.
At 200k ticks, does the read rate increase? Do organisms start
navigating based on trace values?
"""

import time
import json
import random
import math
from collections import Counter
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
from observatory import behavioral_profile, census, cluster_species, genome_distance


OPCODE_NAMES = {
    NOP: 'NOP', INC_R0: 'INC_R0', INC_R1: 'INC_R1', DEC_R0: 'DEC_R0',
    DEC_R1: 'DEC_R1', ADD_RR: 'ADD_RR', SUB_RR: 'SUB_RR', LOAD: 'LOAD',
    STORE: 'STORE', MOVE: 'MOVE', EAT: 'EAT', SHARE: 'SHARE', FORK: 'FORK',
    JMP: 'JMP', JZ: 'JZ', JNZ: 'JNZ', 0x10: 'CMPZ', RAND: 'RAND',
    SENSE: 'SENSE', DIE: 'DIE', SWAP: 'SWAP', SHL: 'SHL', SHR: 'SHR',
    AND_RR: 'AND_RR', OR_RR: 'OR_RR', XOR_RR: 'XOR_RR', SEND: 'SEND',
    RECV: 'RECV', SET_R0: 'SET_R0', SET_R1: 'SET_R1',
}


def deep_census(world: World) -> dict:
    """
    Extended census that measures stigmergy depth beyond gene presence.

    Key additions over observatory.census():
    - load_before_move: fraction of genomes where LOAD appears within
      3 instructions before a MOVE (stigmergic navigation)
    - store_after_computation: fraction where STORE follows arithmetic
      (meaningful trace writing, not just noise)
    - trace_utilization: fraction of grid cells with non-zero traces
    - directional_load_diversity: how many different directions LOAD reads from
    """
    profiles = []
    genome_lengths = []
    energies = []
    ages = []
    all_opcodes = Counter()

    load_before_move_count = 0
    store_after_compute_count = 0
    directional_load_diversity_sum = 0
    total_agents = 0

    for r in range(world.size):
        for c in range(world.size):
            agent = world.grid[r][c].agent
            if agent is None:
                continue
            total_agents += 1
            genome = agent.genome
            profiles.append(behavioral_profile(genome))
            genome_lengths.append(len(genome))
            energies.append(agent.energy)
            ages.append(agent.age)

            for g in genome:
                all_opcodes[g % NUM_OPCODES] += 1

            # Check LOAD-before-MOVE pattern
            has_load_before_move = False
            has_store_after_compute = False
            load_directions = set()

            for i in range(len(genome)):
                op = genome[i] % NUM_OPCODES
                if op == LOAD:
                    # Record the direction operand
                    if i + 1 < len(genome):
                        load_directions.add(genome[i + 1] % 4)
                    # Check if MOVE follows within 3 instructions
                    for j in range(i + 1, min(i + 4, len(genome))):
                        if genome[j] % NUM_OPCODES == MOVE:
                            has_load_before_move = True
                            break

                if op == STORE:
                    # Check if arithmetic immediately precedes STORE
                    if i > 0:
                        prev = genome[i - 1] % NUM_OPCODES
                        if prev in (ADD_RR, SUB_RR, SHL, SHR, AND_RR, OR_RR, XOR_RR,
                                    INC_R0, INC_R1, DEC_R0, DEC_R1):
                            has_store_after_compute = True

            if has_load_before_move:
                load_before_move_count += 1
            if has_store_after_compute:
                store_after_compute_count += 1
            directional_load_diversity_sum += len(load_directions)

    # Trace field utilization
    trace_cells = 0
    trace_values = []
    for r in range(world.size):
        for c in range(world.size):
            if world.grid[r][c].trace != 0:
                trace_cells += 1
                trace_values.append(world.grid[r][c].trace)

    total_cells = world.size * world.size

    # Stigmergy gene frequencies
    total_genes = sum(all_opcodes.values()) or 1
    stigmergy_freq = (all_opcodes.get(LOAD, 0) + all_opcodes.get(STORE, 0)) / total_genes
    comm_freq = (all_opcodes.get(SEND, 0) + all_opcodes.get(RECV, 0)) / total_genes
    load_freq = all_opcodes.get(LOAD, 0) / total_genes
    store_freq = all_opcodes.get(STORE, 0) / total_genes

    # Parasite detection
    parasite_count = 0
    for r in range(world.size):
        for c in range(world.size):
            agent = world.grid[r][c].agent
            if agent is None:
                continue
            gc = Counter(g % NUM_OPCODES for g in agent.genome)
            total = len(agent.genome)
            if total > 0 and gc.get(FORK, 0) / total > 0.15 and gc.get(EAT, 0) / total < 0.03:
                parasite_count += 1

    if total_agents == 0:
        return {'population': 0, 'tick': world.tick}

    return {
        'tick': world.tick,
        'population': total_agents,
        'avg_genome_length': sum(genome_lengths) / len(genome_lengths),
        'genome_length_stdev': math.sqrt(
            sum((l - sum(genome_lengths)/len(genome_lengths))**2 for l in genome_lengths)
            / len(genome_lengths)
        ) if len(genome_lengths) > 1 else 0,
        'avg_energy': sum(energies) / len(energies),
        'avg_age': sum(ages) / len(ages),

        # Stigmergy metrics
        'stigmergy_gene_freq': stigmergy_freq,
        'load_gene_freq': load_freq,
        'store_gene_freq': store_freq,
        'communication_gene_freq': comm_freq,
        'load_before_move_frac': load_before_move_count / total_agents,
        'store_after_compute_frac': store_after_compute_count / total_agents,
        'directional_load_diversity': directional_load_diversity_sum / total_agents,

        # Environment state
        'trace_utilization': trace_cells / total_cells,
        'avg_trace_value': sum(trace_values) / len(trace_values) if trace_values else 0,

        # Parasite pressure
        'parasite_count': parasite_count,
        'parasite_frac': parasite_count / total_agents,

        # Top opcodes
        'top_opcodes': sorted(
            [(OPCODE_NAMES.get(op, f'?{op}'), count / total_genes)
             for op, count in all_opcodes.items()],
            key=lambda x: -x[1]
        )[:8],

        # Average behavioral profile
        'avg_profile': {k: sum(p.get(k, 0) for p in profiles) / len(profiles)
                        for k in profiles[0]} if profiles else {},
    }


def run_experiment(world_class, world_kwargs, ticks=200000, census_interval=10000,
                   label="experiment"):
    """Run a single experiment with full data collection."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  {ticks} ticks, census every {census_interval}")
    print(f"{'='*80}")

    world = world_class(**world_kwargs)
    world.seed_population()

    data = []
    start = time.time()

    for i in range(ticks):
        world.step()

        if i % census_interval == 0:
            elapsed = time.time() - start
            tps = (i + 1) / elapsed if elapsed > 0 else 0
            c = deep_census(world)

            # Species count
            species = cluster_species(world, threshold=0.15)
            c['species_count'] = len(species)
            c['species_sizes'] = [len(s) for s in species[:10]]

            data.append(c)

            # Print progress
            print(f"\nTICK {world.tick:>7d}  ({elapsed:>5.0f}s, {tps:>4.0f} tps)")
            print(f"  pop={c['population']:>5d}  species={c['species_count']:>3d}  "
                  f"avg_glen={c['avg_genome_length']:.1f}±{c['genome_length_stdev']:.1f}")
            print(f"  stigmergy={c['stigmergy_gene_freq']:.3f}  "
                  f"load_before_move={c['load_before_move_frac']:.3f}  "
                  f"store_after_compute={c['store_after_compute_frac']:.3f}")
            print(f"  trace_util={c['trace_utilization']:.3f}  "
                  f"parasites={c['parasite_frac']:.3f}  "
                  f"comm={c['communication_gene_freq']:.3f}")
            print(f"  top: {' '.join(f'{n}={f:.1%}' for n, f in c['top_opcodes'][:5])}")

        if world.population == 0:
            print(f"\n  ⚠ EXTINCTION at tick {world.tick}")
            break

    elapsed = time.time() - start
    print(f"\n  Done: {elapsed:.0f}s, final tick {world.tick}")
    return data


def summarize_trajectory(data, label):
    """Print a summary of how metrics changed over the run."""
    if len(data) < 2:
        print(f"  {label}: insufficient data")
        return

    first = data[0]
    last = data[-1]

    def trend(key):
        vals = [d[key] for d in data if d.get('population', 1) > 0]
        if len(vals) < 3:
            return "insufficient"
        early = sum(vals[:len(vals)//3]) / (len(vals)//3)
        late = sum(vals[-len(vals)//3:]) / (len(vals)//3)
        if late > early * 1.2:
            return "↑"
        elif late < early * 0.8:
            return "↓"
        return "→"

    print(f"\n  {label} trajectory ({len(data)} censuses):")
    print(f"    Population: {first['population']} → {last['population']}  {trend('population')}")
    print(f"    Genome length: {first.get('avg_genome_length', 0):.1f} → {last.get('avg_genome_length', 0):.1f}  {trend('avg_genome_length')}")
    print(f"    Stigmergy genes: {first.get('stigmergy_gene_freq', 0):.3f} → {last.get('stigmergy_gene_freq', 0):.3f}  {trend('stigmergy_gene_freq')}")
    print(f"    LOAD→MOVE: {first.get('load_before_move_frac', 0):.3f} → {last.get('load_before_move_frac', 0):.3f}  {trend('load_before_move_frac')}")
    print(f"    STORE→compute: {first.get('store_after_compute_frac', 0):.3f} → {last.get('store_after_compute_frac', 0):.3f}  {trend('store_after_compute_frac')}")
    print(f"    Trace utilization: {first.get('trace_utilization', 0):.3f} → {last.get('trace_utilization', 0):.3f}  {trend('trace_utilization')}")
    print(f"    Species count: {first.get('species_count', 0)} → {last.get('species_count', 0)}  {trend('species_count')}")
    print(f"    Parasites: {first.get('parasite_frac', 0):.3f} → {last.get('parasite_frac', 0):.3f}  {trend('parasite_frac')}")


if __name__ == '__main__':
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    census_every = int(sys.argv[3]) if len(sys.argv) > 3 else 10000

    results = {}

    if mode in ("patchy", "all"):
        data = run_experiment(
            PatchyWorld,
            dict(patch_interval=3, food_per_patch=400, patch_radius=10),
            ticks=ticks, census_interval=census_every,
            label="PATCHY WORLD"
        )
        summarize_trajectory(data, "Patchy")
        results['patchy'] = data

    if mode in ("oasis", "all"):
        data = run_experiment(
            OasisWorld,
            dict(num_oases=5, oasis_radius=6, refill_interval=100, refill_amount=800),
            ticks=ticks, census_interval=census_every,
            label="OASIS WORLD"
        )
        summarize_trajectory(data, "Oasis")
        results['oasis'] = data

    if mode in ("uniform", "all"):
        data = run_experiment(
            World,
            dict(),
            ticks=ticks, census_interval=census_every,
            label="UNIFORM WORLD (control)"
        )
        summarize_trajectory(data, "Uniform")
        results['uniform'] = data

    # Save results
    out_path = f"runs/experiment_{int(time.time())}.json"
    # Convert tuples to lists for JSON serialization
    for key in results:
        for d in results[key]:
            if 'species_sizes' in d:
                d['species_sizes'] = list(d['species_sizes'])
            if 'top_opcodes' in d:
                d['top_opcodes'] = [list(x) for x in d['top_opcodes']]
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
