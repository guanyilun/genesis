"""
The observatory — tools to analyze what's actually happening inside the ecosystem.

Not just "how many agents" but "what are they doing?" Questions this answers:

1. What opcodes are present in living genomes? Is anything using STORE/LOAD/SEND/RECV?
2. Are there distinct species — clusters of similar genomes separated by large gaps?
3. What's the effective strategy of a genome? (eat-heavy? fork-heavy? nomadic?)
4. Genome diversity — are we seeing convergence to one strategy or radiation?

The approach: extract behavioral fingerprints from genomes. A fingerprint is
a fixed-length vector encoding: opcode frequency, genome length, and structural
features (loop depth, operand diversity). Then cluster by distance.
"""

import random
import math
from collections import Counter
from substrate import NUM_OPCODES, HAS_OPERAND, World


def opcode_frequencies(genome: list[int]) -> list[float]:
    """Normalized frequency of each opcode in a genome."""
    counts = Counter(g % NUM_OPCODES for g in genome)
    total = len(genome) if genome else 1
    return [counts.get(i, 0) / total for i in range(NUM_OPCODES)]


def behavioral_profile(genome: list[int]) -> dict:
    """
    Extract a behavioral profile from a genome — what strategies does it encode?
    Not a simulation, just static analysis of instruction composition.
    """
    counts = Counter(g % NUM_OPCODES for g in genome)
    length = len(genome)
    if length == 0:
        return {}

    # Strategy vectors
    foraging = (counts.get(0x0A, 0) + counts.get(0x12, 0)) / length  # EAT + SENSE
    movement = counts.get(0x09, 0) / length  # MOVE
    reproduction = counts.get(0x0C, 0) / length  # FORK
    communication = (counts.get(0x1A, 0) + counts.get(0x1B, 0)) / length  # SEND + RECV
    stigmergy = (counts.get(0x07, 0) + counts.get(0x08, 0)) / length  # LOAD + STORE
    computation = (counts.get(0x01, 0) + counts.get(0x02, 0) + counts.get(0x03, 0) +
                   counts.get(0x04, 0) + counts.get(0x05, 0) + counts.get(0x06, 0) +
                   counts.get(0x15, 0) + counts.get(0x16, 0) +
                   counts.get(0x17, 0) + counts.get(0x18, 0) + counts.get(0x19, 0)) / length
    control_flow = (counts.get(0x0D, 0) + counts.get(0x0E, 0) + counts.get(0x0F, 0)) / length  # JMP, JZ, JNZ
    sharing = counts.get(0x0B, 0) / length  # SHARE
    death = counts.get(0x13, 0) / length  # DIE

    # Operand diversity — how many different directions/values appear as operands
    operands = set()
    for i, g in enumerate(genome):
        op = g % NUM_OPCODES
        if op in HAS_OPERAND and i + 1 < len(genome):
            operands.add(genome[i + 1] % 4)
    operand_diversity = len(operands) / 4.0

    # Loop density — JMP backwards divided by genome length
    backward_jumps = 0
    for i, g in enumerate(genome):
        op = g % NUM_OPCODES
        if op in (0x0D, 0x0E, 0x0F) and i + 1 < len(genome):
            target = genome[i + 1] % length
            if target <= i:
                backward_jumps += 1

    return {
        'length': length,
        'foraging': foraging,
        'movement': movement,
        'reproduction': reproduction,
        'communication': communication,
        'stigmergy': stigmergy,
        'computation': computation,
        'control_flow': control_flow,
        'sharing': sharing,
        'death': death,
        'operand_diversity': operand_diversity,
        'loop_density': backward_jumps / length if length else 0,
    }


def genome_distance(g1: list[int], g2: list[int]) -> float:
    """
    Distance between two genomes based on behavioral fingerprint.
    Returns 0.0 (identical strategy) to 1.0 (completely different).
    """
    p1 = behavioral_profile(g1)
    p2 = behavioral_profile(g2)

    keys = ['foraging', 'movement', 'reproduction', 'communication', 'stigmergy',
            'computation', 'control_flow', 'sharing', 'operand_diversity', 'loop_density']

    dist = 0
    for k in keys:
        dist += (p1.get(k, 0) - p2.get(k, 0)) ** 2

    # Add genome length difference (normalized)
    len_diff = abs(p1.get('length', 1) - p2.get('length', 1)) / 64.0
    dist += len_diff ** 2

    return math.sqrt(dist) / math.sqrt(len(keys) + 1)


def cluster_species(world: World, threshold: float = 0.15) -> list[list[tuple[int, int, list[int]]]]:
    """
    Cluster living agents into species by genome distance.
    Simple single-linkage clustering: if genome distance < threshold, same species.
    Returns list of species, each a list of (row, col, genome) tuples.
    """
    # Collect all living agents
    agents = []
    for r in range(world.size):
        for c in range(world.size):
            agent = world.grid[r][c].agent
            if agent is not None:
                agents.append((r, c, agent.genome))

    if not agents:
        return []

    # Single-linkage clustering
    n = len(agents)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if genome_distance(agents[i][2], agents[j][2]) < threshold:
                union(i, j)

    # Group by cluster
    clusters = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(agents[i])

    # Sort by size descending
    species_list = sorted(clusters.values(), key=lambda s: len(s), reverse=True)
    return species_list


def census(world: World) -> dict:
    """
    Full census of the living population.
    Returns aggregated statistics about what organisms are doing.
    """
    profiles = []
    all_opcodes = Counter()
    genome_lengths = []
    energies = []
    ages = []

    for r in range(world.size):
        for c in range(world.size):
            agent = world.grid[r][c].agent
            if agent is not None:
                profiles.append(behavioral_profile(agent.genome))
                for g in agent.genome:
                    all_opcodes[g % NUM_OPCODES] += 1
                genome_lengths.append(len(agent.genome))
                energies.append(agent.energy)
                ages.append(agent.age)

    if not profiles:
        return {'population': 0}

    # Average behavioral profile
    avg_profile = {}
    for key in profiles[0]:
        if key == 'length':
            avg_profile[key] = sum(p[key] for p in profiles) / len(profiles)
        else:
            avg_profile[key] = sum(p.get(key, 0) for p in profiles) / len(profiles)

    # Opcode ranking
    total_opcodes = sum(all_opcodes.values())
    opcode_ranking = sorted(
        [(op, count / total_opcodes) for op, count in all_opcodes.items()],
        key=lambda x: -x[1]
    )

    # Key opcodes of interest
    stigmergy_usage = sum(all_opcodes.get(op, 0) for op in [0x07, 0x08])  # LOAD, STORE
    communication_usage = sum(all_opcodes.get(op, 0) for op in [0x1A, 0x1B])  # SEND, RECV
    sharing_usage = all_opcodes.get(0x0B, 0)  # SHARE
    parasite_signature = all_opcodes.get(0x0C, 0) > 0 and all_opcodes.get(0x0A, 0) == 0  # forks but never eats

    return {
        'population': len(profiles),
        'avg_profile': avg_profile,
        'avg_energy': sum(energies) / len(energies),
        'max_energy': max(energies),
        'avg_age': sum(ages) / len(ages),
        'max_age': max(ages),
        'genome_length_dist': {
            'min': min(genome_lengths),
            'max': max(genome_lengths),
            'avg': sum(genome_lengths) / len(genome_lengths),
            'stdev': math.sqrt(sum((l - sum(genome_lengths)/len(genome_lengths))**2 for l in genome_lengths) / len(genome_lengths)),
        },
        'top_opcodes': opcode_ranking[:10],
        'stigmergy_genes': stigmergy_usage,
        'communication_genes': communication_usage,
        'sharing_genes': sharing_usage,
        'parasite_signature': parasite_signature,
    }


def print_census(world: World, species: list = None):
    """Pretty-print a full census."""
    c = census(world)
    if c['population'] == 0:
        print("  [extinct]")
        return

    p = c['avg_profile']
    print(f"  Population: {c['population']}")
    print(f"  Energy: avg={c['avg_energy']:.0f} max={c['max_energy']}")
    print(f"  Age: avg={c['avg_age']:.0f} max={c['max_age']}")
    print(f"  Genome length: {c['genome_length_dist']['min']}-{c['genome_length_dist']['max']} "
          f"(avg={c['genome_length_dist']['avg']:.1f} σ={c['genome_length_dist']['stdev']:.1f})")

    print(f"  Strategy profile:")
    for k in ['foraging', 'movement', 'reproduction', 'stigmergy', 'communication',
              'computation', 'control_flow', 'sharing', 'death']:
        v = p.get(k, 0)
        bar = '█' * int(v * 40)
        print(f"    {k:>14}: {v:.3f} {bar}")

    print(f"  Top opcodes:", end='')
    from substrate import NOP, INC_R0, INC_R1, DEC_R0, DEC_R1, ADD_RR, SUB_RR, LOAD, STORE
    from substrate import MOVE, EAT, SHARE, FORK, JMP, JZ, JNZ, CMPZ, RAND, SENSE, DIE, SWAP
    from substrate import SHL, SHR, AND_RR, OR_RR, XOR_RR, SEND, RECV, SET_R0, SET_R1
    opcode_names = {
        NOP: 'NOP', INC_R0: 'INC_R0', INC_R1: 'INC_R1', DEC_R0: 'DEC_R1',
        DEC_R1: 'DEC_R1', ADD_RR: 'ADD_RR', SUB_RR: 'SUB_RR', LOAD: 'LOAD',
        STORE: 'STORE', MOVE: 'MOVE', EAT: 'EAT', SHARE: 'SHARE', FORK: 'FORK',
        JMP: 'JMP', JZ: 'JZ', JNZ: 'JNZ', CMPZ: 'CMPZ', RAND: 'RAND',
        SENSE: 'SENSE', DIE: 'DIE', SWAP: 'SWAP', SHL: 'SHL', SHR: 'SHR',
        AND_RR: 'AND_RR', OR_RR: 'OR_RR', XOR_RR: 'XOR_RR', SEND: 'SEND',
        RECV: 'RECV', SET_R0: 'SET_R0', SET_R1: 'SET_R1',
    }
    for op, freq in c['top_opcodes']:
        name = opcode_names.get(op, f'?{op}')
        print(f" {name}={freq:.1%}", end='')
    print()

    # Flags for emerging behaviors
    flags = []
    if c['stigmergy_genes'] > 0:
        flags.append(f"STIGMERGY({c['stigmergy_genes']})")
    if c['communication_genes'] > 0:
        flags.append(f"COMMUNICATION({c['communication_genes']})")
    if c['sharing_genes'] > 0:
        flags.append(f"SHARING({c['sharing_genes']})")
    if c['parasite_signature']:
        flags.append("⚠ PARASITE_SIGNATURE")
    if flags:
        print(f"  Emergent behaviors: {' | '.join(flags)}")

    # Species breakdown
    if species:
        print(f"  Species: {len(species)} detected")
        for i, sp in enumerate(species[:5]):
            print(f"    Species {i}: {len(sp)} agents")
            # Show representative genome
            rep = sp[0][2]
            names = []
            for g in rep[:16]:
                op = g % NUM_OPCODES
                names.append(opcode_names.get(op, f'?{op}'))
            print(f"      genome (first 16): {' '.join(names)}")
        if len(species) > 5:
            print(f"    ... and {len(species) - 5} more species")
