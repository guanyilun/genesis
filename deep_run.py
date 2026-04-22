"""
Deep run — long-duration ecosystem observation with phylogeny + species detection.

Runs the world for an extended period, taking detailed censuses at intervals.
The goal: detect emerging behaviors, species formation, and strategy shifts.
"""

import time
import random
import math
from phylogeny import TrackedWorld
from observatory import cluster_species, census, print_census


def deep_run(ticks=100000, census_every=5000, species_threshold=0.15):
    world = TrackedWorld()
    world.seed_population()

    print(f"{'='*80}")
    print(f"GENESIS DEEP RUN — {ticks} ticks, census every {census_every}")
    print(f"{'='*80}")

    start = time.time()
    prune_counter = 0
    prev_species_count = None

    for i in range(ticks):
        world.step()
        prune_counter += 1

        if prune_counter >= 2000:
            world.phylogeny.prune_dead_branches()
            prune_counter = 0

        if i % census_every == 0:
            elapsed = time.time() - start
            tps = (i + 1) / elapsed if elapsed > 0 else 0

            print(f"\n{'─'*80}")
            print(f"TICK {world.tick}  ({elapsed:.0f}s, {tps:.0f} ticks/s)")
            print(f"{'─'*80}")

            s = world.stats()
            p = world.phylogeny.summary()
            print(f"  pop={s['population']}  births={s['total_births']}  deaths={s['total_deaths']}  "
                  f"avg_glen={s['avg_genome_length']:.1f}  avg_nrg={s['avg_energy']:.0f}")
            print(f"  LUCA alive: {'YES' if p['luca_survives'] else 'NO'}  "
                  f"lineages={p['living_lineages']}  max_depth={p['max_depth']}  "
                  f"tree_nodes={p['total_nodes']}")

            # Full census
            print()
            print_census(world)

            # Species detection (subsample for speed if population is large)
            species = cluster_species(world, threshold=species_threshold)

            # Species change alert
            if prev_species_count is not None and len(species) != prev_species_count:
                direction = "↑ DIVERSIFICATION" if len(species) > prev_species_count else "↓ CONVERGENCE"
                print(f"\n  ⚡ SPECIES COUNT CHANGE: {prev_species_count} → {len(species)} {direction}")
            prev_species_count = len(species)

        if world.population == 0:
            print(f"\n{'='*80}")
            print(f"EXTINCTION at tick {world.tick}")
            print(f"{'='*80}")
            break

        if time.time() - start > 55:
            print(f"\n{'='*80}")
            print(f"TIME LIMIT at tick {world.tick}")
            print(f"{'='*80}")
            break

    # Final phylogeny summary
    p = world.phylogeny.summary()
    print(f"\n{'═'*80}")
    print(f"FINAL REPORT")
    print(f"{'═'*80}")
    print(f"  LUCA lineage survives: {'★ YES' if p['luca_survives'] else '✗ NO'}")
    print(f"  Living lineages: {p['living_lineages']}")
    print(f"  Max lineage depth: {p['max_depth']}")
    print(f"  Total tree nodes tracked: {p['total_nodes']}")

    return world


if __name__ == '__main__':
    deep_run()
