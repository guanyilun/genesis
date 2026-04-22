"""
Patchy world variant — food appears in clusters, not uniformly.

The hypothesis: uniform food favors solitary foraging (which we proved).
Patchy food creates:
1. Local population booms around hotspots → resource exhaustion
2. Selection for migration between patches
3. Value in "this patch is depleted" signals → stigmergy becomes useful
4. Possible territorial behavior around rich patches

This is the control experiment. If stigmergy/emergence happens here
but not in the uniform world, we've shown it's the physics, not the
organisms, that determines what evolves.
"""

import random
import math
from substrate import (
    World, Agent, Cell, DIRECTIONS, NUM_OPCODES, HAS_OPERAND,
    WORLD_SIZE, MAX_GENOME, MIN_GENOME, INITIAL_ENERGY,
    TICK_ENERGY_COST, MOVE_ENERGY_COST, FORK_ENERGY_COST,
    EAT_YIELD, MAX_AGE, MUTATION_RATE, INITIAL_POPULATION,
)


class PatchyWorld(World):
    """
    World where food spawns in clusters rather than uniformly.
    
    Every few ticks, a "food event" occurs: a random point is chosen
    as the center of a patch, and food is scattered in a Gaussian
    distribution around it. This creates rich and poor regions.
    """

    def __init__(self, size=WORLD_SIZE, patch_interval=10,
                 food_per_patch=200, patch_radius=8):
        super().__init__(size)
        self.patch_interval = patch_interval
        self.food_per_patch = food_per_patch
        self.patch_radius = patch_radius
        self.patch_count = 0
        self.patch_centers = []  # track where patches appeared

    def spawn_food(self):
        """
        Override: instead of scattering food everywhere, create 
        concentrated patches at intervals.
        """
        if self.tick % self.patch_interval != 0:
            return

        # Choose a random center for the patch
        cr = random.randint(0, self.size - 1)
        cc = random.randint(0, self.size - 1)
        self.patch_centers.append((self.tick, cr, cc))
        self.patch_count += 1

        # Scatter food in a Gaussian around the center
        for _ in range(self.food_per_patch):
            dr = int(random.gauss(0, self.patch_radius))
            dc = int(random.gauss(0, self.patch_radius))
            r, c = (cr + dr) % self.size, (cc + dc) % self.size
            self.grid[r][c].food += random.randint(5, 20)


class OasisWorld(World):
    """
    Even more extreme: food only appears at fixed oasis locations.
    Agents must migrate between oases or starve. Cooperation at
    oases (sharing, signaling) is strongly selected for.
    """

    def __init__(self, size=WORLD_SIZE, num_oases=5, oasis_radius=6,
                 refill_interval=100, refill_amount=800):
        super().__init__(size)
        self.num_oases = num_oases
        self.oasis_radius = oasis_radius
        self.refill_interval = refill_interval
        self.refill_amount = refill_amount
        
        # Place oases at well-separated positions
        self.oases = []
        for _ in range(num_oases * 10):  # try many times
            r = random.randint(oasis_radius, size - oasis_radius)
            c = random.randint(oasis_radius, size - oasis_radius)
            # Check separation from existing oases
            too_close = False
            for or_, oc in self.oases:
                dist = math.sqrt((r - or_)**2 + (c - oc)**2)
                if dist < size / (num_oases ** 0.5):
                    too_close = True
                    break
            if not too_close:
                self.oases.append((r, c))
            if len(self.oases) >= num_oases:
                break

    def spawn_food(self):
        """Override: food only at oases, refilled periodically."""
        if self.tick % self.refill_interval != 0:
            return
        
        for or_, oc in self.oases:
            for _ in range(self.refill_amount // self.num_oases):
                dr = int(random.gauss(0, self.oasis_radius))
                dc = int(random.gauss(0, self.oasis_radius))
                r, c = (or_ + dr) % self.size, (oc + dc) % self.size
                self.grid[r][c].food += random.randint(3, 12)


def run_patchy(ticks=50000, census_every=5000,
               patch_interval=3, food_per_patch=400, patch_radius=10):
    """Run the patchy food experiment."""
    world = PatchyWorld(
        patch_interval=patch_interval,
        food_per_patch=food_per_patch,
        patch_radius=patch_radius,
    )
    world.seed_population()
    
    print(f"{'='*80}")
    print(f"PATCHY FOOD EXPERIMENT — {ticks} ticks")
    print(f"{'='*80}")
    
    import time
    start = time.time()
    
    for i in range(ticks):
        world.step()
        
        if i % census_every == 0:
            elapsed = time.time() - start
            tps = (i + 1) / elapsed if elapsed > 0 else 0
            s = world.stats()
            
            print(f"\nTICK {world.tick}  ({elapsed:.0f}s, {tps:.0f} tps)")
            print(f"  pop={s['population']}  births={s['total_births']}  deaths={s['total_deaths']}")
            print(f"  avg_glen={s['avg_genome_length']:.1f}  avg_nrg={s['avg_energy']:.0f}  "
                  f"patches={world.patch_count}")
        
        if world.population == 0:
            print(f"\nEXTINCTION at tick {world.tick}")
            break
        
        if time.time() - start > 55:
            print(f"\nTIME LIMIT at tick {world.tick}")
            break
    
    return world


if __name__ == '__main__':
    run_patchy()
