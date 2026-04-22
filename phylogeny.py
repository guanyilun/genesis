"""
Phylogenetic tracking — every birth records parent and child.

The World maintains a lineage graph. Each agent gets a unique ID at creation.
FORK records the parent-child edge. This lets us answer:

1. Is LUCA's lineage still alive?
2. How many distinct lineages survive?
3. What's the branching factor — are we seeing radiation or stasis?
4. Can we detect speciation from genome distance + lineage distance?

The graph is stored as adjacency lists. For long runs, we periodically
prune dead leaves (agents with no living descendants) to keep memory bounded.
"""

from __future__ import annotations
import json
import os
import time
import random
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from substrate import (
    World, Agent, Cell, NUM_OPCODES,
    INITIAL_ENERGY, INITIAL_POPULATION
)


_next_id = 0

def _new_id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


@dataclass
class Node:
    """A node in the phylogenetic tree."""
    id: int
    parent_id: Optional[int]
    tick_born: int
    genome: list[int]
    tick_died: Optional[int] = None
    is_luca: bool = False
    children: list[int] = field(default_factory=list)


class Phylogeny:
    """Tracks ancestry of all agents in the world."""

    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.agent_to_node: dict[int, int] = {}  # id(agent obj) -> node id
        self.living: set[int] = set()  # node ids of living agents

    def register_birth(self, agent: Agent, tick: int, parent_node_id: Optional[int] = None,
                       is_luca: bool = False):
        """Called when an agent is created."""
        node_id = _new_id()
        node = Node(
            id=node_id,
            parent_id=parent_node_id,
            tick_born=tick,
            genome=list(agent.genome),
            is_luca=is_luca,
        )
        self.nodes[node_id] = node
        self.agent_to_node[id(agent)] = node_id
        self.living.add(node_id)

        if parent_node_id is not None and parent_node_id in self.nodes:
            self.nodes[parent_node_id].children.append(node_id)

    def register_death(self, agent: Agent, tick: int):
        """Called when an agent dies."""
        node_id = self.agent_to_node.get(id(agent))
        if node_id is not None:
            self.nodes[node_id].tick_died = tick
            self.living.discard(node_id)

    def get_node(self, agent: Agent) -> Optional[Node]:
        node_id = self.agent_to_node.get(id(agent))
        if node_id is not None:
            return self.nodes.get(node_id)
        return None

    def luca_survives(self) -> bool:
        """Are any living agents descended from LUCA?"""
        # Find all LUCA nodes
        luca_nodes = [n for n in self.nodes.values() if n.is_luca]
        if not luca_nodes:
            return False

        # BFS from LUCA nodes through children, check if any are living
        visited = set()
        queue = [n.id for n in luca_nodes]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current in self.living:
                return True
            node = self.nodes.get(current)
            if node:
                queue.extend(node.children)
        return False

    def living_lineages(self) -> int:
        """Count distinct root lineages (agents with no parent in the tree) that have living descendants."""
        roots = set()
        for node in self.nodes.values():
            if node.parent_id is None or node.parent_id not in self.nodes:
                roots.add(node.id)

        living_roots = set()
        for root_id in roots:
            visited = set()
            queue = [root_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                if current in self.living:
                    living_roots.add(root_id)
                    break
                node = self.nodes.get(current)
                if node:
                    queue.extend(node.children)
        return len(living_roots)

    def lineage_depth(self, agent: Agent) -> int:
        """How many generations from root to this agent?"""
        node_id = self.agent_to_node.get(id(agent))
        if node_id is None:
            return 0
        depth = 0
        current = self.nodes.get(node_id)
        while current and current.parent_id is not None:
            depth += 1
            current = self.nodes.get(current.parent_id)
        return depth

    def max_depth(self) -> int:
        """Maximum lineage depth among living agents."""
        if not self.living:
            return 0
        max_d = 0
        for node_id in self.living:
            depth = 0
            current = self.nodes.get(node_id)
            while current and current.parent_id is not None:
                depth += 1
                current = self.nodes.get(current.parent_id)
            max_d = max(max_d, depth)
        return max_d

    def prune_dead_branches(self):
        """Remove nodes that are dead and have no living descendants. Frees memory."""
        # Mark all ancestors of living nodes
        ancestors = set()
        for node_id in self.living:
            current = self.nodes.get(node_id)
            while current:
                ancestors.add(current.id)
                current = self.nodes.get(current.parent_id) if current.parent_id else None

        # Remove nodes not in ancestors and not living
        dead_ids = [nid for nid in self.nodes if nid not in ancestors and nid not in self.living]
        for nid in dead_ids:
            node = self.nodes[nid]
            # Remove from parent's children
            if node.parent_id and node.parent_id in self.nodes:
                parent = self.nodes[node.parent_id]
                parent.children = [c for c in parent.children if c != nid]
            del self.nodes[nid]
            self.living.discard(nid)

    def summary(self) -> dict:
        return {
            'total_nodes': len(self.nodes),
            'living': len(self.living),
            'luca_survives': self.luca_survives(),
            'living_lineages': self.living_lineages(),
            'max_depth': self.max_depth(),
        }


class TrackedWorld(World):
    """
    World subclass that tracks phylogeny. Overrides place_agent and remove_agent
    to register births and deaths. Hooks into FORK to link parent and child.
    """

    def __init__(self, size=128):
        super().__init__(size)
        self.phylogeny = Phylogeny()

    def seed_population(self, count=INITIAL_POPULATION, luca_fraction=0.3):
        """Override to register seeds with phylogeny."""
        luca_count = int(count * luca_fraction)
        for i in range(count):
            if i < luca_count:
                genome = self.luca_genome()
                is_luca = True
            else:
                genome = [random.randint(0, NUM_OPCODES - 1) for _ in range(
                    random.randint(8, 64)
                )]
                is_luca = False
            agent = Agent(genome=genome, energy=INITIAL_ENERGY)
            for _ in range(100):
                r = random.randint(0, self.size - 1)
                c = random.randint(0, self.size - 1)
                if self.grid[r][c].agent is None:
                    self.place_agent_tracked(r, c, agent, tick=0, is_luca=is_luca)
                    break

    def place_agent_tracked(self, r, c, agent, tick, parent_node_id=None, is_luca=False):
        ok = self.place_agent(r, c, agent)
        if ok:
            self.phylogeny.register_birth(agent, tick, parent_node_id, is_luca)
        return ok

    def remove_agent(self, r, c):
        cell = self.get_cell(r, c)
        if cell.agent is not None:
            self.phylogeny.register_death(cell.agent, self.tick)
        super().remove_agent(r, c)

    def _dispatch(self, r, c, agent, opcode, operand):
        """Override to track FORK parent-child."""
        if opcode == 0x0C:  # FORK
            self._tracked_fork(r, c, agent, operand)
            return
        super()._dispatch(r, c, agent, opcode, operand)

    def _tracked_fork(self, r, c, agent, operand):
        """FORK with phylogenetic tracking."""
        if agent.energy < 40:
            return
        from substrate import DIRECTIONS
        nr, nc = self.neighbor_pos(r, c, operand % 4)
        ncell = self.get_cell(nr, nc)
        if ncell.agent is None:
            child_genome = self._mutate(agent.copy_genome())
            child_energy = agent.energy // 3
            agent.energy -= child_energy + 40
            child = Agent(genome=child_genome, energy=child_energy)

            parent_node = self.phylogeny.get_node(agent)
            parent_node_id = parent_node.id if parent_node else None

            ok = self.place_agent(nr, nc, child)
            if ok:
                self.phylogeny.register_birth(child, self.tick, parent_node_id)


def run_tracked(ticks=50000, snapshot_every=5000):
    """Run with full phylogenetic tracking."""
    world = TrackedWorld()
    world.seed_population()

    print(f"{'tick':>7} {'pop':>5} {'births':>7} {'deaths':>7} {'glen':>5} {'nrg':>6} {'luca':>5} {'lineages':>9} {'depth':>6} {'nodes':>7}")
    print("-" * 85)

    start = time.time()
    prune_counter = 0

    for i in range(ticks):
        world.step()
        prune_counter += 1

        if prune_counter >= 1000:
            world.phylogeny.prune_dead_branches()
            prune_counter = 0

        if i % snapshot_every == 0:
            s = world.stats()
            p = world.phylogeny.summary()
            print(f"{s['tick']:>7} {s['population']:>5} {s['total_births']:>7} "
                  f"{s['total_deaths']:>7} {s['avg_genome_length']:>5.1f} {s['avg_energy']:>6.0f} "
                  f"{'YES' if p['luca_survives'] else 'no':>5} {p['living_lineages']:>9} "
                  f"{p['max_depth']:>6} {p['total_nodes']:>7}")

        if world.population == 0:
            print(f"\nExtinction at tick {world.tick}")
            break

        if time.time() - start > 55:
            print(f"\nTime limit at tick {world.tick}")
            break

    # Final analysis
    p = world.phylogeny.summary()
    print(f"\n--- Final Phylogeny ---")
    print(f"LUCA lineage survives: {'YES' if p['luca_survives'] else 'NO'}")
    print(f"Living lineages: {p['living_lineages']}")
    print(f"Max lineage depth: {p['max_depth']}")
    print(f"Total tracked nodes: {p['total_nodes']}")

    # Sample living agents with their lineage info
    print(f"\n--- Living Agent Samples ---")
    count = 0
    for r in range(world.size):
        for c in range(world.size):
            agent = world.grid[r][c].agent
            if agent is not None and count < 8:
                depth = world.phylogeny.lineage_depth(agent)
                node = world.phylogeny.get_node(agent)
                root_type = "LUCA" if (node and _is_luca_descendant(world.phylogeny, node)) else "random"
                print(f"  [{r:>3},{c:>3}] len={len(agent.genome):>2} energy={agent.energy:>5} "
                      f"age={agent.age:>4} depth={depth:>3} origin={root_type}")
                count += 1

    return world


def _is_luca_descendant(phylo: Phylogeny, node: Node) -> bool:
    """Walk up the tree to see if this node descends from a LUCA seed."""
    current = node
    while current:
        if current.is_luca:
            return True
        current = phylo.nodes.get(current.parent_id) if current.parent_id else None
    return False


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════╗")
    print("║    G E N E S I S  —  Phylogenetic Tracker       ║")
    print("║    'Who is still alive? And who sent them?'      ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    run_tracked()
