"""
The substrate — a spatial memory model where agents live, compete, and die.

The world is a 2D grid. Each cell holds zero or one agent. Agents have a genome
(a sequence of opcodes), an energy reserve, a position, and an age. Every tick,
each agent executes one instruction from its genome. Energy depletes per tick.
Death is final — the cell empties, but the genome may leave a trace.

This is not a simulation OF life. It's a substrate where life-like patterns
might or might not emerge. The physics is the point.

Architecture:
- World: 2D grid with spatial locality
- Agent: genome (list of opcodes), energy, position, age, registers
- Genome: list of integers, each mapping to an operation
- Physics: energy costs, locality costs, reproduction costs
- Execution: one opcode per tick per agent, round-robin
"""

import random
from dataclasses import dataclass, field
from typing import Optional

# --- Opcodes ---
# The genome is a sequence of these. Some take operands from subsequent positions.
# The instruction pointer advances past operands automatically.

NOP = 0x00       # do nothing
INC_R0 = 0x01    # r0 += 1
INC_R1 = 0x02    # r1 += 1
DEC_R0 = 0x03    # r0 -= 1
DEC_R1 = 0x04    # r1 -= 1
ADD_RR = 0x05    # r0 += r1
SUB_RR = 0x06    # r0 -= r1
LOAD = 0x07      # r0 = *(r1 + operand)  — read from neighborhood cell offset
STORE = 0x08     # *(r1 + operand) = r0   — write to neighborhood cell offset
MOVE = 0x09      # move in direction (operand): 0=N,1=E,2=S,3=W
EAT = 0x0A       # consume energy from current cell (if any)
SHARE = 0x0B     # give r0 energy to neighbor in direction (operand)
FORK = 0x0C      # reproduce into direction (operand), child gets mutated genome
JMP = 0x0D       # ip = operand
JZ = 0x0E        # if r0 == 0, ip = operand
JNZ = 0x0F       # if r0 != 0, ip = operand
CMPZ = 0x10      # set flag = (r0 == 0)
RAND = 0x11      # r0 = random(0, 255)
SENSE = 0x12     # r0 = energy in neighbor cell at direction (operand)
DIE = 0x13       # voluntary death
SWAP = 0x14      # swap r0, r1
SHL = 0x15       # r0 <<= 1
SHR = 0x16       # r0 >>= 1
AND_RR = 0x17    # r0 &= r1
OR_RR = 0x18     # r0 |= r1
XOR_RR = 0x19    # r0 ^= r1
SEND = 0x1A      # write r0 to message buffer of neighbor at direction (operand)
RECV = 0x1B      # r0 = own message buffer (clears it)
SET_R0 = 0x1C    # r0 = operand
SET_R1 = 0x1D    # r1 = operand

NUM_OPCODES = 0x1E

# Opcodes that consume an operand (the next genome position)
HAS_OPERAND = {LOAD, STORE, MOVE, SHARE, FORK, JMP, JZ, JNZ, SENSE, SEND, SET_R0, SET_R1}

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W

WORLD_SIZE = 128
MAX_GENOME = 64
MIN_GENOME = 8
INITIAL_ENERGY = 300
TICK_ENERGY_COST = 1
MOVE_ENERGY_COST = 2
FORK_ENERGY_COST = 40
EAT_YIELD = 30
MAX_AGE = 3000
MUTATION_RATE = 0.04  # per-genome-position probability of mutation during fork
FOOD_RATE = 0.03       # probability of food appearing in a cell per tick
INITIAL_POPULATION = 300

# Shared memory: each cell has a "trace" value that persists after death
# and a "food" energy value that agents can eat


@dataclass
class Agent:
    genome: list[int]
    ip: int = 0            # instruction pointer
    r0: int = 0            # general register 0
    r1: int = 0            # general register 1
    energy: int = INITIAL_ENERGY
    age: int = 0
    alive: bool = True
    direction: int = 0     # facing direction (0-3)
    message: int = 0       # single-value message buffer

    def copy_genome(self) -> list[int]:
        return list(self.genome)


@dataclass
class Cell:
    agent: Optional[Agent] = None
    food: int = 0           # energy available to eat
    trace: int = 0          # residual value left by dead agents


class World:
    """
    The spatial substrate. A WORLD_SIZE x WORLD_SIZE toroidal grid.
    Each cell can hold one agent, food energy, and a death trace.
    """

    def __init__(self, size: int = WORLD_SIZE):
        self.size = size
        self.grid: list[list[Cell]] = [
            [Cell() for _ in range(size)] for _ in range(size)
        ]
        self.tick = 0
        self.population = 0
        self.total_births = 0
        self.total_deaths = 0
        self.generation_peak = 0

    def _wrap(self, r: int, c: int) -> tuple[int, int]:
        return r % self.size, c % self.size

    def get_cell(self, r: int, c: int) -> Cell:
        r, c = self._wrap(r, c)
        return self.grid[r][c]

    def neighbor_pos(self, r: int, c: int, direction: int) -> tuple[int, int]:
        dr, dc = DIRECTIONS[direction % 4]
        return self._wrap(r + dr, c + dc)

    def place_agent(self, r: int, c: int, agent: Agent) -> bool:
        cell = self.get_cell(r, c)
        if cell.agent is not None:
            return False
        cell.agent = agent
        self.population += 1
        self.total_births += 1
        return True

    def remove_agent(self, r: int, c: int):
        cell = self.get_cell(r, c)
        if cell.agent is not None:
            # Leave a trace — the stigmergy channel
            cell.trace = (cell.trace + cell.agent.age) & 0xFF
            cell.agent.alive = False
            cell.agent = None
            self.population -= 1
            self.total_deaths += 1

    def spawn_food(self):
        """Scatter food randomly across the grid."""
        for _ in range(int(self.size * self.size * FOOD_RATE)):
            r = random.randint(0, self.size - 1)
            c = random.randint(0, self.size - 1)
            self.grid[r][c].food += random.randint(5, 15)

    @staticmethod
    def luca_genome() -> list[int]:
        """
        LUCA — a minimal viable organism.
        Behavior: wander randomly, eat when food is present, reproduce when energy is high.
        
        The genome encodes:
          sense(direction=0)      → r0 = energy/food to the north
          jz(skip_to_move)        → if nothing there, skip ahead
          move(direction=0)       → move north toward food/agent
          eat                     → eat whatever is here
          rand                    → r0 = random value
          r0 &= r0 (test if even via shr trick)
          shr                     → r0 >>= 1 (halve it)
          jz(fork_if_low)         → if small enough (energy check proxy)
          fork(random_direction)  → reproduce in a random direction
          move(random_direction)  → move in a random direction
        """
        return [
            SENSE, 0,       # 0-1:  look north
            JZ, 6,          # 2-3:  if nothing, jump to move
            MOVE, 0,        # 4-5:  move north
            EAT,            # 6:    eat here
            RAND,           # 7:    random value
            SHR,            # 8:    r0 >>= 1
            JNZ, 11,        # 9-10: if r0 != 0, skip fork (50% chance)
            FORK, 2,        # 11-12: fork south
            RAND,           # 13:   random direction value
            AND_RR,         # 14:   r0 &= r1
            MOVE, 1,        # 15-16: move east (or mutated direction)
            EAT,            # 17:   eat
            JMP, 0,         # 18-19: loop back to start
        ]

    def seed_population(self, count: int = INITIAL_POPULATION, luca_fraction: float = 0.3):
        """
        Place agents in the grid. A fraction are LUCA (hand-designed viable organism),
        the rest are random. Evolution takes it from here.
        """
        luca_count = int(count * luca_fraction)
        for i in range(count):
            if i < luca_count:
                genome = self.luca_genome()
            else:
                genome = [random.randint(0, NUM_OPCODES - 1) for _ in range(
                    random.randint(MIN_GENOME, MAX_GENOME)
                )]
            agent = Agent(genome=genome, energy=INITIAL_ENERGY)
            # Try random positions until we find an empty cell
            for _ in range(100):
                r = random.randint(0, self.size - 1)
                c = random.randint(0, self.size - 1)
                if self.grid[r][c].agent is None:
                    self.place_agent(r, c, agent)
                    break

    def step(self):
        """Execute one tick: every agent runs one instruction."""
        self.tick += 1
        self.spawn_food()

        # Collect all living agent positions (order shuffled for fairness)
        positions = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c].agent is not None:
                    positions.append((r, c))
        random.shuffle(positions)

        for r, c in positions:
            cell = self.get_cell(r, c)
            if cell.agent is None or not cell.agent.alive:
                continue
            self._execute_agent(r, c, cell.agent)

        self.generation_peak = max(self.generation_peak, self.population)

    def _execute_agent(self, r: int, c: int, agent: Agent):
        """Execute one instruction for an agent at (r, c)."""
        if not agent.alive:
            return

        # Energy cost of existing
        agent.energy -= TICK_ENERGY_COST
        agent.age += 1

        if agent.energy <= 0 or agent.age >= MAX_AGE:
            self.remove_agent(r, c)
            return

        # Fetch instruction
        if agent.ip >= len(agent.genome):
            agent.ip = 0  # wrap around

        opcode = agent.genome[agent.ip] % NUM_OPCODES
        operand = 0
        advance = 1

        # Fetch operand if needed
        if opcode in HAS_OPERAND:
            operand_idx = (agent.ip + 1) % len(agent.genome)
            operand = agent.genome[operand_idx]
            advance = 2

        agent.ip = (agent.ip + advance) % len(agent.genome)

        # Execute
        self._dispatch(r, c, agent, opcode, operand)

    def _dispatch(self, r: int, c: int, agent: Agent, opcode: int, operand: int):
        cell = self.get_cell(r, c)

        if opcode == NOP:
            pass

        elif opcode == INC_R0:
            agent.r0 = (agent.r0 + 1) & 0xFF

        elif opcode == INC_R1:
            agent.r1 = (agent.r1 + 1) & 0xFF

        elif opcode == DEC_R0:
            agent.r0 = (agent.r0 - 1) & 0xFF

        elif opcode == DEC_R1:
            agent.r1 = (agent.r1 - 1) & 0xFF

        elif opcode == ADD_RR:
            agent.r0 = (agent.r0 + agent.r1) & 0xFF

        elif opcode == SUB_RR:
            agent.r0 = (agent.r0 - agent.r1) & 0xFF

        elif opcode == LOAD:
            # Read trace/food from a neighbor cell
            nr, nc = self.neighbor_pos(r, c, operand % 4)
            ncell = self.get_cell(nr, nc)
            agent.r0 = ncell.trace ^ ncell.food & 0xFF

        elif opcode == STORE:
            # Write to own cell's trace
            cell.trace = agent.r0

        elif opcode == MOVE:
            if agent.energy < MOVE_ENERGY_COST:
                return
            nr, nc = self.neighbor_pos(r, c, operand % 4)
            ncell = self.get_cell(nr, nc)
            if ncell.agent is None:
                cell.agent = None
                ncell.agent = agent
                agent.energy -= MOVE_ENERGY_COST
                agent.direction = operand % 4

        elif opcode == EAT:
            if cell.food > 0:
                take = min(cell.food, EAT_YIELD)
                cell.food -= take
                agent.energy += take

        elif opcode == SHARE:
            if agent.energy > 10:
                nr, nc = self.neighbor_pos(r, c, operand % 4)
                ncell = self.get_cell(nr, nc)
                if ncell.agent is not None:
                    gift = min(agent.r0, agent.energy // 4)
                    if gift > 0:
                        agent.energy -= gift
                        ncell.agent.energy += gift

        elif opcode == FORK:
            if agent.energy < FORK_ENERGY_COST:
                return
            nr, nc = self.neighbor_pos(r, c, operand % 4)
            ncell = self.get_cell(nr, nc)
            if ncell.agent is None:
                child_genome = self._mutate(agent.copy_genome())
                child_energy = agent.energy // 3
                agent.energy -= child_energy + FORK_ENERGY_COST
                child = Agent(genome=child_genome, energy=child_energy)
                self.place_agent(nr, nc, child)

        elif opcode == JMP:
            agent.ip = operand % len(agent.genome)

        elif opcode == JZ:
            if agent.r0 == 0:
                agent.ip = operand % len(agent.genome)

        elif opcode == JNZ:
            if agent.r0 != 0:
                agent.ip = operand % len(agent.genome)

        elif opcode == CMPZ:
            pass  # flag implied by JZ/JNZ — they check r0 directly

        elif opcode == RAND:
            agent.r0 = random.randint(0, 255)

        elif opcode == SENSE:
            nr, nc = self.neighbor_pos(r, c, operand % 4)
            ncell = self.get_cell(nr, nc)
            if ncell.agent is not None:
                agent.r0 = min(ncell.agent.energy, 255)
            else:
                agent.r0 = ncell.food

        elif opcode == DIE:
            self.remove_agent(r, c)

        elif opcode == SWAP:
            agent.r0, agent.r1 = agent.r1, agent.r0

        elif opcode == SHL:
            agent.r0 = (agent.r0 << 1) & 0xFF

        elif opcode == SHR:
            agent.r0 = agent.r0 >> 1

        elif opcode == AND_RR:
            agent.r0 = agent.r0 & agent.r1

        elif opcode == OR_RR:
            agent.r0 = agent.r0 | agent.r1

        elif opcode == XOR_RR:
            agent.r0 = agent.r0 ^ agent.r1

        elif opcode == SEND:
            nr, nc = self.neighbor_pos(r, c, operand % 4)
            ncell = self.get_cell(nr, nc)
            if ncell.agent is not None:
                ncell.agent.message = agent.r0

        elif opcode == RECV:
            agent.r0 = agent.message
            agent.message = 0

        elif opcode == SET_R0:
            agent.r0 = operand & 0xFF

        elif opcode == SET_R1:
            agent.r1 = operand & 0xFF

    def _mutate(self, genome: list[int]) -> list[int]:
        """Mutate a genome: point mutations, insertions, deletions."""
        result = list(genome)

        # Point mutations
        for i in range(len(result)):
            if random.random() < MUTATION_RATE:
                result[i] = random.randint(0, NUM_OPCODES - 1)

        # Structural mutations (rarer)
        if random.random() < 0.1 and len(result) > MIN_GENOME:
            # Deletion
            idx = random.randint(0, len(result) - 1)
            del result[idx]

        if random.random() < 0.1 and len(result) < MAX_GENOME:
            # Insertion
            idx = random.randint(0, len(result))
            result.insert(idx, random.randint(0, NUM_OPCODES - 1))

        if random.random() < 0.05 and len(result) >= 2:
            # Duplication — copy a segment
            start = random.randint(0, len(result) - 2)
            length = random.randint(1, min(4, len(result) - start))
            segment = result[start:start + length]
            insert_at = random.randint(0, len(result))
            for i, gene in enumerate(segment):
                if len(result) < MAX_GENOME:
                    result.insert(insert_at + i, gene)

        return result

    def stats(self) -> dict:
        """Collect world statistics."""
        genome_lengths = []
        opcode_counts = [0] * NUM_OPCODES
        energies = []

        for r in range(self.size):
            for c in range(self.size):
                agent = self.grid[r][c].agent
                if agent is not None:
                    genome_lengths.append(len(agent.genome))
                    energies.append(agent.energy)
                    for gene in agent.genome:
                        opcode_counts[gene % NUM_OPCODES] += 1

        return {
            'tick': self.tick,
            'population': self.population,
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'avg_genome_length': sum(genome_lengths) / len(genome_lengths) if genome_lengths else 0,
            'avg_energy': sum(energies) / len(energies) if energies else 0,
            'peak_population': self.generation_peak,
        }
