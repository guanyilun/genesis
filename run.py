"""
Run the world. Watch what happens.

Usage:
    python run.py                    # run until extinction or Ctrl-C
    python run.py --ticks 10000      # run for N ticks
    python run.py --gui              # terminal GUI with live visualization
"""

import argparse
import sys
import time
import json
import os
from substrate import World

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'runs')


def run_headless(ticks: int, save_every: int = 1000):
    """Run without visualization, print stats periodically."""
    world = World()
    world.seed_population()

    print(f"Genesis: seeded {world.population} agents on a {world.size}x{world.size} grid")
    print(f"Running for {ticks} ticks (or until extinction)...")
    print(f"{'tick':>8} {'pop':>6} {'births':>8} {'deaths':>8} {'avg_glen':>9} {'avg_nrg':>8}")
    print("-" * 60)

    for i in range(ticks):
        world.step()

        if i % save_every == 0:
            s = world.stats()
            print(f"{s['tick']:>8} {s['population']:>6} {s['total_births']:>8} "
                  f"{s['total_deaths']:>8} {s['avg_genome_length']:>9.1f} {s['avg_energy']:>8.1f}")

        if world.population == 0:
            print(f"\nExtinction at tick {world.tick}")
            break

    # Save final state
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_file = os.path.join(OUTPUT_DIR, f"run_{int(time.time())}.json")
    final = world.stats()
    with open(run_file, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\nFinal stats saved to {run_file}")
    return world


def run_gui(ticks: int):
    """
    Terminal GUI — render the world as ASCII art.
    Each cell is one character:
        . = empty
        # = food present
        @ = agent (brightness by energy)
        * = agent + food
        ~ = trace (dead agent residue)
    """
    world = World(size=64)
    world.seed_population()

    try:
        import curses
    except ImportError:
        print("curses not available, falling back to headless mode")
        return run_headless(ticks)

    def curses_main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(0)

        # Colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # agents
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # food
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)     # traces
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # stats

        # Reserve 4 rows for stats panel at the bottom
        max_rows, max_cols = stdscr.getmaxyx()
        stats_rows = 4
        view_size = min(world.size, 50, max_rows - stats_rows, max_cols)
        paused = False
        speed = 1  # ticks per frame

        for frame_tick in range(ticks):
            ch = stdscr.getch()
            if ch == ord('q'):
                break
            elif ch == ord(' '):
                paused = not paused
            elif ch == ord('+') or ch == ord('='):
                speed = min(speed * 2, 64)
            elif ch == ord('-'):
                speed = max(speed // 2, 1)

            if not paused:
                for _ in range(speed):
                    world.step()
                    if world.population == 0:
                        break

            # Render grid
            stdscr.erase()
            for r in range(view_size):
                row_str = []
                for c in range(view_size):
                    cell = world.grid[r][c]
                    if cell.agent is not None:
                        energy_ratio = min(cell.agent.energy / 200, 1.0)
                        if energy_ratio > 0.5:
                            row_str.append(('@', curses.color_pair(1) | curses.A_BOLD))
                        else:
                            row_str.append(('@', curses.color_pair(1)))
                    elif cell.food > 0:
                        row_str.append(('#', curses.color_pair(2)))
                    elif cell.trace > 0:
                        row_str.append(('~', curses.color_pair(3)))
                    else:
                        row_str.append(('.', 0))

                for col, (ch_val, attr) in enumerate(row_str):
                    try:
                        stdscr.addch(r, col, ch_val, attr)
                    except curses.error:
                        pass

            # Stats panel
            s = world.stats()
            stats_y = view_size + 1
            try:
                stdscr.addstr(stats_y, 0,
                    f"tick:{s['tick']} pop:{s['population']} births:{s['total_births']} "
                    f"deaths:{s['total_deaths']} avg_genome:{s['avg_genome_length']:.1f} "
                    f"speed:{speed}x",
                    curses.color_pair(4))
            except curses.error:
                pass

            if world.population == 0:
                try:
                    stdscr.addstr(stats_y + 1, 0, "EXTINCTION", curses.color_pair(3) | curses.A_BOLD)
                except curses.error:
                    pass

            if paused:
                try:
                    stdscr.addstr(stats_y + 1, 0, "PAUSED (space to resume, q to quit)")
                except curses.error:
                    pass

            try:
                stdscr.addstr(stats_y + 2, 0,
                    "controls: space=pause  +/-=speed  q=quit",
                    curses.color_pair(4))
            except curses.error:
                pass

            stdscr.refresh()
            time.sleep(0.05)

            if world.population == 0:
                time.sleep(2)
                break

    curses.wrapper(curses_main)
    return world


def main():
    parser = argparse.ArgumentParser(description='Genesis — self-architecting computational ecosystem')
    parser.add_argument('--ticks', type=int, default=50000, help='max ticks to run')
    parser.add_argument('--gui', action='store_true', help='terminal GUI visualization')
    args = parser.parse_args()

    print("╔══════════════════════════════════════╗")
    print("║          G E N E S I S              ║")
    print("║  self-architecting computational     ║")
    print("║         ecosystem v0.1               ║")
    print("╚══════════════════════════════════════╝")
    print()

    if args.gui:
        run_gui(args.ticks)
    else:
        run_headless(args.ticks)


if __name__ == '__main__':
    main()
