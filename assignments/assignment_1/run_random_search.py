"""Random-search baseline, matched to the EA's evaluation budget.

Random search has no generations, so results are logged against cumulative
evaluation count rather than forced into a generation axis. Plot this against
the EA lines using cumulative evals = (generation + 1) * pop_size for the same
comparison points (see run_ea.py for pop_size/generations).

Usage
-----
    python run_random_search.py --seed 0
    python run_random_search.py --seed 0 --dynamic-scheduler

Random search samples the EA initialization distribution, not the entire
variable-size domain reachable through evolution.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from ea_common import (
    NUM_GENERATIONS,
    POP_SIZE,
    fitness_of,
    load_targets,
    random_valid_tree,
)

HERE = Path(__file__).parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--dynamic-scheduler", action="store_true")
    parser.add_argument("--extreme-decrease", action="store_true",
                        help="Store the sensitivity baseline in separate raw-data paths.")
    args = parser.parse_args(argv)
    if args.extreme_decrease and not args.dynamic_scheduler:
        parser.error("--extreme-decrease requires --dynamic-scheduler")
    if args.pop_size < 1 or args.generations < 0:
        parser.error("pop-size must be positive and generations nonnegative")

    random.seed(args.seed)
    np.random.seed(args.seed)

    total_evals = (args.generations + 1) * args.pop_size

    targets = load_targets()

    folder = "random_search_dynamic_scheduler" if args.dynamic_scheduler else "random_search"
    data_root = HERE / "__data__"
    if args.extreme_decrease:
        data_root /= "extreme_decrease"
    data_dir = data_root / folder / f"seed_{args.seed}"
    data_dir.mkdir(parents=True, exist_ok=not args.dynamic_scheduler)

    best_so_far = float("inf")
    best_genome = None
    rows: list[tuple[int, float, float]] = []
    for eval_index in range(total_evals):
        genome = random_valid_tree()
        fitness = fitness_of(genome, targets)
        if fitness < best_so_far:
            best_so_far = fitness
            best_genome = genome
        rows.append((eval_index, fitness, best_so_far))

    with (data_dir / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_index", "fitness", "best_so_far"])
        writer.writerows(rows)

    with (data_dir / "best_genome.json").open("w") as f:
        json.dump(best_genome.to_dict(), f)

    if args.dynamic_scheduler:
        metadata = {
            "status": "complete", "condition": "random_search", "seed": args.seed,
            "pop_size": args.pop_size, "generations": args.generations,
            "total_evaluations": total_evals, "actual_evaluations": len(rows),
            "baseline": "random search over the depth-valid EA initialization distribution",
        }
        (data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
