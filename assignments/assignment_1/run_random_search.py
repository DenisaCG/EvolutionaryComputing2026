"""Random-search baseline, matched to the EA's evaluation budget.

Random search has no generations, so results are logged against cumulative
evaluation count rather than forced into a generation axis. Plot this against
the EA lines using cumulative evals = generation * pop_size for the same
comparison points (see run_ea.py for pop_size/generations).

Usage
-----
    python run_random_search.py --seed 0
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    total_evals = (args.generations + 1) * args.pop_size

    targets = load_targets()

    data_dir = HERE / "__data__" / "random_search" / f"seed_{args.seed}"
    data_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()
