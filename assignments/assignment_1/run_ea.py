"""Run one EA variant (mutation-only or mutation+crossover) for one seed.

The two variants are identical in every respect except whether `reproduce`
calls crossover before mutation - controlled by --crossover/--no-crossover.
That single toggle is the one isolated aspect this experiment tests.

Usage
-----
    python run_ea.py --seed 0 --no-crossover
    python run_ea.py --seed 0 --crossover
"""

import argparse
import random
from pathlib import Path

import numpy as np

from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from ea_common import (
    NUM_GENERATIONS,
    POP_SIZE,
    fitness_of,
    load_targets,
    random_valid_tree,
    repaired_crossover,
    repaired_mutate,
    tournament_select,
)

HERE = Path(__file__).parent


def make_individual() -> Individual:
    """Create one fresh individual with a random valid tree genome."""
    ind = Individual()
    ind.genotype = random_valid_tree().to_dict()
    return ind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--crossover",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="Enable subtree crossover before mutation (the one aspect under test).",
    )
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    targets = load_targets()

    variant = "mutation_crossover" if args.crossover else "mutation_only"
    data_dir = HERE / "__data__" / "ea" / variant / f"seed_{args.seed}"
    data_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(population: Population) -> Population:
        for ind in population.unevaluated:
            genome = TreeGenome.from_dict(ind.genotype)
            ind.fitness = fitness_of(genome, targets)
        return population

    def reproduce(population: Population) -> Population:
        """Full generational replacement: exactly `pop_size` offspring, no elitism."""
        pool = list(population.alive)
        offspring: list[Individual] = []

        while len(offspring) < args.pop_size:
            if args.crossover:
                parent_a = tournament_select(pool)
                parent_b = tournament_select(pool)
                genome_a = TreeGenome.from_dict(parent_a.genotype)
                genome_b = TreeGenome.from_dict(parent_b.genotype)
                child_a, child_b = repaired_crossover(genome_a, genome_b)
                for child in (child_a, child_b):
                    child = repaired_mutate(child)
                    ind = Individual()
                    ind.genotype = child.to_dict()
                    offspring.append(ind)
            else:
                parent = tournament_select(pool)
                genome = TreeGenome.from_dict(parent.genotype)
                child = repaired_mutate(genome)
                ind = Individual()
                ind.genotype = child.to_dict()
                offspring.append(ind)

        for ind in pool:
            ind.alive = False
        population.extend(offspring[: args.pop_size])
        return population

    initial = Population([make_individual() for _ in range(args.pop_size)])
    initial = evaluate(initial)

    ops = [
        EAOperation(reproduce),
        EAOperation(evaluate),
    ]

    ea = EA(
        initial,
        ops,
        num_steps=args.generations,
        is_maximisation=False,
        db_file_path=data_dir / "database.db",
        db_handling="delete",
    )
    ea.run()


if __name__ == "__main__":
    main()
