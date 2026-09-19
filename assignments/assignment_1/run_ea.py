"""Run a historical EA or a crossover-based mutation-scheduling condition.

Historical: --seed 0 --no-crossover (or --crossover).
New: --seed 0 --mutation-schedule constant (or exponential).
Both new conditions always cross over. Their per-offspring mutation-attempt
gates use a mean-matched constant or the supplied exponential schedule.
New outputs are separate and existing new runs are never overwritten.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from ea_common import (
    NUM_GENERATIONS,
    MUTATION_START,
    MUTATION_END,
    MAX_DEPTH,
    NUM_MODULES,
    TOURNAMENT_SIZE,
    mutation_schedule,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--crossover",
        action=argparse.BooleanOptionalAction,
        help="Enable subtree crossover before mutation (the one aspect under test).",
    )
    mode.add_argument("--mutation-schedule", choices=("constant", "exponential"))
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    args = parser.parse_args(argv)
    dynamic = args.mutation_schedule is not None
    if args.pop_size < TOURNAMENT_SIZE or args.generations < (1 if dynamic else 0):
        parser.error("pop-size must be >= tournament size; dynamic generations must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)

    targets = load_targets()

    variant = (f"dynamic_{args.mutation_schedule}" if dynamic else
               "mutation_crossover" if args.crossover else "mutation_only")
    data_dir = HERE / "__data__" / "ea" / variant / f"seed_{args.seed}"
    data_dir.mkdir(parents=True, exist_ok=not dynamic)
    if dynamic:
        rates, constant = mutation_schedule(args.generations)
        probabilities = rates if args.mutation_schedule == "exponential" else [constant] * args.generations
        metadata = {
            "status": "running", "seed": args.seed, "condition": variant,
            "pop_size": args.pop_size, "generations": args.generations,
            "total_evaluations": (args.generations + 1) * args.pop_size,
            "mutation_start": MUTATION_START, "mutation_end": MUTATION_END,
            "exponential_probabilities": rates, "constant_mutation_probability": constant,
            "mutation_probabilities": probabilities, "max_depth": MAX_DEPTH,
            "initial_noncore_modules": NUM_MODULES, "tournament_size": TOURNAMENT_SIZE,
            "crossover": "always", "replacement": "full generational, no elitism",
        }
        (data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    schedule_index = 0  # offspring generation 1 -> index 0; generation G -> G-1
    objective_evaluations = 0
    generation_log = []

    def evaluate(population: Population) -> Population:
        nonlocal objective_evaluations
        for ind in population.unevaluated:
            genome = TreeGenome.from_dict(ind.genotype)
            ind.fitness = fitness_of(genome, targets)
            objective_evaluations += 1
        return population

    def reproduce(population: Population) -> Population:
        """Full generational replacement: exactly `pop_size` offspring, no elitism."""
        nonlocal schedule_index
        probability = probabilities[schedule_index] if dynamic else 1.0
        mutation_attempts = crossover_calls = 0
        pool = list(population.alive)
        offspring: list[Individual] = []

        while len(offspring) < args.pop_size:
            if dynamic or args.crossover:
                parent_a = tournament_select(pool)
                parent_b = tournament_select(pool)
                genome_a = TreeGenome.from_dict(parent_a.genotype)
                genome_b = TreeGenome.from_dict(parent_b.genotype)
                child_a, child_b = repaired_crossover(genome_a, genome_b)
                crossover_calls += 1
                for child in (child_a, child_b):
                    if dynamic and len(offspring) == args.pop_size:
                        break  # do not gate/mutate a discarded odd-population child
                    if not dynamic or random.random() < probability:
                        child = repaired_mutate(child)
                        mutation_attempts += 1
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
        if dynamic:
            generation_log.append((schedule_index, schedule_index + 1, probability,
                                   mutation_attempts, crossover_calls, len(offspring)))
            schedule_index += 1
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
        first_generation_id=0 if dynamic else None,
        is_maximisation=False,
        db_file_path=data_dir / "database.db",
        db_handling="halt" if dynamic else "delete",
    )
    ea.run()
    if dynamic:
        assert objective_evaluations == metadata["total_evaluations"]
        assert schedule_index == args.generations
        with (data_dir / "generation_log.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["schedule_index", "offspring_generation", "mutation_probability",
                             "mutation_attempts", "crossover_calls", "offspring_count"])
            writer.writerows(generation_log)
        metadata.update(status="complete", actual_evaluations=objective_evaluations,
                        actual_mutation_attempts=sum(row[3] for row in generation_log),
                        actual_crossover_calls=sum(row[4] for row in generation_log))
        (data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
