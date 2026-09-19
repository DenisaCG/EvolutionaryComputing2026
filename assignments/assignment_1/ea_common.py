"""Shared fitness and operators for the historical and mutation-schedule EAs.

Both new conditions always use crossover; only mutation-attempt timing differs.
Random search samples the same depth-valid initialization distribution.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import networkx as nx

from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec import Individual
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,
    crossover_subtree,
    mutate_replace_node,
    mutate_subtree_replacement,
    random_tree,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from tree_edit_distance import mean_plus_std_tree_edit_distance

HERE = Path(__file__).parent
TARGET_DIR = HERE / "target_bodies"

# --- SHARED EXPERIMENT CONSTANTS --- #
# Identical across both EA variants and the random-search baseline.
NUM_MODULES: int = 20  # initial non-core modules; not an evolved size cap
MAX_DEPTH: int = 12  # bloat control, applied identically after mutation and crossover
POP_SIZE: int = 100
NUM_GENERATIONS: int = 50
TOURNAMENT_SIZE: int = 3
MUTATION_REPAIR_ATTEMPTS: int = 20

# Matched evaluation budget: initial population + one full generation's worth
# of offspring per generational step. The random-search baseline draws
# exactly this many bodies.
TOTAL_EVALS: int = (NUM_GENERATIONS + 1) * POP_SIZE

# --- MUTATION SCHEDULE CONSTANTS --- #
MUTATION_START: float = 0.8
# MUTATION_END: float = 0.2  # Original dynamic-scheduler experiment
MUTATION_END: float = 0.01  # Extreme-decrease sensitivity experiment


def exponential_mutation_rate(
    generation: int,
    num_generations: int = NUM_GENERATIONS,
    start: float = MUTATION_START,
    end: float = MUTATION_END,
) -> float:
    """Exponentially decrease mutation probability from start to end."""
    if num_generations <= 1:
        return end

    progress = generation / (num_generations - 1)
    return start * (end / start) ** progress


def mutation_schedule(num_generations: int) -> tuple[list[float], float]:
    """Return the zero-based exponential probabilities and their exact mean."""
    if num_generations < 1:
        raise ValueError("num_generations must be positive")
    rates = [exponential_mutation_rate(g, num_generations) for g in range(num_generations)]
    return rates, sum(rates) / len(rates)


def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory."""
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


def fitness_of(genome: TreeGenome, targets: list[nx.DiGraph]) -> float:
    """Score a tree genome against the target set. Lower is better."""
    return mean_plus_std_tree_edit_distance(genome.to_networkx(), targets)


def random_valid_tree(num_modules: int = NUM_MODULES) -> TreeGenome:
    """Sample the original distribution, requiring non-core content and valid depth."""
    if num_modules < 1:
        raise ValueError("num_modules must be positive")
    while True:
        genome = random_tree(num_modules)
        if len(genome.nodes) > 1 and validate_tree_depth(genome, MAX_DEPTH):
            return genome


def mutate(genome: TreeGenome) -> TreeGenome:
    """Attempt point or subtree mutation (50/50); either may shrink or do nothing.

    Mutation scheduling gates repaired_mutate(), without changing this mixture.
    """
    new = copy.deepcopy(genome)
    if random.random() < 0.5:
        mutate_replace_node(new)
    else:
        mutate_subtree_replacement(new, max_modules=NUM_MODULES)
    _prune_invalid_edges(new)
    return new


def repaired_mutate(genome: TreeGenome) -> TreeGenome:
    """Mutate, retrying from the same parent until the depth cap is met.

    Falls back to an unmutated copy of the parent if no attempt validates -
    the same fallback convention ``crossover_subtree`` itself uses.
    """
    for _ in range(MUTATION_REPAIR_ATTEMPTS):
        candidate = mutate(genome)
        if len(candidate.nodes) > 0 and validate_tree_depth(candidate, MAX_DEPTH):
            return candidate
    return copy.deepcopy(genome)


def repaired_crossover(
    parent_a: TreeGenome,
    parent_b: TreeGenome,
) -> tuple[TreeGenome, TreeGenome]:
    """Subtree crossover, falling back per-child to a parent copy over depth cap.

    ``crossover_subtree`` already validates structural integrity; this adds
    the same ``MAX_DEPTH`` cap used by ``repaired_mutate`` so bloat control
    is identical for both EA variants.
    """
    child_a, child_b = crossover_subtree(parent_a, parent_b)
    if not validate_tree_depth(child_a, MAX_DEPTH):
        child_a = copy.deepcopy(parent_a)
    if not validate_tree_depth(child_b, MAX_DEPTH):
        child_b = copy.deepcopy(parent_b)
    return child_a, child_b


def tournament_select(
    pool: list[Individual],
    k: int = TOURNAMENT_SIZE,
) -> Individual:
    """Pick one individual by k-way tournament (lower fitness wins)."""
    contenders = random.sample(pool, k)
    return min(contenders, key=lambda ind: ind.fitness_)
