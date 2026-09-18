"""Shared pieces for the mutation-only vs. mutation+crossover experiment.

Both EA variants (run_ea.py) and the random-search baseline
(run_random_search.py) import from here, so the crossover toggle in
run_ea.py stays the only difference between the two EA conditions:
targets, fitness, mutation operators, depth cap, and selection are all
defined once, in one place.
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
NUM_MODULES: int = 20
MAX_DEPTH: int = 12  # bloat control, applied identically after mutation and crossover
POP_SIZE: int = 100
NUM_GENERATIONS: int = 50
TOURNAMENT_SIZE: int = 3
MUTATION_REPAIR_ATTEMPTS: int = 20

# Matched evaluation budget: initial population + one full generation's worth
# of offspring per generational step. The random-search baseline draws
# exactly this many bodies.
TOTAL_EVALS: int = (NUM_GENERATIONS + 1) * POP_SIZE


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
    """Sample a random tree genome with at least one non-core module."""
    while True:
        genome = random_tree(num_modules)
        if len(genome.nodes) > 0:
            return genome


def mutate(genome: TreeGenome) -> TreeGenome:
    """Apply one of the two "always active" GP mutation operators (50/50).

    ``mutate_shrink`` and ``mutate_hoist`` are deliberately excluded: they
    are bloat-control operators, not primary variation, and mixing them in
    would add a second uncontrolled factor (tree-size drift) on top of the
    crossover toggle this experiment is isolating.
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
