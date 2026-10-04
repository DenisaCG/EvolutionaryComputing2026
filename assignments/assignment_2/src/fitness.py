"""Fitness: distance reduced towards the target, with a fall penalty.

Uses `ariel.simulation.tasks.targeted_locomotion`'s
`fitness_survival_and_locomotion` unchanged: if the core's height dropped
below 0.05 m at any point during the episode, the robot gets a flat penalty
of 10.0; otherwise its score is the change in planar (x, y) distance to the
target, `final_dist - initial_dist`. Lower is better (CMA-ES here always
minimizes).

With a fixed spawn and target, this ranks controllers exactly like the
template's plain final distance to target (it differs by the constant
initial distance). In the Olympic Arena the fall gate effectively detects
falling off the arena's cliffs: the turtle's core stands at ~0.085 m and
stays above ~0.07 m while moving on the terrain.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from ariel.simulation.tasks.targeted_locomotion import (
    fitness_survival_and_locomotion,
)

from config import ExperimentConfig
from simulate import EpisodeMetrics, run_episode


def evaluate_fitness(config: ExperimentConfig, metrics: EpisodeMetrics) -> float:
    """Delta distance to target, or the fall penalty. Lower is better."""
    return fitness_survival_and_locomotion(
        metrics.initial_position,
        metrics.final_position,
        np.asarray(config.target_position),
        metrics.min_core_height,
    )


def evaluate(config: ExperimentConfig, flat_weights: npt.NDArray[np.float64]) -> float:
    """Run one episode with `flat_weights` and score it. Lower is better."""
    metrics = run_episode(config, flat_weights)
    return evaluate_fitness(config, metrics)
