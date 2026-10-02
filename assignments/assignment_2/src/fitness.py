"""Combined fitness: survival gate, then direct-path distance/efficiency.

Reuses the two ready-made metrics from
`ariel.simulation.tasks.targeted_locomotion` instead of reimplementing them:
if the robot fell (core height dropped below `fall_height_threshold` at any
point during the episode), it gets `fitness_survival_and_locomotion`'s flat
fall penalty; otherwise its score is `fitness_direct_path`, which rewards
closing distance to the target while penalizing wasted (non-straight-line)
movement. Lower is better (CMA-ES here always minimizes).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from ariel.simulation.tasks.targeted_locomotion import (
    fitness_direct_path,
    fitness_survival_and_locomotion,
)

from config import ExperimentConfig
from simulate import EpisodeMetrics, run_episode


def evaluate_fitness(config: ExperimentConfig, metrics: EpisodeMetrics) -> float:
    """Combined survival + direct-path fitness for one episode. Lower is better."""
    target = np.asarray(config.target_position)

    survival_score = fitness_survival_and_locomotion(
        metrics.initial_position,
        metrics.final_position,
        target,
        metrics.min_core_height,
    )
    if metrics.min_core_height < config.fall_height_threshold:
        return survival_score

    return fitness_direct_path(
        metrics.initial_position,
        metrics.final_position,
        target,
        metrics.total_path_length,
    )


def evaluate(config: ExperimentConfig, flat_weights: npt.NDArray[np.float64]) -> float:
    """Run one episode with `flat_weights` and score it. Lower is better."""
    metrics = run_episode(config, flat_weights)
    return evaluate_fitness(config, metrics)
