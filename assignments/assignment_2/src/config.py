"""Experiment configuration shared by every runner and analysis script.

A single dataclass, `ExperimentConfig`, pins down everything that must stay
fixed for two runs to be comparable: body, world, controller architecture,
simulation length, target position, and the fall threshold used by the
fitness function. Algorithm-specific knobs (population size, evaluation
budget, seed) are also here so a run's manifest can record the exact config
that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

BodyName = Literal["turtle", "iguana"]

# Repo-relative root for this assignment; every script derives its data/
# results paths from here so they work regardless of the caller's cwd.
ASSIGNMENT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ExperimentConfig:
    """Fixed parameters for one evaluation pipeline (body + world + task).

    Parameters
    ----------
    body : {"turtle", "iguana"}
        John Set body to spawn. Fixed for a whole run.
    seed : int
        Seed for this run's local RNG (weight sampling / CMA-ES sampling).
        Not shared with any ariel.ec RNG.
    hidden_size : int
        Width of the single hidden layer of the NN controller.
    sim_duration : float
        Seconds of simulated time per fitness evaluation.
    spawn_pos, target_position : tuple[float, float, float]
        World-frame spawn and target positions for the robot core. Defaults
        match the ariel course's own Olympic Arena template (spawn just
        before the flat section, target on the far side of the incline).
    fall_height_threshold : float
        Minimum core height (m) tolerated during an episode before the
        survival penalty in `fitness_survival_and_locomotion` kicks in.
    lambda_ : int or None
        CMA-ES population size. `None` means "use the paper's default
        formula", computed once the genotype length `n` is known
        (see `cma_es.default_lambda`).
    budget : int
        Total number of fitness evaluations for this run. Random search
        must be given the same `budget` as its paired CMA-ES run.
    sigma0 : float
        Initial CMA-ES step size. NN weights have no natural [A, B] bounds
        (unlike the paper's benchmark functions), so this is chosen to
        match the template's own random-weight-init scale rather than
        derived from a search-region width.
    output_root : Path
        Directory under which per-run data (`__data__/...`) is written.
    """

    body: BodyName
    seed: int
    hidden_size: int = 6
    sim_duration: float = 10.0
    spawn_pos: tuple[float, float, float] = (-0.8, 0.0, 0.0)
    target_position: tuple[float, float, float] = (5.0, 0.0, 0.5)
    fall_height_threshold: float = 0.05
    lambda_: int | None = None
    budget: int = 1500
    sigma0: float = 0.5
    output_root: Path = field(default_factory=lambda: ASSIGNMENT_ROOT / "__data__")

    def run_dir(self, algorithm: Literal["cma_es", "random_search"]) -> Path:
        """Per-seed output directory: `__data__/<body>__<algorithm>/seed_<n>/`."""
        return self.output_root / f"{self.body}__{algorithm}" / f"seed_{self.seed}"
