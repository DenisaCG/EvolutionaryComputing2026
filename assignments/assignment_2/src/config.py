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
    clock_hz : float
        Frequency of the controller's sin/cos clock input (see
        `controller.py`). Default ~1.3 Hz follows Clune et al. (2009, 2011).
    sim_duration : float
        Seconds of simulated time per fitness evaluation.
    spawn_pos, target_position : tuple[float, float, float]
        World-frame spawn and target positions for the robot core. Defaults
        match the ariel course's own Olympic Arena template (spawn just
        before the flat section, target on the far side of the incline).
    fall_height_threshold : float
        Minimum core height (m) tolerated during an episode before the
        survival penalty in `fitness_survival_and_locomotion` kicks in.
        Recorded for the manifest only: ariel hardcodes 0.05 in that
        function, so changing this value has no effect.
    terrain_seed : int or None
        Seed for the Olympic Arena's rugged-terrain Perlin noise (see
        `bodies.build_world`), so the world is identical in every
        evaluation. `None` = ariel's default, a new random terrain per
        evaluation (the setup of the `ipop_l10_b8000` pilot).
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
    stagnation_c, stagnation_tol : float, float
        Task-specific stagnation criterion (see `cma_es.CMAES`): a CMA-ES run
        counts as stuck when its best-so-far improved by less than
        `stagnation_tol` (m) over K = 10 + ceil(stagnation_c * n / lambda)
        generations (the paper's equalfunvalhist window form, constant 30 ->
        4.5). For n = 199: K = 100, 55, 33, 22, 16 at lambda = 10..160.
        Chosen from the 2-seed pilot (`results/ipop_l10_b8000`): at
        lambda = 10, CMA-ES stalls of up to ~90 generations were followed by
        real gains (2-6.5 cm); longer stalls only by <= 2 cm.
    max_lambda : int
        IPOP population cap: restarts double lambda until doubling would
        exceed this, then keep the largest lambda reached.
    output_root : Path
        Directory under which per-run data (`__data__/...`) is written.
    """

    body: BodyName
    seed: int
    hidden_size: int = 6
    clock_hz: float = 1.3
    sim_duration: float = 15.0
    spawn_pos: tuple[float, float, float] = (-0.8, 0.0, 0.0)
    target_position: tuple[float, float, float] = (5.0, 0.0, 0.5)
    fall_height_threshold: float = 0.05
    terrain_seed: int | None = 0
    lambda_: int | None = None
    budget: int = 1500
    sigma0: float = 0.5
    stagnation_c: float = 4.5
    stagnation_tol: float = 0.01
    max_lambda: int = 200
    output_root: Path = field(default_factory=lambda: ASSIGNMENT_ROOT / "__data__")

    def run_dir(
        self, algorithm: Literal["cma_es", "ipop_cma_es", "random_search"]
    ) -> Path:
        """Per-seed output directory: `__data__/<body>__<algorithm>/seed_<n>/`."""
        return self.output_root / f"{self.body}__{algorithm}" / f"seed_{self.seed}"
