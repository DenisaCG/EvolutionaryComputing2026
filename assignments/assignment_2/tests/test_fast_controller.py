"""`FastController` must reproduce the reference controller bit for bit.

`simulate.run_episode` uses `FastController` (preallocated buffers) instead
of `nn_controller(controller_inputs(...))` for speed. Any numerical drift
would change MuJoCo's trajectory and so every fitness value, making new runs
incomparable with runs made before the change. Two checks: identical actions
on arbitrary states, and identical fitness for a whole episode compared with
the pre-optimisation callback (reimplemented here as the reference).

Run with: uv run pytest assignments/assignment_2/tests/test_fast_controller.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ariel.utils.runners import simple_runner  # noqa: E402

from bodies import build_robot, build_world  # noqa: E402
from config import ExperimentConfig  # noqa: E402
from controller import (  # noqa: E402
    FastController,
    controller_inputs,
    nn_controller,
    unflatten,
)
from fitness import evaluate, evaluate_fitness  # noqa: E402
from simulate import EpisodeMetrics, genotype_length_for  # noqa: E402


def _compiled(config: ExperimentConfig) -> tuple[mj.MjModel, mj.MjData]:
    mj.set_mjcb_control(None)
    world = build_world(config.terrain_seed)
    world.spawn(
        build_robot(config.body).spec,
        position=list(config.spawn_pos),
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    return model, data


def _reference_fitness(config: ExperimentConfig, flat: np.ndarray) -> float:
    """The episode callback exactly as it was before `FastController`."""
    model, data = _compiled(config)
    target_xy = np.asarray(config.target_position[:2])
    weights = unflatten(
        flat,
        len(controller_inputs(data, target_xy, config.clock_hz)),
        config.hidden_size,
        model.nu,
    )
    initial = np.asarray(data.qpos[0:3]).copy()
    tracker = {"min_core_height": float(initial[2])}

    def callback(m: mj.MjModel, d: mj.MjData) -> None:
        d.ctrl[:] = nn_controller(
            controller_inputs(d, target_xy, config.clock_hz), weights
        )
        tracker["min_core_height"] = min(
            tracker["min_core_height"], float(d.qpos[2])
        )

    mj.set_mjcb_control(callback)
    simple_runner(model, data, duration=config.sim_duration)
    mj.set_mjcb_control(None)
    metrics = EpisodeMetrics(
        initial_position=initial,
        final_position=np.asarray(data.qpos[0:3]).copy(),
        total_path_length=0.0,
        min_core_height=tracker["min_core_height"],
    )
    return evaluate_fitness(config, metrics)


@pytest.mark.parametrize("body", ["turtle", "iguana"])
def test_actions_identical_on_random_states(body: str) -> None:
    config = ExperimentConfig(body=body, seed=0)
    model, data = _compiled(config)
    target_xy = np.asarray(config.target_position[:2])
    n = genotype_length_for(config)
    rng = np.random.default_rng(1)

    for _ in range(200):
        data.qpos[:] = rng.normal(size=model.nq)
        data.time = rng.uniform(0, 30)
        weights = unflatten(
            rng.normal(scale=2.0, size=n),
            len(controller_inputs(data, target_xy, config.clock_hz)),
            config.hidden_size,
            model.nu,
        )
        expected = nn_controller(
            controller_inputs(data, target_xy, config.clock_hz), weights
        )
        out = np.empty(model.nu)
        FastController(weights, target_xy, config.clock_hz)(data, out)
        np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("body", ["turtle", "iguana"])
def test_episode_fitness_identical(body: str) -> None:
    config = ExperimentConfig(body=body, seed=0, sim_duration=5.0)
    n = genotype_length_for(config)
    rng = np.random.default_rng(2)
    for _ in range(3):
        flat = rng.normal(scale=config.sigma0, size=n)
        assert evaluate(config, flat) == _reference_fitness(config, flat)
