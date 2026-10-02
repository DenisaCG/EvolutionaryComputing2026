"""Run one simulation episode and collect the metrics fitness needs.

`simple_runner` (ariel.utils.runners) is a black box: it steps MuJoCo until
`duration` and returns nothing. Per-step path-length and minimum core-height
tracking is instead done inside the `mjcb_control` callback, which MuJoCo
calls every physics step regardless of which runner drives time forward --
so `simple_runner` itself is used unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco as mj
import numpy as np
import numpy.typing as npt
from ariel.utils.runners import simple_runner

from bodies import build_robot, build_world
from config import ExperimentConfig
from controller import genotype_length, nn_controller, unflatten


@dataclass(frozen=True)
class EpisodeMetrics:
    """Raw measurements from one episode, before fitness is computed."""

    initial_position: npt.NDArray[np.float64]
    final_position: npt.NDArray[np.float64]
    total_path_length: float
    min_core_height: float


def probe_dimensions(config: ExperimentConfig) -> tuple[int, int]:
    """Build one instance of the world+robot to read (input_size, output_size).

    `input_size = len(data.qpos)` and `output_size = model.nu` both depend on
    the chosen body, so they must be read from a compiled model once before
    the genotype length (and CMA-ES's `n`) is known.
    """
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot(config.body)
    world.spawn(
        robot.spec,
        position=list(config.spawn_pos),
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)
    return len(data.qpos), model.nu


def genotype_length_for(config: ExperimentConfig) -> int:
    """Total flat-weight-vector length for this config's body."""
    input_size, output_size = probe_dimensions(config)
    return genotype_length(input_size, config.hidden_size, output_size)


def run_episode(
    config: ExperimentConfig,
    flat_weights: npt.NDArray[np.float64],
) -> EpisodeMetrics:
    """Simulate one episode with `flat_weights` and return its raw metrics."""
    mj.set_mjcb_control(None)

    world = build_world()
    robot = build_robot(config.body)
    world.spawn(
        robot.spec,
        position=list(config.spawn_pos),
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    input_size = len(data.qpos)
    output_size = model.nu
    weights = unflatten(flat_weights, input_size, config.hidden_size, output_size)

    initial_position = np.asarray(data.qpos[0:3]).copy()
    tracker = {
        "last_position": initial_position.copy(),
        "total_path_length": 0.0,
        "min_core_height": float(initial_position[2]),
    }

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        actions = nn_controller(m, d, weights)
        d.ctrl[:] = actions

        current_position = np.asarray(d.qpos[0:3]).copy()
        tracker["total_path_length"] += float(
            np.linalg.norm(current_position - tracker["last_position"])
        )
        tracker["last_position"] = current_position
        tracker["min_core_height"] = min(
            tracker["min_core_height"], float(current_position[2])
        )

    mj.set_mjcb_control(control_callback)
    simple_runner(model, data, duration=config.sim_duration)
    mj.set_mjcb_control(None)

    final_position = np.asarray(data.qpos[0:3]).copy()

    return EpisodeMetrics(
        initial_position=initial_position,
        final_position=final_position,
        total_path_length=tracker["total_path_length"],
        min_core_height=tracker["min_core_height"],
    )


def run_episode_trajectory(
    config: ExperimentConfig,
    flat_weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Replay one episode and return the full (x, y) core trajectory.

    Not used during evolution (only the aggregate metrics from `run_episode`
    are needed for fitness) -- this is for the post-hoc trajectory plot in
    `analysis/make_plots.py`, so a single best genome can be visually
    inspected without re-instrumenting the main evaluation path.
    """
    mj.set_mjcb_control(None)

    world = build_world()
    robot = build_robot(config.body)
    world.spawn(
        robot.spec,
        position=list(config.spawn_pos),
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    input_size = len(data.qpos)
    output_size = model.nu
    weights = unflatten(flat_weights, input_size, config.hidden_size, output_size)

    trajectory: list[npt.NDArray[np.float64]] = [np.asarray(data.qpos[0:3]).copy()]

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        actions = nn_controller(m, d, weights)
        d.ctrl[:] = actions
        trajectory.append(np.asarray(d.qpos[0:3]).copy())

    mj.set_mjcb_control(control_callback)
    simple_runner(model, data, duration=config.sim_duration)
    mj.set_mjcb_control(None)

    return np.array(trajectory)
