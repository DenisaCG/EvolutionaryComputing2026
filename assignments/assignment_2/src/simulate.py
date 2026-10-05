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
from ariel.utils.renderers import single_frame_renderer
from ariel.utils.runners import simple_runner
from PIL import Image

# Every ariel world (ariel.simulation.environments._base_world.BaseWorld)
# defines these two cameras in addition to the default top-down "ortho-cam"
# that `single_frame_renderer` uses: an angled perspective view, useful for a
# report figure that actually shows the terrain's shape (ramp, rugged bumps)
# rather than a flat top-down silhouette.
ANGLED_CAMERA_NAME = "pretty-cam"

from bodies import build_robot, build_world
from config import ExperimentConfig
from controller import controller_inputs, genotype_length, nn_controller, unflatten


@dataclass(frozen=True)
class EpisodeMetrics:
    """Raw measurements from one episode, before fitness is computed."""

    initial_position: npt.NDArray[np.float64]
    final_position: npt.NDArray[np.float64]
    total_path_length: float
    min_core_height: float


def probe_dimensions(config: ExperimentConfig) -> tuple[int, int]:
    """Build one instance of the world+robot to read (input_size, output_size).

    `input_size` (length of `controller_inputs`) and `output_size = model.nu`
    both depend on the chosen body, so they must be read from a compiled
    model once before the genotype length (and CMA-ES's `n`) is known.
    """
    mj.set_mjcb_control(None)
    world = build_world(config.terrain_seed)
    robot = build_robot(config.body)
    world.spawn(
        robot.spec,
        position=list(config.spawn_pos),
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)
    inputs = controller_inputs(data, _target_xy(config), config.clock_hz)
    return len(inputs), model.nu


def _target_xy(config: ExperimentConfig) -> npt.NDArray[np.float64]:
    return np.asarray(config.target_position[:2])


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

    world = build_world(config.terrain_seed)
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

    target_xy = _target_xy(config)
    input_size = len(controller_inputs(data, target_xy, config.clock_hz))
    output_size = model.nu
    weights = unflatten(flat_weights, input_size, config.hidden_size, output_size)

    initial_position = np.asarray(data.qpos[0:3]).copy()
    tracker = {
        "last_position": initial_position.copy(),
        "total_path_length": 0.0,
        "min_core_height": float(initial_position[2]),
    }

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        actions = nn_controller(
            controller_inputs(d, target_xy, config.clock_hz), weights
        )
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

    world = build_world(config.terrain_seed)
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

    target_xy = _target_xy(config)
    input_size = len(controller_inputs(data, target_xy, config.clock_hz))
    output_size = model.nu
    weights = unflatten(flat_weights, input_size, config.hidden_size, output_size)

    trajectory: list[npt.NDArray[np.float64]] = [np.asarray(data.qpos[0:3]).copy()]

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        actions = nn_controller(
            controller_inputs(d, target_xy, config.clock_hz), weights
        )
        d.ctrl[:] = actions
        trajectory.append(np.asarray(d.qpos[0:3]).copy())

    mj.set_mjcb_control(control_callback)
    simple_runner(model, data, duration=config.sim_duration)
    mj.set_mjcb_control(None)

    return np.array(trajectory)


def render_environment_snapshot(config: ExperimentConfig) -> Image.Image:
    """Static image of the world + spawned (unposed) robot, for the report.

    Purely visual context (what does the Olympic Arena + this body actually
    look like?) -- no controller is run, the robot is just spawned at
    `config.spawn_pos` and the scene is rendered as-is.
    """
    mj.set_mjcb_control(None)

    world = build_world(config.terrain_seed)
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

    return single_frame_renderer(model, data, steps=1)


def render_environment_angled(
    config: ExperimentConfig,
    camera: str | mj.MjvCamera = ANGLED_CAMERA_NAME,
    width: int = 480,
    height: int = 640,
) -> Image.Image:
    """Angled-perspective render of the world + spawned (unposed) robot.

    `single_frame_renderer` only ever looks up the world's top-down
    "ortho-cam", so an angled view needs its own renderer call. `camera` can
    be another named camera every ariel world defines (`"pretty-cam"`, a
    perspective view) or a custom `mujoco.MjvCamera` (e.g. a free camera at a
    chosen azimuth/elevation) for a specific angle.
    """
    mj.set_mjcb_control(None)

    world = build_world(config.terrain_seed)
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

    with mj.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()

    return Image.fromarray(frame)


def free_camera(
    azimuth: float,
    elevation: float,
    distance: float,
    lookat: tuple[float, float, float],
) -> mj.MjvCamera:
    """Build a free (not world-defined) camera at a chosen angle, for variety
    beyond the world's single built-in "pretty-cam" perspective.
    """
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    camera.lookat[:] = lookat
    return camera
