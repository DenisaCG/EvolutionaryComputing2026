"""Watch a saved controller: python replay.py PATH_TO_RUN_OR_BEST_GENOME."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Run folder or best_genome.json path")
    parser.add_argument("--duration", type=float,
                        help="Simulation seconds (default: saved episode duration)")
    args = parser.parse_args()
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be a positive finite number")
    path = args.path.expanduser().resolve()
    genome_path = path / "best_genome.json" if path.is_dir() else path
    manifest_path = genome_path.parent / "manifest.json"
    for required in (genome_path, manifest_path):
        if not required.is_file():
            parser.error(f"File not found: {required}")
    try:
        settings = json.loads(manifest_path.read_text())["config"]
        genome = json.loads(genome_path.read_text())
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"Cannot read saved run: {error}")

    # Cocoa needs mjpython's UI thread on macOS. Relaunch automatically so
    # users can use the same `python replay.py ...` command on every platform.
    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        launcher = Path(sys.executable).parent / "mjpython"
        if not launcher.is_file():
            parser.error(f"mjpython not found beside Python: {launcher}")
        env = os.environ.copy()
        # uv's interpreter is symlinked into .venv. The shared library lives
        # beside the real interpreter, not in .venv/lib.
        library_dir = str(Path(sys.executable).resolve().parent.parent / "lib")
        previous = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = os.pathsep.join(
            [library_dir, previous] if previous else [library_dir]
        )
        os.execve(sys.executable, [sys.executable, str(launcher),
                  str(Path(__file__).resolve()), *sys.argv[1:]], env)

    import mujoco as mj
    import numpy as np
    from mujoco import viewer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from bodies import build_robot, build_world
    from config import ExperimentConfig
    from controller import controller_inputs, nn_controller, unflatten

    config_fields = {field.name for field in fields(ExperimentConfig)}
    config = ExperimentConfig(**{
        key: value for key, value in settings.items()
        if key in config_fields and key != "output_root"
    })
    duration = config.sim_duration if args.duration is None else args.duration
    mj.set_mjcb_control(None)
    world = build_world(config.terrain_seed)
    robot = build_robot(config.body)
    world.spawn(robot.spec, position=list(config.spawn_pos),
                correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    target = np.asarray(config.target_position[:2])
    weights = unflatten(
        np.asarray(genome["weights"]),
        len(controller_inputs(data, target, config.clock_hz)),
        config.hidden_size, model.nu,
    )

    def control(model, data):
        data.ctrl[:] = nn_controller(
            controller_inputs(data, target, config.clock_hz), weights
        )

    print(f"Replaying {genome_path}")
    print(f"Duration: {duration:g} seconds. Close the viewer to exit.")
    mj.set_mjcb_control(control)
    try:
        with viewer.launch_passive(model, data) as window:
            while window.is_running():
                started = time.perf_counter()
                with window.lock():
                    if data.time < duration:
                        mj.mj_step(model, data)
                window.sync()
                time.sleep(max(0, model.opt.timestep -
                               (time.perf_counter() - started)))
    finally:
        mj.set_mjcb_control(None)


if __name__ == "__main__":
    main()
