"""Qualitative check: render each condition's best body next to its closest target.

Reads the manifest analyze_results.py writes to
__results__/manifests/best_individuals.json (one best genome + its closest
target per condition) and writes one side-by-side comparison image per
condition to __results__/plots/. Purely qualitative - as the assignment
template warns, a fitness number going down is not evidence that the bodies
look anything like the targets; this is the check for that.

Usage
-----
    python analyze_results.py   # must run first, writes best_individuals.json
    python render_bodies.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco as mj
import networkx as nx
import numpy as np

from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer

from ea_common import TARGET_DIR

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "__results__"
PLOTS_DIR = RESULTS_DIR / "plots"
MANIFESTS_DIR = RESULTS_DIR / "manifests"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS: dict[str, str] = {
    "mutation_only": "Mutation-only EA",
    "mutation_crossover": "Mutation + crossover EA",
    "random_search": "Random search",
}

CAM_FOVY = 1.2  # orthographic zoom tuned for these bodies' size range


def render_graph(graph: nx.DiGraph) -> np.ndarray:
    """Build the body in MuJoCo and return a single rendered frame."""
    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(graph)
    world.spawn(
        robot.spec, position=[0.0, 0.0, 0.1], correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    image = single_frame_renderer(model, data, cam_fovy=CAM_FOVY)
    return np.asarray(image)


def main() -> None:
    mj.set_mjcb_control(None)  # global MuJoCo control callback - clear before use

    with (MANIFESTS_DIR / "best_individuals.json").open() as f:
        manifest = json.load(f)

    for condition, info in manifest.items():
        genome = TreeGenome.from_dict(info["genotype"])
        body_image = render_graph(genome.to_networkx())

        target_path = TARGET_DIR / f"{info['closest_target']}.json"
        target_image = render_graph(load_graph_from_json(target_path))

        fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
        axes[0].imshow(body_image)
        axes[0].set_title(
            f"{LABELS[condition]}\nbest evolved body (seed {info['best_seed']})",
        )
        axes[0].axis("off")

        axes[1].imshow(target_image)
        axes[1].set_title(
            f"Closest target: {info['closest_target']}\n"
            f"distance = {info['closest_target_distance']:.2f}",
        )
        axes[1].axis("off")

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"render_{condition}.png", dpi=200)
        plt.close(fig)

    print(f"wrote render_*.png to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
