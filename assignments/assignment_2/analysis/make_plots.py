"""Generate every results plot from the aggregated CSVs, styled consistently.

Run `aggregate_results.py` first. Each `plot_*` function is independent and
writes one PNG to `results/<body>/`; `main()` just calls all of them for
every body found in the data.

    uv run python analysis/make_plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ASSIGNMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ASSIGNMENT_ROOT))
sys.path.insert(0, str(ASSIGNMENT_ROOT / "src"))

from style_guidelines_plots import (  # noqa: E402
    PALETTE,
    apply_style,
    decorate,
    legend_below,
)

from config import ExperimentConfig  # noqa: E402
from simulate import (  # noqa: E402
    free_camera,
    render_environment_angled,
    render_environment_snapshot,
    run_episode_trajectory,
)

DATA_ROOT = ASSIGNMENT_ROOT / "__data__"
RESULTS_ROOT = ASSIGNMENT_ROOT / "results"

ALGORITHM_LABELS = {"cma_es": "CMA-ES", "random_search": "Random search"}


def _savefig(fig: plt.Figure, body: str, name: str) -> None:
    out_dir = RESULTS_ROOT / body
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[plots] wrote {path}")


def plot_convergence(generations: pl.DataFrame, body: str) -> None:
    """Required plot: mean +/- std fitness vs. evaluation count, per algorithm."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    body_df = generations.filter(pl.col("body") == body)

    for i, algorithm in enumerate(sorted(body_df["algorithm"].unique())):
        alg_df = body_df.filter(pl.col("algorithm") == algorithm)
        stats = (
            alg_df.group_by("evals_used")
            .agg(
                pl.col("best_fitness").mean().alias("mean"),
                pl.col("best_fitness").std().alias("std"),
            )
            .sort("evals_used")
        )
        x = stats["evals_used"].to_numpy()
        mean = stats["mean"].to_numpy()
        std = np.nan_to_num(stats["std"].to_numpy())
        color = PALETTE[i]
        ax.plot(x, mean, label=ALGORITHM_LABELS.get(algorithm, algorithm), color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("Best fitness (lower is better)")
    legend_below(ax, ncol=2)
    decorate(
        fig,
        f"CMA-ES vs. Random Search Convergence on {body.capitalize()}",
        subtitle="Mean ± std best fitness per evaluation count, across 5 seeds",
    )
    _savefig(fig, body, "convergence")


def plot_convergence_logscale(generations: pl.DataFrame, body: str) -> None:
    """Best-fitness-only convergence, log scale (echoes the paper's Figure 1)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    body_df = generations.filter(pl.col("body") == body)

    for i, algorithm in enumerate(sorted(body_df["algorithm"].unique())):
        alg_df = body_df.filter(pl.col("algorithm") == algorithm)
        stats = (
            alg_df.group_by("evals_used")
            .agg(pl.col("best_fitness").median().alias("median"))
            .sort("evals_used")
        )
        x = stats["evals_used"].to_numpy()
        y = np.abs(stats["median"].to_numpy()) + 1e-12
        ax.plot(
            x, y, label=ALGORITHM_LABELS.get(algorithm, algorithm), color=PALETTE[i]
        )

    ax.set_yscale("log")
    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("|Median best fitness| (log scale)")
    legend_below(ax, ncol=2)
    decorate(
        fig,
        f"Median Best-Fitness Convergence on {body.capitalize()}",
        subtitle="Log-scale view across 5 seeds, in the style of Auger & Hansen (2005) Fig. 1",
    )
    _savefig(fig, body, "convergence_logscale")


def plot_sigma_evolution(generations: pl.DataFrame, body: str) -> None:
    """CMA-ES step-size (sigma) trace per seed -- internal diagnostic."""
    cma_df = generations.filter(
        (pl.col("body") == body) & (pl.col("algorithm") == "cma_es")
    )
    if cma_df.is_empty():
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, seed in enumerate(sorted(cma_df["seed"].unique())):
        seed_df = cma_df.filter(pl.col("seed") == seed).sort("evals_used")
        ax.plot(
            seed_df["evals_used"],
            seed_df["sigma"],
            label=f"seed {seed}",
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("Step size (sigma)")
    legend_below(ax, ncol=5)
    decorate(
        fig,
        f"CMA-ES Step Size Over Time on {body.capitalize()}",
        subtitle="One line per independent seed",
    )
    _savefig(fig, body, "sigma_evolution")


def plot_final_fitness_distribution(summary: pl.DataFrame, body: str) -> None:
    """Box plot of final best fitness per algorithm, across seeds."""
    body_df = summary.filter(pl.col("body") == body)
    algorithms = sorted(body_df["algorithm"].unique())
    data = [
        body_df.filter(pl.col("algorithm") == alg)["best_fitness"].to_numpy()
        for alg in algorithms
    ]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bp = ax.boxplot(
        data,
        tick_labels=[ALGORITHM_LABELS.get(a, a) for a in algorithms],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], PALETTE, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Final best fitness (lower is better)")
    decorate(
        fig,
        f"Final Fitness Spread Across 5 Seeds on {body.capitalize()}",
        subtitle="Box plot of the best fitness reached by each independent run",
    )
    _savefig(fig, body, "final_fitness_distribution")


def plot_wall_clock(summary: pl.DataFrame, body: str) -> None:
    """Wall-clock duration per algorithm -- box plot (not one bar per run,
    which overlapped unreadably once there were 5 seeds per algorithm) with
    individual seed durations overlaid as jittered points.
    """
    body_df = summary.filter(pl.col("body") == body)
    if body_df.is_empty():
        return

    algorithms = sorted(body_df["algorithm"].unique())
    data = [
        (
            body_df.filter(pl.col("algorithm") == alg)["wall_clock_duration_s"]
            / 60.0
        ).to_numpy()
        for alg in algorithms
    ]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bp = ax.boxplot(
        data,
        tick_labels=[ALGORITHM_LABELS.get(a, a) for a in algorithms],
        patch_artist=True,
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], PALETTE, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            color=PALETTE[i - 1],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_ylabel("Wall-clock duration (minutes)")
    decorate(
        fig,
        f"Per-Run Wall-Clock Time on {body.capitalize()}",
        subtitle="One point per seed; measured on the machine that produced this "
        "data (hardware specs in each run's manifest.json)",
    )
    _savefig(fig, body, "wall_clock_time")


def plot_best_trajectory(summary: pl.DataFrame, body: str) -> None:
    """(x, y) path of the best genome from EVERY seed, one panel per algorithm.

    Each panel overlays all 5 seeds' best-found trajectory (not just the
    single best-of-all-runs) so seed-to-seed variability in the resulting
    gait/path is visible, not just the single luckiest run.
    """
    body_df = summary.filter(pl.col("body") == body)
    if body_df.is_empty():
        return

    algorithms = sorted(body_df["algorithm"].unique())
    fig, axes = plt.subplots(1, len(algorithms), figsize=(7 * len(algorithms), 6), squeeze=False)
    axes = axes[0]

    config = ExperimentConfig(body=body, seed=0)
    target_xy = (config.target_position[0], config.target_position[1])

    for ax, algorithm in zip(axes, algorithms, strict=False):
        alg_df = body_df.filter(pl.col("algorithm") == algorithm).sort("seed")
        for i, row in enumerate(alg_df.iter_rows(named=True)):
            run_dir = DATA_ROOT / f"{body}__{algorithm}" / f"seed_{row['seed']}"
            genome = json.loads((run_dir / "best_genome.json").read_text())
            weights = np.array(genome["weights"])

            seed_config = ExperimentConfig(body=body, seed=row["seed"])
            trajectory = run_episode_trajectory(seed_config, weights)
            color = PALETTE[i % len(PALETTE)]
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                linewidth=1.3,
                label=f"seed {row['seed']}",
            )
            ax.scatter(*trajectory[0, :2], color=color, s=30, zorder=3)

        ax.scatter(
            *target_xy,
            color="black",
            marker="*",
            s=220,
            zorder=4,
            label="Target",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(ALGORITHM_LABELS.get(algorithm, algorithm), fontsize=13)
        ax.set_aspect("equal", adjustable="datalim")

    # Seed colors/labels are identical across panels (both algorithms run the
    # same 5 seeds), so one shared legend (built from the first panel) covers
    # both -- legend_below queues a single figure-level legend, so calling it
    # per-axis would just have the last call silently win.
    legend_below(axes[0], ncol=6)

    decorate(
        fig,
        f"Best-Per-Seed Trajectories on {body.capitalize()}",
        subtitle="One line per seed (its best-found genome); dot = spawn, star = target. "
        "Axes are world-frame meters.",
        has_legend=True,
    )
    _savefig(fig, body, "best_trajectory")


def plot_environment_snapshots(body: str) -> None:
    """Renders of the world + spawned (unposed) body: top-down plus two
    angled perspectives, all under one `environment_snapshots/` directory.
    The top-down view shows terrain layout clearly but flattens the
    ramp/rugged bumps; the angled ones show the actual 3D shape of the course.
    """
    config = ExperimentConfig(body=body, seed=0)
    out_dir = RESULTS_ROOT / body / "environment_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    top_down_image = render_environment_snapshot(config)
    top_down_path = out_dir / "environment_snapshot.png"
    top_down_image.save(top_down_path)
    print(f"[plots] wrote {top_down_path}")

    pretty_cam_image = render_environment_angled(config, camera="pretty-cam")
    pretty_cam_path = out_dir / "pretty_cam.png"
    pretty_cam_image.save(pretty_cam_path)
    print(f"[plots] wrote {pretty_cam_path}")

    side_camera = free_camera(
        azimuth=120, elevation=-20, distance=4.0, lookat=(0.5, 0.0, 0.2)
    )
    side_image = render_environment_angled(config, camera=side_camera)
    side_path = out_dir / "side_angle.png"
    side_image.save(side_path)
    print(f"[plots] wrote {side_path}")


def main() -> None:
    apply_style()

    generations_path = DATA_ROOT / "aggregated_generations.csv"
    summary_path = DATA_ROOT / "aggregated_summary.csv"
    if not generations_path.exists() or not summary_path.exists():
        print("[plots] run analysis/aggregate_results.py first.")
        return

    generations = pl.read_csv(generations_path)
    summary = pl.read_csv(summary_path)

    for body in sorted(summary["body"].unique()):
        plot_convergence(generations, body)
        plot_convergence_logscale(generations, body)
        plot_sigma_evolution(generations, body)
        plot_final_fitness_distribution(summary, body)
        plot_wall_clock(summary, body)
        plot_best_trajectory(summary, body)
        plot_environment_snapshots(body)


if __name__ == "__main__":
    main()
