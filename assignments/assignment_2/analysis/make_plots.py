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
from simulate import run_episode_trajectory  # noqa: E402

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
    """Wall-clock duration per run -- documents hardware feasibility."""
    body_df = summary.filter(pl.col("body") == body).sort(["algorithm", "seed"])
    if body_df.is_empty():
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    algorithms = sorted(body_df["algorithm"].unique())
    x_labels = []
    values = []
    colors = []
    for i, algorithm in enumerate(algorithms):
        alg_df = body_df.filter(pl.col("algorithm") == algorithm).sort("seed")
        for row in alg_df.iter_rows(named=True):
            x_labels.append(f"{ALGORITHM_LABELS.get(algorithm, algorithm)}\nseed {row['seed']}")
            values.append(row["wall_clock_duration_s"] / 60.0)
            colors.append(PALETTE[i])

    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Wall-clock duration (minutes)")
    decorate(
        fig,
        f"Per-Run Wall-Clock Time on {body.capitalize()}",
        subtitle="Measured on the machine that produced this data (see each run's manifest.json)",
    )
    _savefig(fig, body, "wall_clock_time")


def plot_best_trajectory(summary: pl.DataFrame, body: str) -> None:
    """(x, y) path of the single best genome found for this body, any algorithm/seed."""
    body_df = summary.filter(pl.col("body") == body)
    if body_df.is_empty():
        return
    best_row = body_df.sort("best_fitness").row(0, named=True)

    run_dir = (
        DATA_ROOT
        / f"{best_row['body']}__{best_row['algorithm']}"
        / f"seed_{best_row['seed']}"
    )
    genome = json.loads((run_dir / "best_genome.json").read_text())
    weights = np.array(genome["weights"])

    config = ExperimentConfig(body=body, seed=best_row["seed"])
    trajectory = run_episode_trajectory(config, weights)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=PALETTE[0], linewidth=1.5)
    ax.scatter(*trajectory[0, :2], color=PALETTE[2], s=80, zorder=3, label="Spawn")
    ax.scatter(
        config.target_position[0],
        config.target_position[1],
        color=PALETTE[5],
        marker="*",
        s=200,
        zorder=3,
        label="Target",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    legend_below(ax, ncol=2)
    decorate(
        fig,
        f"Best {body.capitalize()} Trajectory "
        f"({ALGORITHM_LABELS.get(best_row['algorithm'], best_row['algorithm'])}, "
        f"seed {best_row['seed']})",
        subtitle=f"Fitness = {best_row['best_fitness']:.4f} (lower is better)",
    )
    _savefig(fig, body, "best_trajectory")


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


if __name__ == "__main__":
    main()
