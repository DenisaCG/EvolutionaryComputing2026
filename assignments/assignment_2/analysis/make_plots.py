"""Generate every results plot for one experiment, styled consistently.

Run `aggregate_results.py --experiment <name>` first. Reads only
`results/<experiment>/` (the aggregated CSVs and the copied per-run files),
so plots can be regenerated from committed data. Each `plot_*` function
writes one PNG to `results/<experiment>/<body>/`.

    python analysis/make_plots.py --experiment ipop_l10_b8000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Offscreen MuJoCo rendering on Linux (headless or not) works via EGL; the
# default GLFW backend fails without a usable display. Must be set before
# mujoco is imported. macOS keeps its default (EGL is unavailable there).
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

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

ALGORITHM_ORDER = ["cma_es", "ipop_cma_es", "random_search"]
ALGORITHM_LABELS = {
    "cma_es": "CMA-ES",
    "ipop_cma_es": "IPOP-CMA-ES",
    "random_search": "Random search",
}
# Fixed per algorithm (not per plot) so colors match across every figure.
# IPOP is a deeper purple than the palette's lavender, and dashed in line
# plots: it is identical to CMA-ES until its first restart, so the lines
# overlap and must stay distinguishable.
ALGORITHM_COLORS = {"cma_es": PALETTE[0], "random_search": PALETTE[1], "ipop_cma_es": "#7E5FAF"}
ALGORITHM_LINESTYLES = {"cma_es": "-", "random_search": "-", "ipop_cma_es": "--"}
CMA_FAMILY = ["cma_es", "ipop_cma_es"]


class Plotter:
    def __init__(self, experiment: str) -> None:
        self.experiment = experiment
        self.root = ASSIGNMENT_ROOT / "results" / experiment
        self.generations = pl.read_csv(self.root / "aggregated_generations.csv")
        self.summary = pl.read_csv(self.root / "aggregated_summary.csv")

    # -- helpers ---------------------------------------------------------------

    def _save(self, fig: plt.Figure, body: str, name: str) -> None:
        out_dir = self.root / body
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"[plots] wrote {path}")

    def _run_dir(self, body: str, algorithm: str, seed: int) -> Path:
        return self.root / "runs" / f"{body}__{algorithm}" / f"seed_{seed}"

    def _manifest(self, body: str, algorithm: str, seed: int) -> dict:
        return json.loads((self._run_dir(body, algorithm, seed) / "manifest.json").read_text())

    def _config(self, body: str, algorithm: str, seed: int) -> ExperimentConfig:
        """Rebuild the exact evaluation config a run used (for trajectory replay)."""
        c = self._manifest(body, algorithm, seed)["config"]
        return ExperimentConfig(
            body=body,
            seed=seed,
            hidden_size=c["hidden_size"],
            clock_hz=c["clock_hz"],
            sim_duration=c["sim_duration"],
            spawn_pos=tuple(c["spawn_pos"]),
            target_position=tuple(c["target_position"]),
            # Manifests from before the terrain fix have no terrain_seed: their
            # runs saw random terrain, so replays can't reproduce them exactly.
            terrain_seed=c.get("terrain_seed"),
        )

    def _initial_distance(self, body: str) -> float:
        row = self.summary.filter(pl.col("body") == body).row(0, named=True)
        c = self._manifest(body, row["algorithm"], row["seed"])["config"]
        spawn, target = c["spawn_pos"], c["target_position"]
        return math.hypot(target[0] - spawn[0], target[1] - spawn[1])

    def _algorithms(self, df: pl.DataFrame) -> list[str]:
        present = set(df["algorithm"].unique())
        return [a for a in ALGORITHM_ORDER if a in present]

    def _n_seeds(self, body: str) -> int:
        return self.summary.filter(pl.col("body") == body)["seed"].n_unique()

    # -- plots -----------------------------------------------------------------

    def plot_convergence(self, body: str) -> None:
        """Required plot: mean +/- std best-so-far distance to target vs. evaluations.

        Runs log at different evaluation counts (IPOP's population grows at
        each restart), so each run's best-so-far is sampled as a step function
        on a common grid before averaging across seeds.
        """
        body_df = self.generations.filter(pl.col("body") == body)
        d0 = self._initial_distance(body)
        grid_step = int(body_df["lambda"].min())
        grid = np.arange(grid_step, int(body_df["evals_used"].max()) + 1, grid_step)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for algorithm in self._algorithms(body_df):
            curves = []
            for seed in body_df.filter(pl.col("algorithm") == algorithm)["seed"].unique().sort():
                run = body_df.filter(
                    (pl.col("algorithm") == algorithm) & (pl.col("seed") == seed)
                ).sort("evals_used")
                evals = run["evals_used"].to_numpy()
                best = run["best_so_far"].to_numpy() + d0
                idx = np.searchsorted(evals, grid, side="right") - 1
                # Past a run's last generation (IPOP can stop a few evals short
                # when the next population doesn't fit) its best-so-far stays
                # at its final value, so seeds keep being averaged together.
                curve = np.where(idx >= 0, best[np.clip(idx, 0, None)], np.nan)
                curves.append(curve)
            curves = np.array(curves)
            mean = np.nanmean(curves, axis=0)
            std = np.nan_to_num(np.nanstd(curves, axis=0, ddof=1 if len(curves) > 1 else 0))
            color = ALGORITHM_COLORS[algorithm]
            ax.plot(grid, mean, label=ALGORITHM_LABELS[algorithm], color=color,
                    linestyle=ALGORITHM_LINESTYLES[algorithm], linewidth=2)
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15)

        ax.axhline(d0, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel("Fitness evaluations")
        ax.set_ylabel("Best-so-far distance to target (m, lower is better)")
        legend_below(ax, ncol=3)
        decorate(
            fig,
            f"Convergence on {body.capitalize()}",
            subtitle=f"Mean ± std of best-so-far distance across {self._n_seeds(body)} seeds; "
            f"dotted line = starting distance ({d0:.1f} m)",
        )
        self._save(fig, body, "convergence")

    def plot_convergence_per_seed(self, body: str) -> None:
        """Best-so-far distance of every run in one plot (color = algorithm,
        line style = seed), with IPOP's restarts marked and the per-generation
        best of IPOP's current stage shown faintly, so restarted stages that
        never beat the best-so-far are still visible.
        """
        body_df = self.generations.filter(pl.col("body") == body)
        d0 = self._initial_distance(body)
        seed_styles = ["-", "--", ":", "-.", (0, (5, 1, 1, 1))]
        widths = {"cma_es": 3.2, "random_search": 1.8, "ipop_cma_es": 1.6}

        fig, ax = plt.subplots(figsize=(10, 6))
        for algorithm in self._algorithms(body_df):
            color = ALGORITHM_COLORS[algorithm]
            seeds = body_df.filter(pl.col("algorithm") == algorithm)["seed"].unique().sort()
            for i, seed in enumerate(seeds):
                run = body_df.filter(
                    (pl.col("algorithm") == algorithm) & (pl.col("seed") == seed)
                ).sort("evals_used")
                evals = run["evals_used"].to_numpy()
                style = seed_styles[i % len(seed_styles)]
                ax.step(evals, run["best_so_far"].to_numpy() + d0, where="post",
                        color=color, linestyle=style, linewidth=widths[algorithm],
                        label=f"{ALGORITHM_LABELS[algorithm]}, seed {seed}",
                        zorder=2 if algorithm == "cma_es" else 3)
                if algorithm != "ipop_cma_es":
                    continue
                ax.plot(evals, run["best_fitness"].to_numpy() + d0, color=color,
                        linestyle=style, linewidth=0.7, alpha=0.35, zorder=1)
                lam = run["lambda"].to_numpy()
                for e in evals[:-1][np.diff(lam) != 0]:
                    ax.axvline(e, color=color, linestyle=style, linewidth=1, alpha=0.6)

        ax.axhline(d0, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel("Fitness evaluations")
        ax.set_ylabel("Distance to target (m, lower is better)")
        legend_below(ax, ncol=3)
        decorate(
            fig,
            f"Convergence per Seed on {body.capitalize()}",
            subtitle="Thick lines: best-so-far per run (line style = seed). Vertical lines: "
            "IPOP restarts. Faint purple: best of each IPOP generation (current stage).",
        )
        self._save(fig, body, "convergence_per_seed")

    def plot_generation_best(self, body: str) -> None:
        """Best fitness of each generation (not best-so-far), one panel per seed.

        CMA-ES and IPOP from the same seed are identical until IPOP's first
        restart; after it, each restart shows as a jump back up (a fresh
        search with doubled lambda) followed by a new descent. Shows why
        IPOP's best-so-far stays flat when no restarted stage beats it.
        """
        body_df = self.generations.filter(
            (pl.col("body") == body) & pl.col("algorithm").is_in(CMA_FAMILY)
        )
        if body_df.filter(pl.col("algorithm") == "ipop_cma_es").is_empty():
            return
        d0 = self._initial_distance(body)
        seeds = body_df.filter(pl.col("algorithm") == "ipop_cma_es")["seed"].unique().sort()

        fig, axes = plt.subplots(len(seeds), 1, figsize=(10, 3.6 * len(seeds)),
                                 squeeze=False, sharex=True, sharey=True)
        for ax, seed in zip(axes[:, 0], seeds, strict=True):
            for algorithm in self._algorithms(body_df):
                run = body_df.filter(
                    (pl.col("algorithm") == algorithm) & (pl.col("seed") == seed)
                ).sort("evals_used")
                if run.is_empty():
                    continue
                ax.plot(run["evals_used"], run["best_fitness"].to_numpy() + d0,
                        color=ALGORITHM_COLORS[algorithm],
                        linewidth=1.8 if algorithm == "cma_es" else 1.1,
                        alpha=0.9, label=ALGORITHM_LABELS[algorithm],
                        zorder=2 if algorithm == "cma_es" else 3)
                if algorithm != "ipop_cma_es":
                    continue
                evals = run["evals_used"].to_numpy()
                lam = run["lambda"].to_numpy()
                starts = [0, *evals[:-1][np.diff(lam) != 0]]
                for start, stage_lam in zip(starts, [lam[0], *lam[1:][np.diff(lam) != 0]],
                                            strict=True):
                    if start > 0:
                        ax.axvline(start, color=ALGORITHM_COLORS[algorithm],
                                   linestyle="--", linewidth=1.2)
                    ax.annotate(f"λ={stage_lam}", xy=(start, 1), xycoords=("data", "axes fraction"),
                                xytext=(4, -4), textcoords="offset points", va="top",
                                fontsize=10, color=ALGORITHM_COLORS[algorithm])
            ax.set_title(f"Seed {seed}", fontsize=12, loc="left")
            ax.set_ylabel("Generation best (m)")
        axes[-1, 0].set_xlabel("Fitness evaluations")
        legend_below(axes[-1, 0], ncol=2)
        decorate(
            fig,
            f"Best Fitness per Generation on {body.capitalize()}",
            subtitle="Distance to target of each generation's best controller (lower is "
            "better); dashed lines = IPOP restarts with doubled λ",
            has_legend=True,
        )
        self._save(fig, body, "generation_best")

    def plot_population_size(self, body: str) -> None:
        """IPOP population size (lambda) over evaluations; each step is a restart."""
        ipop = self.generations.filter(
            (pl.col("body") == body) & (pl.col("algorithm") == "ipop_cma_es")
        )
        if ipop.is_empty():
            return
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for i, seed in enumerate(ipop["seed"].unique().sort()):
            run = ipop.filter(pl.col("seed") == seed).sort("evals_used")
            ax.step(run["evals_used"], run["lambda"], where="pre",
                    label=f"seed {seed}", color=PALETTE[i % len(PALETTE)])
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Fitness evaluations")
        ax.set_ylabel("Population size λ (log₂ scale)")
        legend_below(ax, ncol=5)
        decorate(
            fig,
            f"IPOP-CMA-ES Population Size on {body.capitalize()}",
            subtitle="One line per seed; each step up is a restart with doubled λ",
        )
        self._save(fig, body, "ipop_population_size")

    def plot_sigma_evolution(self, body: str) -> None:
        """Step size per seed, one panel per CMA-ES variant (restarts reset sigma)."""
        body_df = self.generations.filter(
            (pl.col("body") == body) & pl.col("algorithm").is_in(CMA_FAMILY)
        )
        algorithms = self._algorithms(body_df)
        if not algorithms:
            return
        fig, axes = plt.subplots(1, len(algorithms), figsize=(7 * len(algorithms), 5.5),
                                 squeeze=False, sharey=True)
        for ax, algorithm in zip(axes[0], algorithms, strict=True):
            alg_df = body_df.filter(pl.col("algorithm") == algorithm)
            for i, seed in enumerate(alg_df["seed"].unique().sort()):
                run = alg_df.filter(pl.col("seed") == seed).sort("evals_used")
                ax.plot(run["evals_used"], run["sigma"], label=f"seed {seed}",
                        color=PALETTE[i % len(PALETTE)])
            ax.set_title(ALGORITHM_LABELS[algorithm], fontsize=13)
            ax.set_xlabel("Fitness evaluations")
        axes[0][0].set_ylabel("Step size σ")
        legend_below(axes[0][0], ncol=5)
        decorate(
            fig,
            f"CMA-ES Step Size Over Time on {body.capitalize()}",
            subtitle="One line per seed; IPOP resets σ to σ₀ at each restart",
            has_legend=True,
        )
        self._save(fig, body, "sigma_evolution")

    def plot_final_distance(self, body: str) -> None:
        """Box plot of each run's final best distance to target, per algorithm."""
        body_df = self.summary.filter(pl.col("body") == body)
        algorithms = self._algorithms(body_df)
        data = [
            body_df.filter(pl.col("algorithm") == a)["final_distance_m"].drop_nulls().to_numpy()
            for a in algorithms
        ]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        bp = ax.boxplot(data, tick_labels=[ALGORITHM_LABELS[a] for a in algorithms],
                        patch_artist=True, widths=0.5)
        rng = np.random.default_rng(0)
        for i, (patch, algorithm, values) in enumerate(
            zip(bp["boxes"], algorithms, data, strict=True), start=1
        ):
            patch.set_facecolor(ALGORITHM_COLORS[algorithm])
            patch.set_alpha(0.5)
            ax.scatter(i + rng.uniform(-0.08, 0.08, len(values)), values,
                       color=ALGORITHM_COLORS[algorithm], edgecolor="white", zorder=3)
        ax.set_ylabel("Final best distance to target (m, lower is better)")
        decorate(
            fig,
            f"Final Distance to Target on {body.capitalize()}",
            subtitle=f"One point per seed ({self._n_seeds(body)} seeds), at the end of the budget",
        )
        self._save(fig, body, "final_distance_distribution")

    def plot_wall_clock(self, body: str) -> None:
        """Wall-clock duration per algorithm, one point per seed."""
        body_df = self.summary.filter(pl.col("body") == body)
        algorithms = self._algorithms(body_df)
        data = [
            (body_df.filter(pl.col("algorithm") == a)["wall_clock_duration_s"] / 60).to_numpy()
            for a in algorithms
        ]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        bp = ax.boxplot(data, tick_labels=[ALGORITHM_LABELS[a] for a in algorithms],
                        patch_artist=True, widths=0.5)
        rng = np.random.default_rng(0)
        for i, (patch, algorithm, values) in enumerate(
            zip(bp["boxes"], algorithms, data, strict=True), start=1
        ):
            patch.set_facecolor(ALGORITHM_COLORS[algorithm])
            patch.set_alpha(0.5)
            ax.scatter(i + rng.uniform(-0.08, 0.08, len(values)), values,
                       color=ALGORITHM_COLORS[algorithm], edgecolor="white", zorder=3)
        ax.set_ylabel("Wall-clock duration (minutes)")
        decorate(
            fig,
            f"Per-Run Wall-Clock Time on {body.capitalize()}",
            subtitle="One point per seed; runs executed in parallel on the machine "
            "recorded in each manifest.json",
        )
        self._save(fig, body, "wall_clock_time")

    def plot_best_trajectory(self, body: str) -> None:
        """(x, y) path of every seed's best genome, one panel per algorithm."""
        body_df = self.summary.filter(pl.col("body") == body)
        algorithms = self._algorithms(body_df)
        fig, axes = plt.subplots(1, len(algorithms), figsize=(7 * len(algorithms), 6),
                                 squeeze=False)
        for ax, algorithm in zip(axes[0], algorithms, strict=True):
            for i, seed in enumerate(
                body_df.filter(pl.col("algorithm") == algorithm)["seed"].sort()
            ):
                genome = json.loads(
                    (self._run_dir(body, algorithm, seed) / "best_genome.json").read_text()
                )
                config = self._config(body, algorithm, seed)
                trajectory = run_episode_trajectory(config, np.array(genome["weights"]))
                color = PALETTE[i % len(PALETTE)]
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=1.3,
                        label=f"seed {seed}")
                ax.scatter(*trajectory[0, :2], color=color, s=30, zorder=3)
            ax.scatter(config.target_position[0], config.target_position[1], color="black",
                       marker="*", s=220, zorder=4, label="Target")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(ALGORITHM_LABELS[algorithm], fontsize=13)
            ax.set_aspect("equal", adjustable="datalim")
        legend_below(axes[0][0], ncol=6)
        decorate(
            fig,
            f"Best-Per-Seed Trajectories on {body.capitalize()}",
            subtitle="One line per seed (its best-found genome); dot = spawn, star = target. "
            "Axes are world-frame meters.",
            has_legend=True,
        )
        self._save(fig, body, "best_trajectory")

    def plot_environment_snapshots(self, body: str) -> None:
        """Top-down and two angled renders of the world + spawned body."""
        config = ExperimentConfig(body=body, seed=0)
        out_dir = self.root / body / "environment_snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        render_environment_snapshot(config).save(out_dir / "environment_snapshot.png")
        render_environment_angled(config, camera="pretty-cam").save(out_dir / "pretty_cam.png")
        side_camera = free_camera(azimuth=120, elevation=-20, distance=4.0,
                                  lookat=(0.5, 0.0, 0.2))
        render_environment_angled(config, camera=side_camera).save(out_dir / "side_angle.png")
        print(f"[plots] wrote {out_dir}/*.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--skip-renders", action="store_true",
                        help="Skip trajectory replays and environment renders (slow).")
    args = parser.parse_args()

    if not (ASSIGNMENT_ROOT / "results" / args.experiment / "aggregated_summary.csv").exists():
        print(f"[plots] run analysis/aggregate_results.py --experiment {args.experiment} first.")
        return

    apply_style()
    plotter = Plotter(args.experiment)
    for body in plotter.summary["body"].unique().sort():
        plotter.plot_convergence(body)
        plotter.plot_convergence_per_seed(body)
        plotter.plot_generation_best(body)
        plotter.plot_population_size(body)
        plotter.plot_sigma_evolution(body)
        plotter.plot_final_distance(body)
        plotter.plot_wall_clock(body)
        if not args.skip_renders:
            plotter.plot_best_trajectory(body)
            plotter.plot_environment_snapshots(body)


if __name__ == "__main__":
    main()
