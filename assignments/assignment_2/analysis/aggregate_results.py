"""Load every run of one experiment into tidy tables, plus summary statistics.

Scans `__data__/<experiment>/<body>__<algorithm>/seed_<n>/` and writes, to
`results/<experiment>/` (small enough to commit, unlike `__data__/`):
  - `aggregated_generations.csv`: one row per generation per run, with the
    run's best-so-far fitness and the population size of that generation
    (for convergence, lambda and sigma plots);
  - `aggregated_summary.csv`: one row per run (final fitness, final distance
    to target, restarts, timing);
  - `summary_stats.csv`: mean/std/median per body x algorithm of the final
    distance to target, and `pairwise_tests.csv`: two-sided Mann-Whitney U
    tests between algorithms;
  - `runs/<body>__<algorithm>/seed_<n>/`: a copy of each run's
    generations.csv, manifest.json and best_genome.json, so plots (including
    trajectory replays) can be regenerated without `__data__/`.

    python analysis/aggregate_results.py --experiment ipop_l10_b8000
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shutil
import sys
from pathlib import Path

import polars as pl
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import ASSIGNMENT_ROOT  # noqa: E402

CONDITION_RE = re.compile(r"^(?P<body>[a-z]+)__(?P<algorithm>[a-z_]+)$")
SEED_RE = re.compile(r"^seed_(?P<seed>\d+)$")
RUN_FILES = ("generations.csv", "manifest.json", "best_genome.json")
FALL_PENALTY = 10.0  # fitness_survival_and_locomotion's flat penalty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    return parser.parse_args()


def find_runs(data_root: Path) -> list[tuple[Path, str, str, int]]:
    """(run_dir, body, algorithm, seed) for every finished run (has a manifest)."""
    runs = []
    for manifest_path in sorted(data_root.glob("*/seed_*/manifest.json")):
        run_dir = manifest_path.parent
        condition = CONDITION_RE.match(run_dir.parent.name)
        seed = SEED_RE.match(run_dir.name)
        if condition and seed:
            runs.append((run_dir, condition["body"], condition["algorithm"], int(seed["seed"])))
    return runs


def initial_distance(manifest: dict) -> float:
    """Planar spawn-to-target distance; fitness = final distance - this."""
    spawn, target = manifest["config"]["spawn_pos"], manifest["config"]["target_position"]
    return math.hypot(target[0] - spawn[0], target[1] - spawn[1])


def load_generations(runs: list[tuple[Path, str, str, int]]) -> pl.DataFrame:
    frames = []
    for run_dir, body, algorithm, seed in runs:
        df = pl.read_csv(run_dir / "generations.csv").sort("generation")
        frames.append(
            df.with_columns(
                pl.lit(body).alias("body"),
                pl.lit(algorithm).alias("algorithm"),
                pl.lit(seed).alias("seed"),
                pl.col("best_fitness").cum_min().alias("best_so_far"),
                pl.col("evals_used").diff().fill_null(pl.col("evals_used")).alias("lambda"),
            )
        )
    return pl.concat(frames, how="vertical") if frames else pl.DataFrame()


def load_summary(runs: list[tuple[Path, str, str, int]]) -> pl.DataFrame:
    rows = []
    for run_dir, body, algorithm, seed in runs:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        best_fitness = json.loads((run_dir / "best_genome.json").read_text())["fitness"]
        fell = best_fitness >= FALL_PENALTY
        restarts = manifest.get("restart_history") or []
        first_stop = manifest.get("first_local_stop")
        rows.append({
            "body": body,
            "algorithm": algorithm,
            "seed": seed,
            "best_fitness": best_fitness,
            "final_distance_m": None if fell else best_fitness + initial_distance(manifest),
            "total_evals": manifest["total_evals"],
            "n_restarts": len(restarts),
            "final_lambda": restarts[-1]["new_lambda"] if restarts else manifest["config"]["lambda_"],
            "first_local_stop_evals": first_stop["evals_used"] if first_stop else None,
            "wall_clock_duration_s": manifest["wall_clock_duration_s"],
            "termination_reason": manifest["termination_reason"],
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def summary_stats(summary: pl.DataFrame) -> pl.DataFrame:
    return (
        summary.group_by("body", "algorithm")
        .agg(
            pl.len().alias("n_runs"),
            pl.col("final_distance_m").mean().alias("mean_final_distance_m"),
            pl.col("final_distance_m").std().alias("std_final_distance_m"),
            pl.col("final_distance_m").median().alias("median_final_distance_m"),
            pl.col("final_distance_m").min().alias("best_final_distance_m"),
            pl.col("n_restarts").mean().alias("mean_restarts"),
        )
        .sort("body", "algorithm")
    )


def pairwise_tests(summary: pl.DataFrame) -> pl.DataFrame:
    """Two-sided Mann-Whitney U on final distance. Needs >= 2 runs per side;
    with few seeds the p-values are uninformative (minimum attainable p is large).
    """
    rows = []
    for body in summary["body"].unique().sort():
        body_df = summary.filter(pl.col("body") == body)
        for a, b in itertools.combinations(body_df["algorithm"].unique().sort(), 2):
            xa = body_df.filter(pl.col("algorithm") == a)["final_distance_m"].drop_nulls()
            xb = body_df.filter(pl.col("algorithm") == b)["final_distance_m"].drop_nulls()
            if len(xa) < 2 or len(xb) < 2:
                continue
            test = mannwhitneyu(xa.to_numpy(), xb.to_numpy(), alternative="two-sided")
            rows.append({
                "body": body, "algorithm_a": a, "algorithm_b": b,
                "n_a": len(xa), "n_b": len(xb),
                "U": float(test.statistic), "p_value": float(test.pvalue),
            })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def main() -> None:
    args = parse_args()
    data_root = ASSIGNMENT_ROOT / "__data__" / args.experiment
    results_root = ASSIGNMENT_ROOT / "results" / args.experiment
    runs = find_runs(data_root)
    if not runs:
        print(f"[aggregate] no finished runs under {data_root}. Run experiments first.")
        return

    results_root.mkdir(parents=True, exist_ok=True)
    for run_dir, body, algorithm, seed in runs:
        dest = results_root / "runs" / f"{body}__{algorithm}" / f"seed_{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        for name in RUN_FILES:
            shutil.copy2(run_dir / name, dest / name)

    generations = load_generations(runs)
    summary = load_summary(runs)
    generations.write_csv(results_root / "aggregated_generations.csv")
    summary.write_csv(results_root / "aggregated_summary.csv")
    stats = summary_stats(summary)
    stats.write_csv(results_root / "summary_stats.csv")
    tests = pairwise_tests(summary)
    if not tests.is_empty():
        tests.write_csv(results_root / "pairwise_tests.csv")

    print(f"[aggregate] {len(runs)} runs -> {results_root}")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(stats)
        if not tests.is_empty():
            print(tests)


if __name__ == "__main__":
    main()
