"""Load every run's generations.csv + manifest.json into two tidy tables.

Scans `__data__/<body>__<algorithm>/seed_<n>/` and produces:
  - a long-format per-generation table (one row per generation per run), for
    convergence plots;
  - a per-run summary table (one row per body/algorithm/seed), for final-
    fitness distributions and timing tables.

Both are written to `__data__/aggregated_generations.csv` and
`__data__/aggregated_summary.csv` so `make_plots.py` (or the report) can read
them without re-parsing every run directory.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import ASSIGNMENT_ROOT  # noqa: E402

DATA_ROOT = ASSIGNMENT_ROOT / "__data__"
CONDITION_RE = re.compile(r"^(?P<body>[a-z]+)__(?P<algorithm>[a-z_]+)$")
SEED_RE = re.compile(r"^seed_(?P<seed>\d+)$")


def find_run_dirs() -> list[Path]:
    """Every `<body>__<algorithm>/seed_<n>/` directory containing a manifest."""
    if not DATA_ROOT.exists():
        return []
    return sorted(DATA_ROOT.glob("*/seed_*/manifest.json"))


def load_generations() -> pl.DataFrame:
    frames = []
    for manifest_path in find_run_dirs():
        run_dir = manifest_path.parent
        condition_match = CONDITION_RE.match(run_dir.parent.name)
        seed_match = SEED_RE.match(run_dir.name)
        if not condition_match or not seed_match:
            continue

        csv_path = run_dir / "generations.csv"
        if not csv_path.exists():
            continue

        df = pl.read_csv(csv_path)
        df = df.with_columns(
            pl.lit(condition_match["body"]).alias("body"),
            pl.lit(condition_match["algorithm"]).alias("algorithm"),
            pl.lit(int(seed_match["seed"])).alias("seed"),
        )
        frames.append(df)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical")


def load_summary() -> pl.DataFrame:
    rows = []
    for manifest_path in find_run_dirs():
        run_dir = manifest_path.parent
        condition_match = CONDITION_RE.match(run_dir.parent.name)
        seed_match = SEED_RE.match(run_dir.name)
        if not condition_match or not seed_match:
            continue

        manifest = json.loads(manifest_path.read_text())
        best_genome_path = run_dir / "best_genome.json"
        best_fitness = None
        if best_genome_path.exists():
            best_fitness = json.loads(best_genome_path.read_text())["fitness"]

        rows.append({
            "body": condition_match["body"],
            "algorithm": condition_match["algorithm"],
            "seed": int(seed_match["seed"]),
            "best_fitness": best_fitness,
            "total_evals": manifest["total_evals"],
            "wall_clock_duration_s": manifest["wall_clock_duration_s"],
            "termination_reason": manifest["termination_reason"],
        })

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def main() -> None:
    generations = load_generations()
    summary = load_summary()

    if generations.is_empty() or summary.is_empty():
        print(f"[aggregate] no runs found under {DATA_ROOT}. Run experiments first.")
        return

    generations.write_csv(DATA_ROOT / "aggregated_generations.csv")
    summary.write_csv(DATA_ROOT / "aggregated_summary.csv")
    print(
        f"[aggregate] wrote {len(generations)} generation-rows and "
        f"{len(summary)} run-rows to {DATA_ROOT}"
    )


if __name__ == "__main__":
    main()
