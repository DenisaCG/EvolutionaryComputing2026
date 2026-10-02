"""Per-run logging: generation CSV, best-genome JSON, and a reproducibility manifest.

The manifest exists so a run can be understood (and, if needed, argued about
in the report) without re-running it: exact config, hardware, software
versions, timing, and why it stopped.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from cma_es import GenerationRecord
from config import ExperimentConfig

GENERATIONS_CSV_FIELDS = [
    "generation",
    "evals_used",
    "best_fitness",
    "mean_fitness",
    "std_fitness",
    "sigma",
    "condition_number",
    "wall_time_s",
]


class RunLogger:
    """Accumulates per-generation records and writes them plus a manifest at the end."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._start_time = time.monotonic()
        self._wall_clock_start = time.time()

    def log_generation(self, record: GenerationRecord) -> None:
        row = asdict(record)
        row["wall_time_s"] = time.monotonic() - self._start_time
        self._records.append(row)

    def write_generations_csv(self) -> None:
        path = self.run_dir / "generations.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GENERATIONS_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self._records)

    def write_best_genome(self, genotype: np.ndarray, fitness: float) -> None:
        path = self.run_dir / "best_genome.json"
        path.write_text(
            json.dumps(
                {"fitness": fitness, "weights": genotype.tolist()},
                indent=2,
            )
        )

    def write_manifest(
        self,
        *,
        algorithm: str,
        config: ExperimentConfig,
        resolved_params: dict[str, Any],
        termination_reason: str,
        total_evals: int,
    ) -> None:
        path = self.run_dir / "manifest.json"
        manifest = {
            "algorithm": algorithm,
            "timestamp_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._wall_clock_start)
            ),
            "wall_clock_duration_s": time.monotonic() - self._start_time,
            "termination_reason": termination_reason,
            "total_evals": total_evals,
            "config": {
                k: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                for k, v in asdict(config).items()
            },
            "resolved_params": resolved_params,
            "hardware": _hardware_info(),
            "software": _software_versions(),
            "git_commit": _git_commit_hash(),
        }
        path.write_text(json.dumps(manifest, indent=2, default=str))


def _hardware_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "cpu_count": _cpu_count(),
        "python_implementation": platform.python_implementation(),
    }


def _cpu_count() -> int | None:
    import os

    return os.cpu_count()


def _software_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
    }


def _git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parent,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
