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
from config import ASSIGNMENT_ROOT, ExperimentConfig

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
        # Rewritten every generation so a crashed/killed run keeps its curve.
        self.write_generations_csv()

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
        extra: dict[str, Any] | None = None,
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
                k: (list(v) if isinstance(v, tuple) else _portable_path(v) if isinstance(v, Path) else v)
                for k, v in asdict(config).items()
            },
            "resolved_params": resolved_params,
            "hardware": _hardware_info(),
            "software": _software_versions(),
            "git_commit": _git_commit_hash(),
            "git_uncommitted_changes": _git_has_uncommitted_changes(),
            **(extra or {}),
        }
        path.write_text(json.dumps(manifest, indent=2, default=str))


def _portable_path(path: Path) -> str:
    """Path relative to the assignment folder, so manifests don't record the
    machine-specific absolute path (which includes the user's home directory).
    """
    try:
        return str(path.resolve().relative_to(ASSIGNMENT_ROOT))
    except ValueError:
        return str(path)


def _hardware_info() -> dict[str, Any]:
    """Hardware specs relevant to interpreting wall-clock timing.

    Wall-clock duration is only meaningful relative to the machine that
    produced it, so this records CPU model, core counts, and total RAM in
    addition to OS/platform -- not just `platform.processor()`, which on
    Apple Silicon only ever returns the generic string "arm".
    """
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": _cpu_count(),
        "python_implementation": platform.python_implementation(),
    }
    info.update(_cpu_and_memory_info())
    return info


def _cpu_count() -> int | None:
    import os

    return os.cpu_count()


def _cpu_and_memory_info() -> dict[str, Any]:
    """CPU model/core-type breakdown and total RAM.

    No identifying/sensitive fields (serial number, hardware UUID) are
    collected -- only what's needed to judge compute cost.
    """
    system = platform.system()
    if system == "Darwin":
        return _macos_hardware_info()
    if system == "Linux":
        return _linux_hardware_info()
    return {"cpu_model": platform.processor() or "unknown", "total_ram_gb": None}


def _macos_hardware_info() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-json"],
            stderr=subprocess.DEVNULL,
        )
        hw = json.loads(raw)["SPHardwareDataType"][0]
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError, IndexError):
        return {"cpu_model": "unknown", "total_ram_gb": None}

    # "number_processors" looks like "proc 10:8:2:0" -> total:performance:efficiency.
    core_breakdown = hw.get("number_processors", "")
    parts = core_breakdown.replace("proc ", "").split(":")
    cores = {}
    if len(parts) >= 3:
        cores = {
            "total_cores": int(parts[0]),
            "performance_cores": int(parts[1]),
            "efficiency_cores": int(parts[2]),
        }

    return {
        "cpu_model": hw.get("chip_type"),
        "machine_model": hw.get("machine_model"),
        "machine_name": hw.get("machine_name"),
        "total_ram": hw.get("physical_memory"),
        **cores,
    }


def _linux_hardware_info() -> dict[str, Any]:
    cpu_model = "unknown"
    try:
        with Path("/proc/cpuinfo").open() as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    total_ram_gb = None
    try:
        with Path("/proc/meminfo").open() as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    total_ram_gb = round(kb / (1024**2), 1)
                    break
    except OSError:
        pass

    return {"cpu_model": cpu_model, "total_ram_gb": total_ram_gb}


def _software_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
    }


def _git_has_uncommitted_changes() -> bool | None:
    """Whether this assignment's code differed from `git_commit` when run."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".."],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(status.strip())


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
