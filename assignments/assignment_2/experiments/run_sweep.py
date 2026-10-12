#!/usr/bin/env python3
"""Run every condition x seed of one experiment, in parallel, with one command.

Example (the 2-seed round of the IPOP experiment)
-------
    python experiments/run_sweep.py --experiment ipop_l10_b8000 --seeds 0 1

All three conditions get the same body, budget, initial population size and
seeds: CMA-ES (no restarts), IPOP-CMA-ES, and random search (batch size =
initial lambda, for logging parity only). Runs are written to
`__data__/<experiment>/<body>__<algorithm>/seed_<n>/`, with each run's stdout
in `__data__/<experiment>/logs/`. A run whose `manifest.json` already exists
is skipped, so the 5-seed round is the same command with `--seeds 0 1 2 3 4`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXPERIMENTS_DIR.parent
CONDITIONS = ["cma_es", "ipop_cma_es", "random_search"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Output subfolder name.")
    parser.add_argument(
        "--body", choices=["turtle", "iguana"], nargs="+", default=["turtle"],
        help="One or more bodies; every body gets every condition x seed.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument(
        "--sim-duration", type=float, default=30.0,
        help="Seconds of simulated time per evaluation (15 for the ipop_l10_* runs).",
    )
    parser.add_argument(
        "--lambda0", type=int, default=None,
        help="Initial population size. Default: CMA-ES formula 4+floor(3*ln(n)) "
        "(19 for the turtle), resolved by each run script.",
    )
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument(
        "--jobs", type=int, default=os.cpu_count(),
        help="Runs executed in parallel (default: all logical CPUs).",
    )
    return parser.parse_args()


def command(
    body: str, algorithm: str, seed: int, args: argparse.Namespace, out: Path
) -> list[str]:
    common = [
        "--body", body, "--seed", str(seed),
        "--budget", str(args.budget), "--sim-duration", str(args.sim_duration),
        "--output-root", str(out),
    ]
    if args.lambda0 is None:
        extra = []
    elif algorithm == "random_search":
        extra = ["--batch-size", str(args.lambda0)]
    else:
        extra = ["--lambda_", str(args.lambda0)]
    script = EXPERIMENTS_DIR / f"run_{algorithm}.py"
    return [sys.executable, str(script), *common, *extra]


def main() -> None:
    args = parse_args()
    out = ASSIGNMENT_ROOT / "__data__" / args.experiment
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for body in args.body:
        for seed in args.seeds:
            for algorithm in args.conditions:
                run_dir = out / f"{body}__{algorithm}" / f"seed_{seed}"
                if (run_dir / "manifest.json").exists():
                    print(f"[sweep] skip {body} {algorithm} seed {seed} (already finished)")
                    continue
                jobs.append((body, algorithm, seed))

    # One run = one sequential process; parallelism comes from running many
    # runs at once. Pin numpy's BLAS to one thread per process so the tiny
    # controller mat-vecs don't spawn thread pools that fight over cores.
    env = {**os.environ, **dict.fromkeys(
        ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"], "1"
    )}

    def run(job: tuple[str, str, int]) -> tuple[str, str, int, int, float]:
        body, algorithm, seed = job
        start = time.monotonic()
        log_path = log_dir / f"{body}__{algorithm}__seed_{seed}.log"
        with log_path.open("w") as log:
            code = subprocess.call(
                command(body, algorithm, seed, args, out),
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
        return body, algorithm, seed, code, time.monotonic() - start

    print(f"[sweep] {len(jobs)} runs, {args.jobs} in parallel -> {out}")
    failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for body, algorithm, seed, code, elapsed in pool.map(run, jobs):
            status = "ok" if code == 0 else f"FAILED (exit {code})"
            failed += code != 0
            print(
                f"[sweep] {body} {algorithm} seed {seed}: {status} in {elapsed / 60:.1f} min",
                flush=True,
            )

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
