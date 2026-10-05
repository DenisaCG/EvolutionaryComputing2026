#!/usr/bin/env python3
"""Timing pilot: measure real wall-clock seconds per fitness evaluation.

Run this FIRST, on the machine that will run the full sweep, before choosing
--budget for run_cma_es.py / run_random_search.py. Physics simulation time
scales with --sim-duration and body complexity, not with CMA-ES itself, so a
handful of random-weight episodes gives a good estimate of a full run's cost.

Example
-------
    uv run python experiments/benchmark_eval_time.py --body turtle --n-samples 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import ExperimentConfig  # noqa: E402
from fitness import evaluate  # noqa: E402
from simulate import genotype_length_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", choices=["turtle", "iguana"], required=True)
    parser.add_argument("--sim-duration", type=float, default=15.0)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument(
        "--target-minutes",
        type=float,
        default=15.0,
        help="Wall-clock ceiling to size a recommended budget for (default: 15 min).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(body=args.body, seed=0, sim_duration=args.sim_duration)
    n = genotype_length_for(config)
    rng = np.random.default_rng(0)

    print(f"[benchmark] body={args.body} n_weights={n} sim_duration={args.sim_duration}s")
    times = []
    for i in range(args.n_samples):
        weights = rng.normal(scale=config.sigma0, size=n)
        start = time.monotonic()
        evaluate(config, weights)
        elapsed = time.monotonic() - start
        times.append(elapsed)
        print(f"  eval {i + 1}/{args.n_samples}: {elapsed:.3f}s")

    times = np.array(times)
    mean_s, std_s, median_s = times.mean(), times.std(), np.median(times)
    print(
        f"\n[benchmark] mean={mean_s:.3f}s  median={median_s:.3f}s  "
        f"std={std_s:.3f}s  (n={args.n_samples})"
    )

    budget_for_ceiling = int((args.target_minutes * 60) / mean_s)
    print(
        f"[benchmark] at this rate, a {args.target_minutes:.0f}-minute run affords "
        f"~{budget_for_ceiling} fitness evaluations (--budget {budget_for_ceiling})."
    )


if __name__ == "__main__":
    main()
