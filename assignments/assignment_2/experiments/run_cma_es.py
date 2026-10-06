#!/usr/bin/env python3
"""CLI runner: one CMA-ES run (one body, one seed, one evaluation budget).

Example
-------
    uv run python experiments/run_cma_es.py --body turtle --seed 0 --budget 1000

Run once per seed (repeat with --seed 0..4 for the assignment's required
>=5 independent runs). Output goes to
`__data__/<body>__cma_es/seed_<n>/{generations.csv, best_genome.json, manifest.json, database.db}`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cma_es import CMAES  # noqa: E402
from config import ExperimentConfig  # noqa: E402
from ec_engine import build_ea  # noqa: E402
from fitness import evaluate  # noqa: E402
from logging_utils import RunLogger  # noqa: E402
from simulate import genotype_length_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", choices=["turtle", "iguana"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--budget",
        type=int,
        default=1500,
        help="Total fitness evaluations for this run (default: 1500; see "
        "benchmark_eval_time.py to size this for your hardware).",
    )
    parser.add_argument(
        "--lambda_",
        type=int,
        default=None,
        help="CMA-ES population size. Default: paper's formula 4+floor(3*ln(n)).",
    )
    parser.add_argument("--sim-duration", type=float, default=30.0)
    parser.add_argument("--clock-hz", type=float, default=1.3)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--stagnation-c", type=float, default=4.5)
    parser.add_argument("--stagnation-tol", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=6)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where per-run data is written (default: assignment_2/__data__).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        body=args.body,
        seed=args.seed,
        hidden_size=args.hidden_size,
        sim_duration=args.sim_duration,
        clock_hz=args.clock_hz,
        lambda_=args.lambda_,
        budget=args.budget,
        sigma0=args.sigma0,
        stagnation_c=args.stagnation_c,
        stagnation_tol=args.stagnation_tol,
        **({"output_root": args.output_root} if args.output_root else {}),
    )

    n = genotype_length_for(config)
    cma = CMAES(
        n=n,
        lambda_=config.lambda_,
        sigma0=config.sigma0,
        seed=config.seed,
        stagnation_c=config.stagnation_c,
        stagnation_tol=config.stagnation_tol,
    )
    logger = RunLogger(config.run_dir("cma_es"))

    print(
        f"[cma_es] body={config.body} seed={config.seed} n={n} "
        f"lambda={cma.lambda_} mu={cma.params.mu} budget={config.budget}"
    )

    history: list = []
    ea = build_ea(
        cma,
        lambda x: evaluate(config, x),
        config.run_dir("cma_es") / "database.db",
        history,
    )

    # Plain CMA-ES (no restarts) uses the whole budget so it is compared with
    # IPOP and random search at equal evaluations; local stopping criteria
    # are only logged (first firing) -- that is where IPOP would restart.
    # Only full generations are run, so the budget is never exceeded.
    first_local_stop: dict | None = None
    while cma.evals_used + cma.lambda_ <= config.budget:
        ea.step()
        record, fitnesses = history[-1]
        logger.log_generation(record)
        print(
            f"  gen {record.generation:4d}  evals {record.evals_used:5d}  "
            f"best {record.best_fitness:9.4f}  sigma {record.sigma:8.4f}"
        )

        reason = cma.stopping_reason(fitnesses)
        if reason is not None and first_local_stop is None:
            first_local_stop = {
                "reason": reason,
                "generation": record.generation,
                "evals_used": record.evals_used,
            }
            print(f"  (local stop '{reason}' -- continuing, no restarts)")

    logger.write_generations_csv()
    logger.write_best_genome(cma.best_genotype, cma.best_fitness)

    resolved_params = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in asdict(cma.params).items()
    }
    logger.write_manifest(
        algorithm="cma_es",
        config=config,
        resolved_params=resolved_params,
        termination_reason="budget_exhausted",
        total_evals=cma.evals_used,
        extra={"first_local_stop": first_local_stop},
    )

    print(
        f"[cma_es] done: best_fitness={cma.best_fitness:.4f} "
        f"first_local_stop={first_local_stop} evals={cma.evals_used}"
    )


if __name__ == "__main__":
    main()
