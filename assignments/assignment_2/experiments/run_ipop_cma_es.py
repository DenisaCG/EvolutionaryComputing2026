#!/usr/bin/env python3
"""CLI runner: one IPOP-CMA-ES run (one body, one seed, one evaluation budget).

Example
-------
    python experiments/run_ipop_cma_es.py --body turtle --seed 0 --budget 8000 --lambda_ 10

CMA-ES restarts with a doubled population whenever a local stopping
criterion fires (the paper's 5 plus the task-specific `stagnation` one),
until doubling would exceed `--max-lambda`; later restarts keep the largest
population reached. Output goes to
`__data__/<body>__ipop_cma_es/seed_<n>/{generations.csv, best_genome.json, manifest.json, database.db}`;
the manifest also records every restart (`restart_history`).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import ExperimentConfig  # noqa: E402
from ec_engine import build_ea  # noqa: E402
from fitness import evaluate  # noqa: E402
from ipop_cma_es import IPOPCMAES  # noqa: E402
from logging_utils import RunLogger  # noqa: E402
from simulate import genotype_length_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", choices=["turtle", "iguana"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument(
        "--lambda_",
        type=int,
        default=None,
        help="Initial population size. Default: paper's formula 4+floor(3*ln(n)).",
    )
    parser.add_argument("--max-lambda", type=int, default=200)
    parser.add_argument("--stagnation-c", type=float, default=4.5)
    parser.add_argument("--stagnation-tol", type=float, default=0.01)
    parser.add_argument("--sim-duration", type=float, default=30.0)
    parser.add_argument("--clock-hz", type=float, default=1.3)
    parser.add_argument("--sigma0", type=float, default=0.5)
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
        max_lambda=args.max_lambda,
        **({"output_root": args.output_root} if args.output_root else {}),
    )

    n = genotype_length_for(config)
    ipop = IPOPCMAES(
        n=n,
        budget=config.budget,
        lambda_=config.lambda_,
        sigma0=config.sigma0,
        seed=config.seed,
        max_lambda=config.max_lambda,
        stagnation_c=config.stagnation_c,
        stagnation_tol=config.stagnation_tol,
    )
    logger = RunLogger(config.run_dir("ipop_cma_es"))

    print(
        f"[ipop_cma_es] body={config.body} seed={config.seed} n={n} "
        f"lambda0={ipop.lambda_} max_lambda={config.max_lambda} budget={config.budget}"
    )

    history: list = []
    ea = build_ea(
        ipop,
        lambda x: evaluate(config, x),
        config.run_dir("ipop_cma_es") / "database.db",
        history,
    )

    restarts_seen = 0
    while ipop.stopping_reason() is None:
        ea.step()
        record, _ = history[-1]
        logger.log_generation(record)
        print(
            f"  gen {record.generation:4d}  evals {record.evals_used:5d}  "
            f"lambda {ipop.lambda_:3d}  best {record.best_fitness:9.4f}  "
            f"best_so_far {ipop.best_fitness:9.4f}  sigma {record.sigma:8.4f}"
        )
        if len(ipop.restart_history) > restarts_seen:
            restarts_seen = len(ipop.restart_history)
            r = ipop.restart_history[-1]
            print(f"  restart {r.restart} ({r.reason}): lambda {r.old_lambda} -> {r.new_lambda}")

    logger.write_generations_csv()
    logger.write_best_genome(ipop.best_genotype, ipop.best_fitness)
    logger.write_manifest(
        algorithm="ipop_cma_es",
        config=config,
        resolved_params={
            "initial_lambda": ipop.initial_lambda,
            "final_lambda": ipop.lambda_,
            "max_lambda": config.max_lambda,
        },
        termination_reason=ipop.stopping_reason(),
        total_evals=ipop.evals_used,
        extra={"restart_history": [asdict(r) for r in ipop.restart_history]},
    )

    print(
        f"[ipop_cma_es] done: best_fitness={ipop.best_fitness:.4f} "
        f"restarts={len(ipop.restart_history)} final_lambda={ipop.lambda_} "
        f"evals={ipop.evals_used}"
    )


if __name__ == "__main__":
    main()
