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
    parser.add_argument("--sim-duration", type=float, default=10.0)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        body=args.body,
        seed=args.seed,
        hidden_size=args.hidden_size,
        sim_duration=args.sim_duration,
        lambda_=args.lambda_,
        budget=args.budget,
        sigma0=args.sigma0,
    )

    n = genotype_length_for(config)
    cma = CMAES(n=n, lambda_=config.lambda_, sigma0=config.sigma0, seed=config.seed)
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

    termination_reason = "budget_exhausted"
    while cma.evals_used < config.budget:
        ea.step()
        record, fitnesses = history[-1]
        logger.log_generation(record)
        print(
            f"  gen {record.generation:4d}  evals {record.evals_used:5d}  "
            f"best {record.best_fitness:9.4f}  sigma {record.sigma:8.4f}"
        )

        reason = cma.stopping_reason(fitnesses)
        if reason is not None:
            termination_reason = reason
            break

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
        termination_reason=termination_reason,
        total_evals=cma.evals_used,
    )

    print(
        f"[cma_es] done: best_fitness={cma.best_fitness:.4f} "
        f"reason={termination_reason} evals={cma.evals_used}"
    )


if __name__ == "__main__":
    main()
