#!/usr/bin/env python3
"""CLI runner: random-search baseline, matched to a CMA-ES run's evaluation budget.

Example
-------
    uv run python experiments/run_random_search.py --body turtle --seed 0 --budget 1000 --batch-size 17

`--batch-size` should equal the CMA-ES run's resolved `lambda_` for that body
(printed by `run_cma_es.py`, or read from its manifest) purely so the two
runs' per-"generation" logs line up for plotting -- it has no effect on the
random-search algorithm itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cma_es import default_lambda  # noqa: E402
from config import ExperimentConfig  # noqa: E402
from ec_engine import build_ea  # noqa: E402
from fitness import evaluate  # noqa: E402
from logging_utils import RunLogger  # noqa: E402
from random_search import RandomSearch  # noqa: E402
from simulate import genotype_length_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", choices=["turtle", "iguana"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Match this to the paired CMA-ES run's resolved lambda_. "
        "Default: CMA-ES formula 4+floor(3*ln(n)).",
    )
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
        budget=args.budget,
        sigma0=args.sigma0,
        **({"output_root": args.output_root} if args.output_root else {}),
    )

    n = genotype_length_for(config)
    batch_size = args.batch_size if args.batch_size is not None else default_lambda(n)
    search = RandomSearch(
        n=n, batch_size=batch_size, sigma0=config.sigma0, seed=config.seed
    )
    logger = RunLogger(config.run_dir("random_search"))

    print(
        f"[random_search] body={config.body} seed={config.seed} n={n} "
        f"batch_size={batch_size} budget={config.budget}"
    )

    history: list = []
    ea = build_ea(
        search,
        lambda x: evaluate(config, x),
        config.run_dir("random_search") / "database.db",
        history,
    )

    while search.evals_used + search.batch_size <= config.budget:
        ea.step()
        record, _ = history[-1]
        logger.log_generation(record)
        print(
            f"  batch {record.generation:4d}  evals {record.evals_used:5d}  "
            f"best {record.best_fitness:9.4f}"
        )

    logger.write_generations_csv()
    logger.write_best_genome(search.best_genotype, search.best_fitness)
    logger.write_manifest(
        algorithm="random_search",
        config=config,
        resolved_params={"batch_size": batch_size},
        termination_reason="budget_exhausted",
        total_evals=search.evals_used,
    )

    print(
        f"[random_search] done: best_fitness={search.best_fitness:.4f} "
        f"evals={search.evals_used}"
    )


if __name__ == "__main__":
    main()
