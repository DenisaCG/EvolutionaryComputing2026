"""Sanity check for the hand-written CMA-ES: does it converge at all?

Runs it on the analytic sphere function f(x) = sum(x_i^2), which is cheap
(no MuJoCo) and has a well-known optimum (x*=0, f*=0). If this fails, the
implementation has a bug -- no point spending MuJoCo compute finding out.

Run with: uv run pytest assignments/assignment_2/tests/test_cma_es_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cma_es import CMAES, default_lambda  # noqa: E402


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def test_cma_es_converges_on_sphere() -> None:
    n = 10
    cma = CMAES(n=n, sigma0=1.0, mean0=np.full(n, 5.0), seed=0)

    budget = 5000
    while cma.evals_used < budget:
        candidates = cma.ask()
        fitnesses = np.array([sphere(x) for x in candidates])
        cma.tell(fitnesses)
        if cma.best_fitness < 1e-8:
            break

    assert cma.best_fitness < 1e-6, (
        f"CMA-ES did not converge on the sphere function: "
        f"best_fitness={cma.best_fitness} after {cma.evals_used} evals"
    )


def test_default_lambda_matches_paper_examples() -> None:
    # Paper: lambda = 10, 14, 15 for n = 10, 30, 50.
    assert default_lambda(10) == 10
    assert default_lambda(30) == 14
    assert default_lambda(50) == 15
