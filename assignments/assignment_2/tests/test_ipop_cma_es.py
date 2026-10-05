"""Checks for the IPOP restart wrapper and the stagnation criterion (no MuJoCo).

Run with: uv run pytest assignments/assignment_2/tests/test_ipop_cma_es.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cma_es import CMAES  # noqa: E402
from ipop_cma_es import IPOPCMAES  # noqa: E402


def rastrigin(x: np.ndarray) -> float:
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def run_ipop(budget: int, **kwargs: object) -> IPOPCMAES:
    ipop = IPOPCMAES(n=10, budget=budget, sigma0=0.5, seed=0, **kwargs)
    while ipop.stopping_reason() is None:
        candidates = ipop.ask()
        ipop.tell(np.array([rastrigin(x) for x in candidates]))
    return ipop


def test_stagnation_fires_only_after_k_flat_generations() -> None:
    cma = CMAES(n=5, lambda_=6, seed=0, stagnation_gens=3, stagnation_tol=0.01)
    flat = np.ones(6)
    for _ in range(3):
        cma.ask()
        cma.tell(flat)
        assert cma.stopping_reason(flat) is None
    cma.ask()
    cma.tell(flat)
    assert cma.stopping_reason(flat) == "stagnation"


def test_stagnation_disabled_by_default() -> None:
    cma = CMAES(n=5, lambda_=6, seed=0)
    flat = np.ones(6)
    for _ in range(10):
        cma.ask()
        cma.tell(flat)
    assert cma.stopping_reason(flat) is None


def test_lambda_doubles_then_is_capped() -> None:
    ipop = run_ipop(20000, lambda_=10, max_lambda=40, stagnation_gens=5)
    lambdas = [r.new_lambda for r in ipop.restart_history]
    assert lambdas[:2] == [20, 40]
    assert len(lambdas) > 2, "expected restarts beyond the cap"
    assert all(lam == 40 for lam in lambdas[2:])
    assert all(r.reason == "stagnation" for r in ipop.restart_history)


def test_budget_never_exceeded() -> None:
    for budget in (95, 1000, 4321):
        ipop = run_ipop(budget, lambda_=10, max_lambda=40, stagnation_gens=5)
        assert ipop.evals_used <= budget
        assert budget - ipop.evals_used < ipop.next_lambda


def test_first_run_matches_plain_cma_es() -> None:
    """Same seed -> IPOP's first run is identical to CMA-ES until it restarts."""
    ipop = IPOPCMAES(n=10, budget=10**6, lambda_=10, seed=3, stagnation_gens=25)
    cma = CMAES(n=10, lambda_=10, seed=3, stagnation_gens=25)
    for _ in range(20):
        x_ipop, x_cma = ipop.ask(), cma.ask()
        np.testing.assert_array_equal(x_ipop, x_cma)
        f = np.array([rastrigin(x) for x in x_cma])
        ipop.tell(f)
        cma.tell(f)
