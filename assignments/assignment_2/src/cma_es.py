"""Hand-written (mu_W, lambda)-CMA-ES.

Follows the algorithm the Auger & Hansen (CEC 2005) restart paper builds on
(Hansen & Kern 2004's (mu_W, lambda)-CMA-ES, "Eqs 2-8" in the paper -- not
repeated numerically there, so the well-established canonical default
formulas are used here, matching Hansen's own tutorial/reference
implementation). No off-the-shelf CMA-ES library is used or imported.

This module implements ONLY the basic algorithm (a single run, ask/tell,
default stopping criteria). The IPOP restart-with-increasing-population
strategy that decides WHEN and HOW to restart on top of this is deliberately
left out -- that is the next piece of the research question, built on top of
this class without changing it: a restart is just "construct a new CMAES
with lambda_ doubled and re-run".

Positive-only recombination weights (i = 1..mu) are used, matching the
(mu_W, lambda) notation in the 2005 paper (no active/negative weights,
which is a later refinement not described there).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def default_lambda(n: int) -> int:
    """Paper's default population size: lambda = 4 + floor(3 * ln(n))."""
    return 4 + int(3 * math.log(n))


@dataclass
class CMAESParams:
    """Resolved (not per-generation-changing) strategy parameters for a given n, lambda_."""

    n: int
    lambda_: int
    mu: int
    weights: FloatArray
    mu_eff: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c_1: float
    c_mu: float
    chi_n: float


def resolve_params(n: int, lambda_: int) -> CMAESParams:
    """Compute the canonical CMA-ES default strategy parameters for (n, lambda_)."""
    mu = lambda_ // 2
    raw_weights = np.array(
        [math.log(mu + 0.5) - math.log(i) for i in range(1, mu + 1)]
    )
    weights = raw_weights / raw_weights.sum()
    mu_eff = 1.0 / np.sum(weights**2)

    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0.0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    return CMAESParams(
        n=n,
        lambda_=lambda_,
        mu=mu,
        weights=weights,
        mu_eff=mu_eff,
        c_sigma=c_sigma,
        d_sigma=d_sigma,
        c_c=c_c,
        c_1=c_1,
        c_mu=c_mu,
        chi_n=chi_n,
    )


@dataclass
class GenerationRecord:
    """Snapshot of one completed generation, for logging."""

    generation: int
    evals_used: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    sigma: float
    condition_number: float


class CMAES:
    """Basic (mu_W, lambda)-CMA-ES: sample (`ask`), score externally, update (`tell`).

    Parameters
    ----------
    n : int
        Search-space dimension (flat genotype length).
    lambda_ : int or None
        Population size. `None` uses `default_lambda(n)`.
    sigma0 : float
        Initial step size.
    mean0 : array-like or None
        Initial mean. Defaults to the zero vector (no natural [A, B] bounds
        for NN weights, unlike the paper's benchmark functions).
    seed : int or None
        Seeds this instance's own RNG (independent of any ariel.ec RNG).
    stagnation_gens : int or None
        If set, adds a task-specific `stagnation` stopping criterion: fires
        when the best-so-far fitness improved by less than `stagnation_tol`
        over the last `stagnation_gens` generations. `None` (default) keeps
        only the paper's 5 criteria.
    stagnation_tol : float
        Minimum improvement (fitness units; metres for our fitness) that
        counts as progress for the `stagnation` criterion.
    """

    def __init__(
        self,
        n: int,
        lambda_: int | None = None,
        sigma0: float = 0.5,
        mean0: FloatArray | None = None,
        seed: int | None = None,
        stagnation_gens: int | None = None,
        stagnation_tol: float = 0.01,
    ) -> None:
        self.n = n
        resolved_lambda = lambda_ if lambda_ is not None else default_lambda(n)
        self.params = resolve_params(n, resolved_lambda)

        self.sigma = sigma0
        self.sigma0 = sigma0
        self.mean = np.zeros(n) if mean0 is None else np.array(mean0, dtype=float)
        self.C = np.eye(n)
        self.p_sigma = np.zeros(n)
        self.p_c = np.zeros(n)
        self.B = np.eye(n)
        self.D = np.ones(n)

        self.generation = 0
        self.evals_used = 0
        self.tol_x = 1e-12 * sigma0
        self.stagnation_gens = stagnation_gens
        self.stagnation_tol = stagnation_tol

        self.best_genotype: FloatArray | None = None
        self.best_fitness = math.inf
        self._best_fitness_history: list[float] = []

        self._rng = np.random.default_rng(seed)
        self._last_z: FloatArray | None = None
        self._last_y: FloatArray | None = None
        self._last_x: FloatArray | None = None

    @property
    def lambda_(self) -> int:
        return self.params.lambda_

    def ask(self) -> FloatArray:
        """Sample `lambda_` candidate solutions from the current distribution."""
        z = self._rng.standard_normal((self.params.lambda_, self.n))
        y = (z * self.D[np.newaxis, :]) @ self.B.T
        x = self.mean[np.newaxis, :] + self.sigma * y

        self._last_z, self._last_y, self._last_x = z, y, x
        return x

    def tell(self, fitnesses: FloatArray) -> GenerationRecord:
        """Update the distribution from candidate fitnesses (lower is better)."""
        if self._last_x is None:
            msg = "tell() called before ask()."
            raise RuntimeError(msg)

        p = self.params
        order = np.argsort(fitnesses)
        selected = order[: p.mu]

        y_selected = self._last_y[selected]
        z_selected = self._last_z[selected]

        y_w = p.weights @ y_selected
        z_w = p.weights @ z_selected

        mean_new = self.mean + self.sigma * y_w

        p_sigma_new = (1 - p.c_sigma) * self.p_sigma + math.sqrt(
            p.c_sigma * (2 - p.c_sigma) * p.mu_eff
        ) * (self.B @ z_w)

        sigma_new = self.sigma * math.exp(
            (p.c_sigma / p.d_sigma) * (np.linalg.norm(p_sigma_new) / p.chi_n - 1)
        )

        norm_correction = math.sqrt(
            1 - (1 - p.c_sigma) ** (2 * (self.generation + 1))
        )
        h_sigma = float(
            np.linalg.norm(p_sigma_new) / norm_correction
            < (1.4 + 2 / (self.n + 1)) * p.chi_n
        )

        p_c_new = (1 - p.c_c) * self.p_c + h_sigma * math.sqrt(
            p.c_c * (2 - p.c_c) * p.mu_eff
        ) * y_w

        rank_mu = np.einsum("i,ij,ik->jk", p.weights, y_selected, y_selected)
        delta_h = (1 - h_sigma) * p.c_c * (2 - p.c_c)
        C_new = (
            (1 - p.c_1 - p.c_mu) * self.C
            + p.c_1 * (np.outer(p_c_new, p_c_new) + delta_h * self.C)
            + p.c_mu * rank_mu
        )
        C_new = (C_new + C_new.T) / 2

        self.mean = mean_new
        self.C = C_new
        self.p_sigma = p_sigma_new
        self.p_c = p_c_new
        self.sigma = sigma_new

        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.clip(eigvals, 1e-20, None)
        self.D = np.sqrt(eigvals)
        self.B = eigvecs

        best_idx = order[0]
        best_this_gen = float(fitnesses[best_idx])
        if best_this_gen < self.best_fitness:
            self.best_fitness = best_this_gen
            self.best_genotype = self._last_x[best_idx].copy()
        self._best_fitness_history.append(best_this_gen)

        self.generation += 1
        self.evals_used += p.lambda_

        record = GenerationRecord(
            generation=self.generation,
            evals_used=self.evals_used,
            best_fitness=best_this_gen,
            mean_fitness=float(np.mean(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            sigma=self.sigma,
            condition_number=float(eigvals.max() / eigvals.min()),
        )

        self._last_z = self._last_y = self._last_x = None
        return record

    def stopping_reason(self, current_fitnesses: FloatArray) -> str | None:
        """Check the paper's 5 default stopping criteria. `None` if none fired.

        Criteria (Section 2 of the paper, with its published erratum applied
        to `noeffectcoord`: "any coordinate", not "each"):
        equalfunvalhist/Tolfun, TolX, noeffectaxis, noeffectcoord, conditioncov.
        Plus the optional task-specific `stagnation` criterion (see __init__).
        """
        n, sigma, C, D, B = self.n, self.sigma, self.C, self.D, self.B

        window = 10 + math.ceil(30 * n / self.params.lambda_)
        history = self._best_fitness_history
        if len(history) >= window:
            recent = np.array(history[-window:])
            spread = recent.max() - recent.min()
            if spread == 0.0 or (
                spread < 1e-12
                and (current_fitnesses.max() - current_fitnesses.min()) < 1e-12
            ):
                return "equalfunvalhist"

        if np.all(sigma * np.sqrt(np.diag(C)) < self.tol_x) and np.all(
            np.abs(sigma * self.p_c) < self.tol_x
        ):
            return "tolx"

        axis_idx = self.generation % n
        perturbed = self.mean + 0.1 * sigma * D[axis_idx] * B[:, axis_idx]
        if np.allclose(perturbed, self.mean, rtol=0, atol=0):
            return "noeffectaxis"

        coord_perturbed = self.mean + 0.2 * sigma * np.sqrt(np.diag(C))
        if np.any(coord_perturbed == self.mean):
            return "noeffectcoord"

        eigvals = D**2
        if eigvals.max() / eigvals.min() > 1e14:
            return "conditioncov"

        # Not in the paper: equalfunvalhist's 1e-12 tolerance and
        # 10 + 30n/lambda window are sized for cheap benchmark functions and
        # never fire within an affordable MuJoCo budget. Same idea, scaled
        # to the task: best-so-far improved by < stagnation_tol in K gens.
        k = self.stagnation_gens
        if k is not None and len(history) > k:
            best_so_far = np.minimum.accumulate(history)
            if best_so_far[-k - 1] - best_so_far[-1] < self.stagnation_tol:
                return "stagnation"

        return None
