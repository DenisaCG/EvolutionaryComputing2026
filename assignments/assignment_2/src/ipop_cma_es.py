"""IPOP restarts around the assignment's handwritten CMAES (minimization).

Reference: Auger & Hansen (2005), A Restart CMA Evolution Strategy With
Increasing Population Size, CEC, pp. 1769-1776, Sections 2-3.
https://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf

Population doubles on a local stopping criterion; each restart resets the
entire strategy. For these unbounded NN weights, the first run uses mean0
(zero by default), and later means are sampled from N(mean0, sigma0**2 I).
This initialization is a task-specific adaptation, not the paper's bounded
uniform initialization. The initial step size is restored on every restart.
No simulation, file writing, or experiment is performed by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from cma_es import CMAES, FloatArray, GenerationRecord, default_lambda


@dataclass(frozen=True)
class RestartRecord:
    """A restart actually launched by ask(), suitable for manifest metadata."""

    restart: int
    generation: int
    evals_used: int
    reason: str
    old_lambda: int
    new_lambda: int


class IPOPCMAES:
    """Budget-aware ask/tell wrapper; fitness evaluation remains external.

    Use ``while optimizer.stopping_reason() is None:`` around ask/evaluate/
    tell. tell() checks the existing CMAES stopping criteria automatically.
    A local stop schedules a restart for the next ask(), rather than ending
    the whole search. Global termination occurs when the remaining budget
    cannot accommodate the next full population (including a doubled one).
    There are no partial generations and no budget overshoot.

    ``generation`` and ``evals_used`` count across all restarts. tell() returns
    the existing GenerationRecord schema with these cumulative counters,
    but fitness/sigma/condition statistics refer to the completed generation.
    ``best_fitness`` and ``best_genotype`` retain the best across the whole
    search. ``lambda_`` is the current population; ``next_lambda`` includes a
    pending doubling. ``restart_history`` records only launched restarts.

    The initial run uses exactly the supplied seed, mean and sigma, allowing
    comparison with basic CMAES before the first restart. Subsequent runs
    use distinct SeedSequence child streams. No fixed restart interval or
    additional stagnation heuristic is imposed; small budgets may produce
    zero restarts. No target-fitness threshold is assumed for locomotion.
    """

    def __init__(
        self,
        n: int,
        budget: int,
        lambda_: int | None = None,
        sigma0: float = 0.5,
        mean0: FloatArray | None = None,
        seed: int | None = None,
    ) -> None:
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError("n must be a positive integer.")
        if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 0:
            raise ValueError("budget must be a nonnegative integer.")
        initial_lambda = default_lambda(n) if lambda_ is None else lambda_
        if (
            isinstance(initial_lambda, bool)
            or not isinstance(initial_lambda, Integral)
            or initial_lambda < 2
        ):
            raise ValueError("lambda_ must be an integer >= 2.")
        if not math.isfinite(sigma0) or sigma0 <= 0:
            raise ValueError("sigma0 must be finite and positive.")
        center = np.zeros(n) if mean0 is None else np.array(mean0, dtype=float)
        if center.shape != (n,) or not np.all(np.isfinite(center)):
            raise ValueError("mean0 must contain n finite values.")

        self.n = int(n)
        self.budget = int(budget)
        self.sigma0 = float(sigma0)
        self.initial_lambda = int(initial_lambda)
        self._center = center.copy()
        self._restart_seeds = np.random.SeedSequence(seed)
        self._cma = CMAES(
            n=self.n, lambda_=self.initial_lambda, sigma0=self.sigma0,
            mean0=self._center, seed=seed,
        )
        self.generation = 0
        self.evals_used = 0
        self.best_fitness = math.inf
        self.best_genotype: FloatArray | None = None
        self.restart_history: list[RestartRecord] = []
        self._pending_restart: str | None = None
        self._awaiting_fitness = False

    @property
    def lambda_(self) -> int:
        return self._cma.lambda_

    @property
    def next_lambda(self) -> int:
        return self.lambda_ * (2 if self._pending_restart is not None else 1)

    def stopping_reason(self) -> str | None:
        """Global budget stop, not the local criterion that triggers a restart.

        A pending ask() must be scored before this reports global termination.
        A zero/too-small initial budget therefore terminates without a genome.
        """
        if self._awaiting_fitness:
            return None
        remaining = self.budget - self.evals_used
        if remaining == 0:
            return "budget_exhausted"
        if remaining < self.next_lambda:
            return "insufficient_budget_for_population"
        return None

    def ask(self) -> FloatArray:
        """Launch a pending restart, then sample one complete population."""
        if self._awaiting_fitness:
            raise RuntimeError("tell() must follow ask() before asking again.")
        reason = self.stopping_reason()
        if reason is not None:
            raise RuntimeError(f"IPOP-CMA-ES has stopped: {reason}.")

        if self._pending_restart is not None:
            old_lambda = self.lambda_
            mean_seed, search_seed = self._restart_seeds.spawn(2)
            rng = np.random.default_rng(mean_seed)
            mean = rng.normal(loc=self._center, scale=self.sigma0, size=self.n)
            # New construction resets covariance, paths, sigma, local history
            # and counters, and recalculates parameters for the larger lambda.
            self._cma = CMAES(
                n=self.n, lambda_=2 * old_lambda, sigma0=self.sigma0,
                mean0=mean, seed=int(search_seed.generate_state(1)[0]),
            )
            self.restart_history.append(RestartRecord(
                restart=len(self.restart_history) + 1,
                generation=self.generation,
                evals_used=self.evals_used,
                reason=self._pending_restart,
                old_lambda=old_lambda,
                new_lambda=self.lambda_,
            ))
            self._pending_restart = None

        candidates = self._cma.ask()
        self._awaiting_fitness = True
        return candidates

    def tell(self, fitnesses: FloatArray) -> GenerationRecord:
        """Update CMA-ES, retain the global best, and check for a local stop."""
        if not self._awaiting_fitness:
            raise RuntimeError("tell() called before ask().")
        values = np.asarray(fitnesses, dtype=float)
        if values.shape != (self.lambda_,) or not np.all(np.isfinite(values)):
            raise ValueError("Provide one finite fitness per sampled candidate.")

        record = self._cma.tell(values)
        self._awaiting_fitness = False
        self.generation += 1
        self.evals_used += self.lambda_
        if self._cma.best_fitness < self.best_fitness:
            self.best_fitness = self._cma.best_fitness
            self.best_genotype = self._cma.best_genotype.copy()
        self._pending_restart = self._cma.stopping_reason(values)
        return replace(record, generation=self.generation, evals_used=self.evals_used)
