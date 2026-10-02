"""Random-search baseline, matched to the same evaluation budget as CMA-ES.

Per `CLAUDE.md`'s hard constraint: the baseline must match the EA's total
number of fitness evaluations, not its generation count or population size.
Candidates are drawn i.i.d. from the same N(0, sigma0) distribution used to
initialize CMA-ES's search (no adaptation, no memory of past evaluations
beyond tracking the best-so-far) -- batches of `batch_size` evaluations are
only a logging convenience (to plot mean/std per "generation" the same way
CMA-ES's log is plotted), not a generational algorithm.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from cma_es import GenerationRecord

FloatArray = npt.NDArray[np.float64]


class RandomSearch:
    """Samples i.i.d. N(0, sigma0) candidates, `batch_size` at a time.

    Parameters
    ----------
    n : int
        Search-space dimension (flat genotype length).
    batch_size : int
        Number of candidates sampled per `ask()` call (for logging parity
        with CMA-ES's per-generation records; has no algorithmic effect).
    sigma0 : float
        Standard deviation of the sampling distribution.
    seed : int or None
        Seeds this instance's own RNG.
    """

    def __init__(
        self,
        n: int,
        batch_size: int,
        sigma0: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.n = n
        self.batch_size = batch_size
        self.sigma0 = sigma0

        self.generation = 0
        self.evals_used = 0
        self.best_genotype: FloatArray | None = None
        self.best_fitness = math.inf

        self._rng = np.random.default_rng(seed)
        self._last_x: FloatArray | None = None

    def ask(self) -> FloatArray:
        """Draw `batch_size` fresh i.i.d. candidates."""
        self._last_x = self._rng.normal(
            loc=0.0, scale=self.sigma0, size=(self.batch_size, self.n)
        )
        return self._last_x

    def tell(self, fitnesses: FloatArray) -> GenerationRecord:
        """Record batch statistics and update the best-so-far. No adaptation."""
        if self._last_x is None:
            msg = "tell() called before ask()."
            raise RuntimeError(msg)

        best_idx = int(np.argmin(fitnesses))
        best_this_batch = float(fitnesses[best_idx])
        if best_this_batch < self.best_fitness:
            self.best_fitness = best_this_batch
            self.best_genotype = self._last_x[best_idx].copy()

        self.generation += 1
        self.evals_used += self.batch_size

        record = GenerationRecord(
            generation=self.generation,
            evals_used=self.evals_used,
            best_fitness=best_this_batch,
            mean_fitness=float(np.mean(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            sigma=self.sigma0,
            condition_number=1.0,
        )
        self._last_x = None
        return record
