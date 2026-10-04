"""Run an ask/tell strategy as `ariel.ec` EA operations.

One generation of CMA-ES (or IPOP-CMA-ES, or random search) is expressed as
three `EAOperation`s executed by `ariel.ec.EA`:

    sample   -> ask the strategy for lambda candidates, wrap each as an
                `Individual` in the `Population`;
    evaluate -> simulate every unevaluated individual and set its fitness;
    update   -> pass the fitnesses back to the strategy (`tell`), then mark
                the generation dead.

The strategy's update equations stay in `cma_es.py` / `ipop_cma_es.py`;
`ariel.ec` provides the population data model, the generational loop, and
persistence of every individual to an SQLite database. Each individual is
alive for exactly one generation, matching CMA-ES's non-elitist (mu, lambda)
selection: the next generation is sampled from the updated distribution,
not from surviving individuals.

No `from __future__ import annotations` here: `EAOperation` checks that the
first parameter is annotated with the `Population` class itself, which
string annotations would fail.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from ariel.ec import EA, EAOperation, Individual, Population

from cma_es import FloatArray, GenerationRecord

CANDIDATE_TAG = "candidate"


@EAOperation
def sample(population: Population, strategy: Any) -> Population:
    """Append one individual per candidate the strategy proposes."""
    for idx, x in enumerate(strategy.ask()):
        ind = Individual()
        ind.genotype = x.tolist()
        ind.tags = {CANDIDATE_TAG: idx}
        population.append(ind)
    return population


@EAOperation
def evaluate(
    population: Population,
    fitness_fn: Callable[[FloatArray], float],
) -> Population:
    """Simulate every individual that has no fitness yet."""
    for ind in population.unevaluated:
        ind.fitness = fitness_fn(np.asarray(ind.genotype, dtype=float))
    return population


@EAOperation
def update(
    population: Population,
    strategy: Any,
    history: list[tuple[GenerationRecord, FloatArray]],
) -> Population:
    """Tell the strategy this generation's fitnesses, in the order it sampled them."""
    current = sorted(population.alive, key=lambda ind: ind.tags[CANDIDATE_TAG])
    fitnesses = np.array([ind.fitness for ind in current])
    record = strategy.tell(fitnesses)
    history.append((record, fitnesses))
    for ind in current:
        ind.alive = False
    return population


def build_ea(
    strategy: Any,
    fitness_fn: Callable[[FloatArray], float],
    db_file_path: Path,
    history: list[tuple[GenerationRecord, FloatArray]],
) -> EA:
    """`ariel.ec.EA` that advances `strategy` by one generation per `step()`.

    `strategy` is any object with `ask()` / `tell(fitnesses)` (CMAES,
    IPOPCMAES, RandomSearch). After each `step()`, `history[-1]` holds that
    generation's `(GenerationRecord, fitnesses)`. The caller drives `step()`
    itself, since CMA-ES stops on budget/stopping criteria rather than a
    fixed number of generations.
    """
    return EA(
        Population([]),
        [
            sample(strategy=strategy),
            evaluate(fitness_fn=fitness_fn),
            update(strategy=strategy, history=history),
        ],
        is_maximisation=False,
        quiet=True,
        db_file_path=db_file_path,
        db_handling="delete",
    )
