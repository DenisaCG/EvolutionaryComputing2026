# `src/` — core library

Everything here is imported by the scripts in `experiments/` and
`analysis/`; nothing in this directory is run directly.

- **`config.py`** — `ExperimentConfig`: the one dataclass that pins down
  body, world, controller size, simulation length, target, and
  algorithm-specific knobs (population size, evaluation budget, seed) for a
  single run. Every script builds one of these from CLI args and threads it
  through everything else.
- **`bodies.py`** — maps a body name (`"turtle"` / `"iguana"`) to the ariel
  prebuilt-robot factory that constructs it, and builds the (fixed) Olympic
  Arena world. Add a new body here only.
- **`controller.py`** — the NN controller: its inputs (hinge angles, a
  sin/cos clock at `clock_hz`, and the bearing to the target relative to the
  core's heading), the forward pass (input -> tanh hidden -> tanh output,
  biases on both layers, rescaled to the hinge range) and the
  flatten/unflatten glue between a flat weight vector (what CMA-ES searches
  over) and the two weight matrices the forward pass needs.
- **`simulate.py`** — `run_episode()`: builds the world+robot, wires a
  MuJoCo control callback that both runs the controller AND accumulates
  per-step path length / minimum core height (needed by the fitness
  function), then calls `simple_runner`. Also `run_episode_trajectory()`,
  a separate post-hoc replay used only for the trajectory plot.
- **`fitness.py`** — `ariel.simulation.tasks.targeted_locomotion`'s
  `fitness_survival_and_locomotion`: change in planar distance to the target,
  or a flat penalty of 10 if the core dropped below 0.05 m. Path length is
  still tracked by `simulate.py` but no longer used in the score.
- **`cma_es.py`** — the hand-written (mu_W, lambda)-CMA-ES: `ask()`/`tell()`
  interface, the paper's default parameter formulas, and its 5 default
  stopping criteria. Restart logic lives separately in `ipop_cma_es.py`.
  Also `RandomSearch`'s `GenerationRecord`
  type, shared so both algorithms log identically.
- **`ec_engine.py`** — `build_ea()`: wraps any ask/tell strategy (CMA-ES,
  IPOP-CMA-ES, random search) as `ariel.ec` `EAOperation`s run by
  `ariel.ec.EA`, which also persists every individual to `database.db`.
- **`random_search.py`** — the baseline: i.i.d. N(0, sigma0) sampling,
  matched to the same evaluation budget as its paired CMA-ES run.
- **`logging_utils.py`** — `RunLogger`: writes the per-generation CSV, the
  best-genome JSON, and a reproducibility manifest (hardware, software
  versions, git commit, full config, timing, termination reason) for every
  run.

## Design choices worth knowing

**IPOP-CMA-ES** (`ipop_cma_es.py`: `IPOPCMAES`, `RestartRecord`) composes
the existing `CMAES` without altering its update equations. Source: Auger &
Hansen, *A Restart CMA Evolution Strategy With Increasing Population Size*,
CEC 2005, pp. 1769–1776, Sections 2–3
([paper](https://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf)).
Used by `experiments/run_ipop_cma_es.py`; tested in `tests/test_ipop_cma_es.py`.

- Population sizes follow `lambda0, 2*lambda0, 4*lambda0, ...` until doubling
  would exceed `max_lambda`; later restarts keep the largest `lambda`
  reached (not in the paper, which has no cap).
- After `tell()`, a local stopping criterion schedules a restart: the
  paper's 5, plus the task-specific `stagnation` criterion in `CMAES`
  (best-so-far improved by less than `stagnation_tol` over
  K = 10 + ceil(`stagnation_c` * n / lambda) generations -- the paper's
  equalfunvalhist window form with constant 4.5 instead of 30; not in the
  paper). The paper's criteria are
  sized for cheap benchmark functions (e.g. `equalfunvalhist` needs a 1e-12
  spread over 10 + 30n/lambda generations, ~6000 evaluations here) and never
  fire within an affordable MuJoCo budget. The next `ask()` creates a fresh
  optimizer: identity covariance, zero evolution paths, reset local history,
  restored `sigma0`, and strategy parameters recalculated for the new
  population.
- Initialization is adapted to unbounded NN weights: the first run starts at
  `mean0` (zero by default), matching basic CMA-ES with the same seed
  exactly; subsequent means are fresh draws from `N(mean0, sigma0**2 I)`.
  The paper instead draws means uniformly in a bounded search region.
  Restart random streams are reproducible from the supplied seed. Restarts
  do not initialize at the previous winner.
- One budget covers all runs. Only complete populations are evaluated, so
  the budget is never exceeded; a remainder smaller than the next population
  stays unused.
- `best_genotype`/`best_fitness` retain the overall winner. `tell()` returns
  the existing `GenerationRecord`, with cumulative generations/evaluations
  but generation-local statistics. `restart_history` records each launched
  restart's cause, population sizes, and cumulative counters.

- **Runs are driven by `ariel.ec`'s `EA` engine; the strategy update is our
  own.** `ec_engine.py` expresses one generation as three `EAOperation`s
  (`sample` -> `evaluate` -> `update`), so `ariel.ec` provides the
  `Individual`/`Population` data model, the generational loop, and an SQLite
  database of every evaluated individual (`database.db` per run). The
  mean/sigma/covariance update stays hand-written in `cma_es.py`, since
  `ariel.ec`'s selection/crossover/mutation operators don't map onto
  CMA-ES's distribution update. Each individual lives one generation,
  matching non-elitist (mu, lambda) selection. `ariel.ec.set_seed` is unused
  — it only reseeds RNG state private to `ariel.ec`'s own
  generator/mutator/crossover functions, none of which are used here.
- **No boundary handling on NN weights.** The paper's boundary-penalty
  mechanism assumes a bounded `[A, B]^n` domain (its benchmark functions all
  have one); NN weights don't have a natural bound, so this is a documented
  deviation, not an oversight.
- **`sigma0` default (0.5)** matches the assignment template's own
  random-weight-init scale, chosen instead of the paper's
  `(B-A)/2`-from-search-region rule, which doesn't apply here.
