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
- **`controller.py`** — the NN controller's forward pass (input -> tanh
  hidden -> tanh output, rescaled to the hinge range) and the
  flatten/unflatten glue between a flat weight vector (what CMA-ES searches
  over) and the two weight matrices the forward pass needs.
- **`simulate.py`** — `run_episode()`: builds the world+robot, wires a
  MuJoCo control callback that both runs the controller AND accumulates
  per-step path length / minimum core height (needed by the fitness
  function), then calls `simple_runner`. Also `run_episode_trajectory()`,
  a separate post-hoc replay used only for the trajectory plot.
- **`fitness.py`** — combines `ariel.simulation.tasks.targeted_locomotion`'s
  `fitness_survival_and_locomotion` (fall penalty) and `fitness_direct_path`
  (distance + wasted-path penalty) into the single score CMA-ES optimizes.
- **`cma_es.py`** — the hand-written (mu_W, lambda)-CMA-ES: `ask()`/`tell()`
  interface, the paper's default parameter formulas, and its 5 default
  stopping criteria. No restart logic (that's a later addition built on top
  of this class, not a change to it). Also `RandomSearch`'s `GenerationRecord`
  type, shared so both algorithms log identically.
- **`random_search.py`** — the baseline: i.i.d. N(0, sigma0) sampling,
  matched to the same evaluation budget as its paired CMA-ES run.
- **`logging_utils.py`** — `RunLogger`: writes the per-generation CSV, the
  best-genome JSON, and a reproducibility manifest (hardware, software
  versions, git commit, full config, timing, termination reason) for every
  run.

## Design choices worth knowing

- **CMA-ES is not built on `ariel.ec`'s `EA`/`EAOperation` engine.** That
  engine is a generational GA pipeline (parent-selection -> crossover ->
  mutation -> survivor-selection); CMA-ES's ask/tell, mean/covariance-matrix
  update doesn't map onto it, so forcing it through would add complexity
  without benefit. `ariel.ec.set_seed` is likewise unused — it only reseeds
  RNG state private to `ariel.ec`'s own generator/mutator/crossover
  functions, none of which are used here.
- **No boundary handling on NN weights.** The paper's boundary-penalty
  mechanism assumes a bounded `[A, B]^n` domain (its benchmark functions all
  have one); NN weights don't have a natural bound, so this is a documented
  deviation, not an oversight.
- **`sigma0` default (0.5)** matches the assignment template's own
  random-weight-init scale, chosen instead of the paper's
  `(B-A)/2`-from-search-region rule, which doesn't apply here.
