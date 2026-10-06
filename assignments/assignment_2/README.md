# Assignment 2 — CMA-ES Neuroevolution for Targeted Locomotion

**Research question:** *Does adding a restart strategy with increasing
population size improve CMA-ES performance?* (IPOP-CMA-ES, Auger & Hansen,
CEC 2005 — [paper](https://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf).)

Setup: the **turtle** body in the **Olympic Arena**, a neural-network
controller whose weights are evolved, and three conditions compared at an
equal evaluation budget: CMA-ES (no restarts), IPOP-CMA-ES, and random search.

## What's implemented

- A **hand-written** (mu_W, lambda)-CMA-ES (`src/cma_es.py`) — no imported
  optimizer library, per the assignment's rules. Validated against the
  analytic sphere function in `tests/test_cma_es_sanity.py`.
- **IPOP restarts** (`src/ipop_cma_es.py`, after Auger & Hansen 2005): a
  local stopping criterion triggers a fresh CMA-ES run with doubled `lambda`
  (10, 20, ..., 160; capped at `max_lambda = 200`), under one shared budget.
  Besides the paper's 5 criteria, a task-specific `stagnation` criterion
  (best-so-far improved < 0.01 m over K = 10 + ceil(4.5 n / lambda)
  generations, the form of the paper's equalfunvalhist window) makes restarts reachable
  within an affordable MuJoCo budget. Tested in `tests/test_ipop_cma_es.py`.
- A **random-search baseline** (`src/random_search.py`), matched on total
  evaluations.
- A **controller** (`src/controller.py`): feedforward NN (hinge angles, a
  sin/cos clock at 1.3 Hz, and the bearing to the target as inputs; one
  hidden layer of 6 tanh units; biases), 199 weights for the turtle.
- A **fixed world**: the Olympic Arena with its rugged terrain seeded
  (`terrain_seed = 0`, `src/bodies.py`). Without this, ariel generates new
  random terrain every time the world is built, i.e. every evaluation, which
  made fitness noisy for robots reaching the rugged section.
- A **fitness function** (`src/fitness.py`): `ariel`'s
  `fitness_survival_and_locomotion` — change in planar distance to the
  target, or a flat penalty of 10 if the core drops below 0.05 m (falling
  off the arena).
- Full **reproducibility logging** per run: per-generation CSV (written every
  generation), best genome, an `ariel.ec` database of every individual, and a
  manifest recording hardware, software versions, git commit (+ whether the
  code had uncommitted changes), config, timing, and restarts.
- A **sweep runner** and **plots** for the required convergence comparison
  plus diagnostics (see `experiments/README.md`, `analysis/README.md`).

## Quickstart

```bash
cd assignments/assignment_2

# 1. Tests (cheap, no MuJoCo).
uv run pytest tests/

# 2. All conditions x seeds in parallel (see experiments/README.md).
uv run python experiments/run_sweep.py --experiment ipop_l10_b8000_v2 --seeds 0 1 \
    --budget 8000 --lambda0 10

# 3. Aggregate and plot -> results/ipop_l10_b8000_v2/
uv run python analysis/aggregate_results.py --experiment ipop_l10_b8000_v2
uv run python analysis/make_plots.py --experiment ipop_l10_b8000_v2
```

## Directory layout

```
src/            core library (config, bodies, controller, simulate, fitness,
                cma_es, ipop_cma_es, ec_engine, random_search, logging_utils)
                — see src/README.md
experiments/    CLI runners (run_sweep, run_cma_es, run_ipop_cma_es,
                run_random_search, benchmark) — see experiments/README.md
analysis/       aggregate_results.py, make_plots.py — see analysis/README.md
tests/          CMA-ES and IPOP checks on analytic functions (no MuJoCo)
__data__/       raw per-run data, one subfolder per experiment (gitignored)
results/        per experiment: aggregated CSVs, copied run files, plots
                (results/turtle, results/iguana: earlier, non-comparable runs)
```

## Key design decisions (and why)

- **Built on `ariel.ec`, with a hand-written CMA-ES update.** Each
  generation runs as `ariel.ec` `EAOperation`s (sample -> evaluate ->
  update) inside `ariel.ec.EA`, which also stores every individual in an
  SQLite database. The CMA-ES mean/covariance update itself is our own
  (`src/cma_es.py`), since `ariel.ec`'s GA operators don't fit it. See
  `src/README.md` for the full rationale.
- **Initial population size `lambda0 = 10`** for CMA-ES and IPOP (the
  paper's default formula would give 19 for n = 199). The smaller start
  converges faster, so restarts happen within the budget, and leaves room
  for four doublings below the cap of 200.
- **Fitness is distance reduced plus a fall penalty**
  (`fitness_survival_and_locomotion`). With a fixed spawn and target it ranks
  controllers exactly like the template's plain distance to target. The
  earlier `fitness_direct_path` wasted-path penalty was dropped: path length
  summed per physics step counts the turtle's crawling wobble as waste, so
  most random controllers scored worse than standing still (17/20 in a
  check), and the straight +x route to the target already rewards a direct
  path.
- **Budget of 8000 evaluations per run.** The paper's `n * 10^4` is meant for
  cheap analytic functions; one 15 s episode costs ~0.5 s (~1 s with 6 runs
  in parallel on a laptop), and episodes are now 30 s, which roughly doubles
  that; measure with `experiments/benchmark_eval_time.py`. A restart needs at least K stalled generations
  (100 x 10 = 1000 evaluations at lambda0 = 10, ~1100 at lambda = 20), so
  8000 evaluations leave room for about 1-3 restarts.
- **Same seeds across conditions.** IPOP's first run is identical to the
  CMA-ES run with the same seed, so any difference comes from the restarts.
