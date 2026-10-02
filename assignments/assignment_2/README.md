# Assignment 2 — CMA-ES Neuroevolution for Targeted Locomotion

**Research question:** *Does adding a restart strategy with increasing
population size improve CMA-ES performance?* (IPOP-CMA-ES, Auger & Hansen,
CEC 2005 — see `cec2005ipopcmaes.pdf`.)

This codebase currently implements the **basic (mu_W, lambda)-CMA-ES**
(no restart yet) plus a full, reusable experiment pipeline — config, logging,
plotting — that the restart variant plugs into without restructuring
anything. Two bodies (`turtle`, `iguana`) are run side by side right now to
assess feasibility and results before the team restricts to one.

## What's implemented

- A **hand-written** (mu_W, lambda)-CMA-ES (`src/cma_es.py`) — no imported
  optimizer library, per the assignment's rules. Validated against the
  analytic sphere function in `tests/test_cma_es_sanity.py` before ever
  touching MuJoCo.
- A **random-search baseline** (`src/random_search.py`), matched to the same
  total evaluation budget as its paired CMA-ES run.
- A **combined fitness function** (`src/fitness.py`): a survival gate (did
  the robot fall?) followed by a direct-path score (distance to target,
  penalized for wasted movement), built on
  `ariel.simulation.tasks.targeted_locomotion`'s existing functions.
- Full **reproducibility logging** per run: per-generation CSV, best genome,
  and a manifest recording hardware, software versions, git commit, config,
  timing, and why the run stopped.
- **Plots** for the required convergence comparison plus diagnostics (see
  `analysis/README.md`).

## What's NOT implemented yet

- The **restart-with-increasing-population** strategy itself (IPOP) — the
  actual research question. `src/cma_es.py` is deliberately structured so
  that a restart is just "construct a new `CMAES` with `lambda_` doubled",
  without modifying the class.
- A final choice of body/world — both `turtle` and `iguana` (with the fixed
  Olympic Arena world) are run in parallel for now.

## Quickstart

```bash
cd assignments/assignment_2

# 1. Size a feasible evaluation budget for THIS machine.
uv run python experiments/benchmark_eval_time.py --body turtle

# 2. Sanity-check the CMA-ES implementation (cheap, no MuJoCo).
uv run pytest tests/test_cma_es_sanity.py

# 3. Run the sweep (see experiments/README.md for the full loop).
uv run python experiments/run_cma_es.py --body turtle --seed 0 --budget 1500
uv run python experiments/run_random_search.py --body turtle --seed 0 --budget 1500 --batch-size <printed lambda>

# 4. Aggregate and plot.
uv run python analysis/aggregate_results.py
uv run python analysis/make_plots.py
```

## Directory layout

```
src/            core library (config, bodies, controller, simulate, fitness,
                cma_es, random_search, logging_utils) — see src/README.md
experiments/    CLI runners (benchmark, run_cma_es, run_random_search)
                — see experiments/README.md
analysis/       aggregate_results.py, make_plots.py — see analysis/README.md
tests/          CMA-ES sanity check (sphere function, no MuJoCo)
__data__/       raw per-run logs (gitignored)
results/        generated plots, one subfolder per body
```

## Key design decisions (and why)

- **CMA-ES is hand-written**, not built on `ariel.ec`'s generational
  `EA`/`EAOperation` pipeline — that pipeline is shaped for
  parent-selection/crossover/mutation/survivor-selection GAs, which CMA-ES's
  ask/tell, mean/covariance update doesn't fit. See `src/README.md` for the
  full rationale.
- **Population size (`lambda`) defaults to the paper's formula**
  (`4 + floor(3*ln(n))`) but is fully configurable per run — this is both
  the correct starting point for the future restart variant (which must
  start from this default and double it) and the axis a manual
  population-size sweep can vary.
- **Fitness combines both given metrics** (fall penalty + direct-path
  distance/efficiency) into one score, rather than running them as separate
  experiment conditions.
- **Evaluation budget is sized empirically** via `benchmark_eval_time.py`
  rather than assumed, since the paper's own budget (`n * 10^4` evaluations)
  is calibrated for cheap analytic functions, not ~10s MuJoCo rollouts. On an
  M1/M2 MacBook, both `turtle` and `iguana` run at roughly 0.15-0.25s per
  evaluation at a 10s sim duration, so the default `--budget 1500` costs a
  few minutes per run.
- **Body and world are config parameters, not hardcoded** — both `turtle`
  and `iguana` (fixed Olympic Arena world) run through the identical
  pipeline now, so restricting to one later is a config change, not a
  rewrite.
