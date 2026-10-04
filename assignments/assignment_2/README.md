# Assignment 2 — CMA-ES Neuroevolution for Targeted Locomotion

**Research question:** *Does adding a restart strategy with increasing
population size improve CMA-ES performance?* (IPOP-CMA-ES, Auger & Hansen,
CEC 2005 — [paper](https://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf).)

This codebase currently implements the **basic (mu_W, lambda)-CMA-ES**
plus a reusable experiment pipeline — config, logging, plotting — and an
IPOP restart wrapper at the library level. The existing experiment scripts
still run basic CMA-ES and random search. Two bodies (`turtle`, `iguana`) are run side by side right now to
assess feasibility and results before the team restricts to one.

Arshana Update: Added `src/ipop_cma_es.py` with `IPOPCMAES`, a wrapper around
the existing handwritten optimizer. Following Auger & Hansen (2005), Section 2,
a local stopping criterion triggers a fresh CMA-ES run with twice the population.
The wrapper preserves the overall best solution and cumulative evaluation count
under one shared budget. See `src/README.md` for initialization and API details.
This addition has not been executed or tested; no experiments or figures were
generated. Runner/configuration/analysis integration remains future work.

## What's implemented

- A **hand-written** (mu_W, lambda)-CMA-ES (`src/cma_es.py`) — no imported
  optimizer library, per the assignment's rules. Validated against the
  analytic sphere function in `tests/test_cma_es_sanity.py` before ever
  touching MuJoCo.
- A **random-search baseline** (`src/random_search.py`), matched to the same
  total evaluation budget as its paired CMA-ES run.
- Arshana Update: **IPOP restart logic** (`src/ipop_cma_es.py`), available as
  an ask/tell library class; not yet selected by an experiment runner.
- A **fitness function** (`src/fitness.py`): `ariel`'s
  `fitness_survival_and_locomotion` — change in planar distance to the
  target, or a flat penalty of 10 if the core drops below 0.05 m (falling
  off the arena).
- Full **reproducibility logging** per run: per-generation CSV, best genome,
  and a manifest recording hardware, software versions, git commit, config,
  timing, and why the run stopped.
- **Plots** for the required convergence comparison plus diagnostics (see
  `analysis/README.md`).

## What's NOT implemented yet

- Arshana Update: **IPOP experiment integration and evaluation** — the strategy
  now exists, but a dedicated runner/output condition and comparison runs still
  need to be added before the research question can be answered.
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
                cma_es, ipop_cma_es, ec_engine, random_search, logging_utils)
                — see src/README.md
experiments/    CLI runners (benchmark, run_cma_es, run_random_search)
                — see experiments/README.md
analysis/       aggregate_results.py, make_plots.py — see analysis/README.md
tests/          CMA-ES sanity check (sphere function, no MuJoCo)
__data__/       raw per-run logs (gitignored)
results/        generated plots, one subfolder per body
```

## Key design decisions (and why)

- **Built on `ariel.ec`, with a hand-written CMA-ES update.** Each
  generation runs as `ariel.ec` `EAOperation`s (sample -> evaluate ->
  update) inside `ariel.ec.EA`, which also stores every individual in an
  SQLite database. The CMA-ES mean/covariance update itself is our own
  (`src/cma_es.py`), since `ariel.ec`'s GA operators don't fit it. See
  `src/README.md` for the full rationale.
- **Population size (`lambda`) defaults to the paper's formula**
  (`4 + floor(3*ln(n))`) but is fully configurable per run — this is both
  the default starting point for the restart wrapper (which doubles it
  on each restart) and the axis a manual
  population-size sweep can vary.
- **Fitness is distance reduced plus a fall penalty**
  (`fitness_survival_and_locomotion`). With a fixed spawn and target it ranks
  controllers exactly like the template's plain distance to target. The
  earlier `fitness_direct_path` wasted-path penalty was dropped: path length
  summed per physics step counts the turtle's crawling wobble as waste, so
  most random controllers scored worse than standing still (17/20 in a
  check), and the straight +x route to the target already rewards a direct
  path.
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
