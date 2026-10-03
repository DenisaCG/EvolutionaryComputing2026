# `experiments/` — CLI runners

Arshana Update: IPOP restart logic is now available as `IPOPCMAES` in
`../src/ipop_cma_es.py`; see `../src/README.md` for its ask/tell usage.
This change adds only the strategy library: the scripts below still launch
basic CMA-ES or random search, and there is no IPOP CLI flag or runner yet.
No experiments or tests were executed for this addition.

A future IPOP runner should use one wrapper for the entire evaluation budget,
check its argument-free `stopping_reason()`, and save results under a separate
algorithm condition (for example `ipop_cma_es`). The current configuration's
algorithm type annotation still lists only the two original algorithms.
Use cumulative records for `RunLogger`, retain `restart_history` in manifest
metadata (each record can be converted with `dataclasses.asdict`), and save
the wrapper's overall best genome. Record initial and final population sizes.
If no full population fits, there is no genome to save. Match comparisons to
actual `evals_used`: the wrapper can leave a remainder smaller than the next
full population, whereas the existing runners can overshoot their budgets.

Each script here is a standalone process producing one run's worth of data
under `__data__/`. Run from the `assignment_2/` directory (or anywhere — each
script resolves paths relative to its own location, not the caller's cwd).

- **`benchmark_eval_time.py`** — run this FIRST, on the machine that will run
  the full sweep. Times a handful of random-weight episodes for a given body
  and prints seconds/evaluation plus a recommended `--budget` for a chosen
  wall-clock ceiling. Physics cost scales with body complexity and
  `--sim-duration`, not with CMA-ES itself.
- **`run_cma_es.py`** — one CMA-ES run: one body, one seed, one budget. Run
  once per seed (`--seed 0` through `--seed 4` for the assignment's required
  >=5 independent runs). Prints the resolved population size (`lambda`) —
  note it down, `run_random_search.py`'s `--batch-size` should match it.
- **`run_random_search.py`** — the baseline, matched to the same
  `--budget` (total evaluations) as its paired CMA-ES run, per the hard
  constraint that random search must match evaluation count, not generation
  count or population size.

## A full sweep

```bash
for body in turtle iguana; do
  for seed in 0 1 2 3 4; do
    uv run python run_cma_es.py --body $body --seed $seed --budget 1500
    uv run python run_random_search.py --body $body --seed $seed --budget 1500 --batch-size <lambda from above>
  done
done
```

`<lambda from above>` is body-dependent (it's a function of that body's NN
weight count) but seed-independent — run one CMA-ES seed first, read its
printed `lambda=...`, then reuse it for every seed's paired random-search run
for that body.
