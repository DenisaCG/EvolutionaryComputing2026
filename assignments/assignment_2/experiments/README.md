# `experiments/` — CLI runners

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
