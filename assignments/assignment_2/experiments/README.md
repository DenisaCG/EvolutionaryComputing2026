# `experiments/` — CLI runners

Each runner is a standalone process producing one run's worth of data
(`generations.csv`, `best_genome.json`, `manifest.json`, `database.db`) under
`__data__/` (or `--output-root`). Run from anywhere — each script resolves
paths relative to its own location, not the caller's cwd.

- **`run_sweep.py`** — the entry point for real experiments. Runs every
  condition (CMA-ES, IPOP-CMA-ES, random search) x seed of one experiment in
  parallel with identical body, budget, initial population size and seeds,
  writing to `__data__/<experiment>/` with per-run logs in
  `__data__/<experiment>/logs/`. Finished runs (manifest present) are
  skipped, so extending from 2 to 5 seeds is the same command with more
  `--seeds`.
- **`run_cma_es.py`** — one CMA-ES run without restarts. Uses the whole
  budget; local stopping criteria are only logged (`first_local_stop` in the
  manifest — where IPOP would have restarted).
- **`run_ipop_cma_es.py`** — one IPOP-CMA-ES run: restarts with doubled
  `lambda` when a local stopping criterion fires (the paper's 5 plus the
  task-specific `stagnation` criterion), up to `--max-lambda`; then restarts
  keep the largest `lambda` reached. The manifest records `restart_history`.
- **`run_random_search.py`** — the baseline, matched on total evaluations;
  `--batch-size` only groups evaluations for logging.
- **`benchmark_eval_time.py`** — times random-weight episodes to size a
  `--budget` for a wall-clock ceiling on the machine that will run the sweep.

All runners only evaluate full generations, so they never exceed `--budget`.

## The IPOP experiment

```bash
# 2-seed round, then the same command with --seeds 0 1 2 3 4
python experiments/run_sweep.py --experiment ipop_l10_b8000 --seeds 0 1 \
    --budget 8000 --lambda0 10 --jobs 6
```

Settings (defaults in `src/config.py`): turtle, Olympic Arena, 15 s episodes,
clock 1.3 Hz, `sigma0 = 0.5`, `lambda0 = 10`, `max_lambda = 200`, stagnation
= < 0.01 m best-so-far improvement over K = 10 + ceil(4.5 n / lambda)
generations (100, 55, 33, 22, 16 at lambda = 10..160). With the same seed,
IPOP's first run is identical to the CMA-ES run, so differences come only
from the restarts.
