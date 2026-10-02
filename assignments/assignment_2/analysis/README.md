# `analysis/` — aggregation and plotting

Run after `experiments/` has produced at least one run per body/algorithm.

- **`aggregate_results.py`** — scans every
  `__data__/<body>__<algorithm>/seed_<n>/` directory and writes two tidy
  CSVs: `__data__/aggregated_generations.csv` (long format, one row per
  generation per run — for convergence plots) and
  `__data__/aggregated_summary.csv` (one row per run — for final-fitness and
  timing plots). Re-run this whenever new runs are added; it always rescans
  everything under `__data__/`.
- **`make_plots.py`** — reads those two CSVs and writes every plot to
  `results/<body>/*.png`, styled via the shared `style_guidelines_plots.py`
  at the assignment root. Produces, per body: the required convergence line
  plot (mean +/- std fitness vs. evaluation count), a log-scale
  best-fitness convergence plot, CMA-ES's step-size trace per seed, a
  final-fitness box plot, per-run wall-clock timing, and the best genome's
  trajectory in the arena.

```bash
uv run python analysis/aggregate_results.py
uv run python analysis/make_plots.py
```
