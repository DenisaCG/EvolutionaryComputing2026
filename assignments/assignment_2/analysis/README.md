# `analysis/` — aggregation and plotting

Arshana Update: `../src/ipop_cma_es.py` now implements the IPOP restart
strategy at the library level. No IPOP run data or new plots were generated;
the existing results still compare basic CMA-ES with random search.

The wrapper returns the existing `GenerationRecord` schema with cumulative
evaluation/generation counters across restarts, so a future runner can use
the same generation CSV format. Restart events are available separately via
`restart_history`. Analysis code is unchanged: an `ipop_cma_es` condition
fits the aggregator's directory-name pattern, but plot labels/titles and the
CMA-only sigma diagnostic still need updating for an IPOP comparison.
Restarts occur at different evaluation counts across seeds; the current
exact-count grouping does not align those traces onto a common evaluation
grid. That alignment must be addressed before interpreting cross-seed means.

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
