# `analysis/` — aggregation and plotting

Both scripts work per experiment (`--experiment <name>`, the folder under
`__data__/` written by `experiments/run_sweep.py`).

- **`aggregate_results.py`** — scans
  `__data__/<experiment>/<body>__<algorithm>/seed_<n>/` and writes, to
  `results/<experiment>/` (small, meant to be committed):
  `aggregated_generations.csv` (one row per generation per run, incl.
  best-so-far and lambda), `aggregated_summary.csv` (one row per run: final
  fitness, final distance to target, restarts, timing), `summary_stats.csv`
  (mean/std/median final distance per algorithm), `pairwise_tests.csv`
  (two-sided Mann-Whitney U between algorithms; needs >= 2 runs each, and is
  only meaningful with 5 seeds), and a copy of every run's small files under
  `runs/`. Re-run whenever runs are added.
- **`make_plots.py`** — reads only `results/<experiment>/`, so plots can be
  regenerated from committed data, and writes to
  `results/<experiment>/<body>/`:
  - `convergence.png` — the required plot: mean ± std of best-so-far
    distance to target vs. evaluations, per algorithm. Each run is sampled on
    a common evaluation grid first, since IPOP's growing population makes
    runs log at different evaluation counts.
  - `ipop_population_size.png` — lambda over evaluations per IPOP seed
    (each step = a restart).
  - `sigma_evolution.png` — step size per seed for CMA-ES and IPOP.
  - `final_distance_distribution.png` — final distance per run.
  - `wall_clock_time.png`, `best_trajectory.png` (replays each seed's best
    genome with the config from its manifest), `environment_snapshots/`.

  `--skip-renders` skips the (slow) trajectory replays and renders. On Linux
  the script renders through EGL (`MUJOCO_GL=egl`).

```bash
python analysis/aggregate_results.py --experiment ipop_l10_b8000
python analysis/make_plots.py --experiment ipop_l10_b8000
```

`results/turtle/` and `results/iguana/` hold the earlier CMA-ES vs. random
search plots (old fitness/controller, 10 s episodes, lambda = 25); they are
not comparable with the IPOP experiment.
