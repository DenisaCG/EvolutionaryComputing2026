# `analysis/` — aggregation and plotting

The aggregation and plotting scripts work per experiment (`--experiment <name>`, the folder under
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

`results/ipop_l10_b8000/` is the 2-seed pilot of the IPOP experiment, run
before two fixes: random rugged terrain per evaluation (now seeded) and a
fixed 25-generation stagnation window (now K = 10 + ceil(4.5 n / lambda)).
Use it as the tuning record, not as final results.

To watch a saved controller, run this from the repository root on macOS or
Windows:

```bash
uv run python assignments/assignment_2/analysis/replay.py assignments/assignment_2/__data__/ipop_l10_b12000/turtle__cma_es/seed_2/best_genome.json
```

You can also pass the run folder instead of `best_genome.json`. Its sibling
`manifest.json` supplies the body, controller settings, and terrain seed.
Paths are relative to your terminal's current directory; absolute paths work
from anywhere. Quote paths containing spaces. On macOS, the script switches
to `mjpython` automatically and supplies uv's Python library search path.
The simulation starts automatically, stops at the saved episode duration,
and leaves the viewer open until you close it.
Pass `--duration 60` to replay for 60 simulated seconds instead of the saved
episode duration. This does not change the saved run or its fitness.

`results/turtle/` and `results/iguana/` hold the earlier CMA-ES vs. random
search plots (old fitness/controller, 10 s episodes, lambda = 25); they are
not comparable with the IPOP experiment.
