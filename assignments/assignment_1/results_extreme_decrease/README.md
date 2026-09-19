# Extreme-decrease mutation sensitivity experiment

**Completed: 60 runs, 306,000 objective evaluations. The steeper decrease did not produce a clear exponential-scheduling advantage: paired wins remain 10/10, with Wilcoxon W=102 and p=0.927279.**

## Motivation and scope

This follow-up was selected after observing the original 0.8 -> 0.2
experiment, which showed no clear advantage for exponential scheduling over
its mean-matched constant baseline. It tests whether retaining less mutation
late in evolution produces a clearer difference. It is a sensitivity analysis,
not the original preselected experiment.

Only the exponential endpoint changes, from 0.2 to 0.01. The old assignment
remains commented immediately above the active endpoint in `ea_common.py`.
The constant baseline is recalculated from the new schedule; it is not a
hard-coded probability. All operators, selection, replacement, initialization,
targets and fitness remain unchanged. The current analysis includes Denisa's
population best/mean/worst statistics and both new plots, without rewriting
those functions.

## Schedule and shared settings

For schedule index `i = 0, ..., 49`:

```text
p_exp(i) = 0.8 * (0.01 / 0.8) ** (i / 49)
p_constant = sum(p_exp(i) for i in range(50)) / 50
```

- Offspring generation 1 uses index 0: exactly **0.8**.
- Offspring generation 50 uses index 49: exactly **0.01**.
- Arithmetic mean / constant probability: **0.18489397506306005**.
- Expected mutation attempts per run in either condition:
  `100 * sum(rates) = 100 * 50 * p_constant` = **924.4698753153002**.

These decimals are the actual Python float values used in the experiment.
[mutation_schedule.csv](mutation_schedule.csv) records all probabilities and
indices. Mutation probability gates a call to the existing repaired mutation
operator. A gate success need not change the genome: existing operator no-ops,
repair retries and fallback behavior are preserved.

| Setting | Value |
|---|---|
| Seeds | 0 through 19, paired across conditions |
| Conditions | Constant, exponential, freshly rerun random search |
| Population | 100 |
| Offspring generations | 50 |
| Evaluations per run | 100 initial + 50 x 100 = 5,100 |
| Total full-run evaluations | 60 x 5,100 = 306,000 |
| Representation | ARIEL TreeGenome |
| Initialization | 20 non-core modules plus core; 21 nodes, depth-valid draws |
| Maximum depth | 12, also enforced during initialization |
| Selection | Tournament size 3, minimizing fitness |
| Crossover | Repaired subtree crossover, always invoked; 50 calls/generation |
| Mutation mixture | Existing 50/50 node replacement / subtree replacement |
| Mutation repair | Up to 20 attempts, then existing parent-copy fallback |
| Replacement | Full generational replacement; no elitism |
| Fitness | Mean + population std of weighted tree-edit distances to five targets |
| Target sizes | 7, 11, 15, 19 and 25 nodes |
| Metric costs | Insertion/deletion/type mismatch 1; rotation mismatch 0.5 |

The objective evaluation count includes all initial and offspring evaluations,
including unchanged copies. Each objective compares one candidate with five
targets. Reporting best-so-far does not introduce elitism: a historical best
may no longer survive in the current population. Population node count is
variable, whereas population cardinality stays exactly 100.

Random search samples the depth-valid initialization distribution for 5,100
evaluations per seed, independently drawing 21-node bodies. It does not sample
the full variable-size domain reachable through evolution. Its 20 runs were
rerun for this experiment, not copied from earlier outputs.

## Reproduction and raw data

Commands from the repository root (the raw runners refuse existing seed folders):

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
$env:MPLCONFIGDIR = "$PWD/assignments/assignment_1/__data__/extreme_decrease_support/matplotlib"
foreach ($seed in 0..19) {
    .\.venv\Scripts\python.exe -B assignments/assignment_1/run_ea.py --seed $seed --mutation-schedule constant --extreme-decrease
    .\.venv\Scripts\python.exe -B assignments/assignment_1/run_ea.py --seed $seed --mutation-schedule exponential --extreme-decrease
    .\.venv\Scripts\python.exe -B assignments/assignment_1/run_random_search.py --seed $seed --dynamic-scheduler --extreme-decrease
}
.\.venv\Scripts\python.exe -B assignments/assignment_1/analyze_results.py --experiment extreme
.\.venv\Scripts\python.exe -B assignments/assignment_1/render_bodies.py --experiment extreme
```

The actual batch dispatches these same per-run commands through six independent
subprocesses, as in the original experiment. Each process seeds both Python and
NumPy. The execution manifest records exact commands, environment, versions,
source/target hashes, timings, exit codes and logs. No dependencies were added.
The `--extreme-decrease` flag selects the new raw paths; the endpoint itself is
the active constant in `ea_common.py`. `--experiment extreme` selects the new
analysis/render paths. Existing `dynamic` analysis reads recorded old schedules.
Do not run old analysis/render commands when preserving old artifacts unchanged.

All paths below are relative to `assignments/assignment_1`:

```text
__data__/extreme_decrease/ea/dynamic_constant/seed_N/
    database.db, metadata.json, generation_log.csv
__data__/extreme_decrease/ea/dynamic_exponential/seed_N/
    database.db, metadata.json, generation_log.csv
__data__/extreme_decrease/random_search_dynamic_scheduler/seed_N/
    results.csv, best_genome.json, metadata.json
__data__/extreme_decrease_support/
    execution_manifest.json, smoke_report.json, validation_report.json
    before_sha256.json, run_logs/, smoke/, matplotlib/
results_extreme_decrease/
    README.md, mutation_schedule.csv, tables/, plots/, manifests/
```

The inner condition names remain `dynamic_constant` and `dynamic_exponential`
so the existing pipeline can be reused. The enclosing `extreme_decrease` raw
root and `results_extreme_decrease` result root unambiguously isolate this study.
Raw data and PNGs may be excluded by existing Git ignore rules; include them
explicitly when packaging or sharing results. No commit or push was made.

## Verification before the full batch

All requested preflight checks passed; see [smoke report](manifests/smoke_report.json).
They verify exact endpoints and mean matching, generation/index mapping,
unchanged operators and analysis functions, real-metric paired initialization,
depth rejection/resampling, preserved crossover counts and overwrite refusal.
Isolated full-length checks use real ARIEL and reproduction with a cheap
node-count objective to check exactly 5,100 evaluations per EA and random-search
run before expensive runs. These smoke results are separate from the full
experiment, which uses the unchanged actual fitness throughout.

## Completed runs and final best-so-far fitness

All **60 runs completed**, each with **5,100 actual objective evaluations**: **306,000 total**. Six-worker batch time was 2105.7 seconds (35.09 minutes). No failed run required a retry.

| Condition | Mean | Std | Median | Minimum | Maximum |
|---|---|---|---|---|---|
| Constant | 12.801965681 | 0.287214462 | 12.801920241 | 12.200000000 | 13.380624847 |
| Exponential | 12.853572732 | 0.383350550 | 12.838046165 | 12.227361850 | 13.557583690 |
| Random search | 17.126858443 | 0.235418323 | 17.200000000 | 16.378232998 | 17.460232527 |

All descriptive standard deviations use `ddof=0`. Plot bands are standard deviations, not confidence intervals. The endpoint is the best fitness found anywhere in a run.

### Paired comparison

| Statistic | Value |
|---|---|
| Mean exponential minus constant | 0.051607050905891946 |
| Median difference | 0.02070828735973773 |
| Std of differences | 0.5260856417778716 |
| Exponential wins | 10 |
| Constant wins | 10 |
| Ties | 0 |
| Two-sided Wilcoxon W | 102.0 |
| Wilcoxon p | 0.9272785186767578 |

Negative differences favor exponential mutation. The existing SciPy Wilcoxon test uses `alternative="two-sided"`, `zero_method="wilcox"`, `method="auto"`; only test inputs are rounded to 12 decimal places. Unrounded differences remain in the CSV. These are exploratory, unadjusted follow-up statistics. A non-significant result does not establish equivalence.

### Actual mutation attempts

| Condition | Mean | Std | Minimum | Maximum |
|---|---|---|---|---|
| Constant | 922.900000 | 23.458261 | 890.000000 | 966.000000 |
| Exponential | 919.250000 | 21.201120 | 878.000000 | 975.000000 |

Both conditions have the same expected 924.4698753153002 attempts, but realized gate counts fluctuate. Counts exclude internal repair retries and do not imply successful genotype changes. Every full EA run recorded 2,500 crossover calls.

## Convergence and current population analysis

| Evaluations | Constant best-so-far | Exponential best-so-far | Random-search best-so-far |
|---|---|---|---|
| 100 | 18.245128 | 18.245128 | 18.245128 |
| 500 | 14.825109 | 14.787639 | 17.769256 |
| 1000 | 14.013049 | 14.007090 | 17.451209 |
| 2000 | 13.324811 | 13.443937 | 17.269788 |
| 3000 | 13.083042 | 13.117632 | 17.208075 |
| 4000 | 12.878240 | 12.941653 | 17.163668 |
| 5100 | 12.801966 | 12.853573 | 17.126858 |

The current pipeline records each generation's own best/mean/worst fitness as well as the historical best-so-far. Across-seed means at generation 50:

| Condition | Generation best | Population mean | Generation worst | Historical best-so-far |
|---|---|---|---|---|
| Constant | 13.127865 | 14.762258 | 18.441282 | 12.801966 |
| Exponential | 13.046224 | 14.695926 | 18.889800 | 12.853573 |
| Random search | 18.226569 | 20.790974 | 23.293656 | 17.126858 |

Random-search population statistics refer to independent batches of 100 draws. The generation-best curve can worsen because replacement has no elitism; best-so-far cannot worsen.

### 80%-of-total-improvement checkpoints

These reproduce Denisa's current plotting calculation exactly: on the **across-seed mean generation-best curve**, take the first checkpoint where `initial_mean_generation_best - current_mean_generation_best >= 0.8 * (initial_mean_generation_best - final_mean_generation_best)`. They are not median per-run convergence times, not based on best-so-far, and not evidence of permanent convergence. A worse final generation can also change this retrospective threshold.

| Experiment | Condition | Generation | Evaluations |
|---|---|---|---|
| original_0.8_to_0.2 | Constant | 8 | 900 |
| original_0.8_to_0.2 | Exponential | 12 | 1300 |
| extreme_0.8_to_0.01 | Constant | 9 | 1000 |
| extreme_0.8_to_0.01 | Exponential | 11 | 1200 |

The old checkpoints are derived from the existing old `condition_summary.csv`, without regenerating any old output. Exact thresholds are in [improvement_80pct.csv](tables/improvement_80pct.csv).

The unchanged alternative convergence measure uses the common post-hoc threshold **13.442063965096**, 1.05 times the better final EA mean.

| Condition | Runs reaching threshold | Median evaluations among reaching runs | Range |
|---|---|---|---|
| Constant | 20/20 | 1950.0 | 800 to 3700 |
| Exponential | 17/20 | 1900.0 | 700 to 3500 |
| Random search | 0/20 | None | Not reached |

## Per-target distances, tree size and best bodies

| Condition | target_00 | target_01 | target_02 | target_03 | target_04 |
|---|---|---|---|---|---|
| Constant | 11.275000 | 11.250000 | 11.650000 | 12.375000 | 13.150000 |
| Exponential | 10.975000 | 11.050000 | 11.250000 | 12.350000 | 13.350000 |
| Random search | 16.125000 | 16.100000 | 16.025000 | 16.625000 | 16.875000 |

These are distances from each seed's best body, averaged over 20 seeds. Distances are not normalized for target size; the objective includes their mean and standard deviation.

| Condition | Initial mean nodes | Final mean nodes | Std across final seed means | Lowest mean nodes | Generation at minimum |
|---|---|---|---|---|---|
| Constant | 21.0 | 12.176500 | 0.621862 | 10.408000 | 6 |
| Exponential | 21.0 | 12.210500 | 0.703189 | 9.957500 | 5 |

Population cardinality remains 100 throughout. The tree-size plot measures node count, including the core; the initial 21-node size is a reference, not a cap on evolved size.

| Condition | Best seed | Fitness | Nodes | Closest target | Distance |
|---|---|---|---|---|---|
| Constant | 17 | 12.200000000 | 15 | target_01 | 11.5 |
| Exponential | 2 | 12.227361850 | 13 | target_00 | 10.5 |
| Random search | 7 | 16.378232998 | 21 | target_00 | 14.5 |

Best-body renders show each selected body beside its nearest target using the unchanged rendering pipeline. They are qualitative morphology comparisons, not locomotion or collision-validity evaluations.

## Descriptive comparison with the original experiment

| Schedule | Condition | Mean final best | Std |
|---|---|---|---|
| 0.8 -> 0.2 | Constant | 12.905212242 | 0.246997364 |
| 0.8 -> 0.2 | Exponential | 12.902021982 | 0.313511477 |
| 0.8 -> 0.2 | Random search | 17.126858443 | 0.235418323 |
| 0.8 -> 0.01 | Constant | 12.801965681 | 0.287214462 |
| 0.8 -> 0.01 | Exponential | 12.853572732 | 0.383350550 |
| 0.8 -> 0.01 | Random search | 17.126858443 | 0.235418323 |
| Schedule | Mean paired delta | Median paired delta | Exp / constant wins | Wilcoxon W | p |
|---|---|---|---|---|---|
| 0.8 -> 0.2 | -0.003190260 | -0.038748763 | 10 / 10 | 100.0 | 0.8694877624511719 |
| 0.8 -> 0.01 | 0.051607051 | 0.020708287 | 10 / 10 | 102.0 | 0.9272785186767578 |

The authoritative old values come from the existing `results_dynamic_scheduler/tables/final_fitness_summary.csv`, `paired_summary.json` and `condition_summary.csv`. No old analysis was rerun. Within each experiment expected mutation totals are matched, but across experiments the total falls from 2170.9031668279813 to 924.4698753153002. Thus a cross-experiment change cannot be attributed solely to temporal allocation. Matching seeds are reused, so these are not 40 new independent seeds per condition. The identical random-search results are expected from rerunning an unchanged deterministic seeded procedure.

## Output inventory

### Plots

- [best_mean_worst_plot.png](plots/best_mean_worst_plot.png)
- [best_so_far_vs_generation_best.png](plots/best_so_far_vs_generation_best.png)
- [convergence_plot.png](plots/convergence_plot.png)
- [final_fitness_violin.png](plots/final_fitness_violin.png)
- [per_target_distance.png](plots/per_target_distance.png)
- [render_dynamic_constant.png](plots/render_dynamic_constant.png)
- [render_dynamic_exponential.png](plots/render_dynamic_exponential.png)
- [render_random_search.png](plots/render_random_search.png)
- [tree_size_plot.png](plots/tree_size_plot.png)

### Tables

- [condition_summary.csv](tables/condition_summary.csv)
- [convergence_speed.csv](tables/convergence_speed.csv)
- [final_fitness_summary.csv](tables/final_fitness_summary.csv)
- [generation_summary.csv](tables/generation_summary.csv)
- [improvement_80pct.csv](tables/improvement_80pct.csv)
- [paired_differences.csv](tables/paired_differences.csv)
- [paired_summary.json](tables/paired_summary.json)
- [per_target_distance.csv](tables/per_target_distance.csv)
- [tree_size_summary.csv](tables/tree_size_summary.csv)

### Manifests

- [artifact_verification.json](manifests/artifact_verification.json)
- [analysis_findings.json](manifests/analysis_findings.json)
- [best_individuals.json](manifests/best_individuals.json)
- [execution_manifest.json](manifests/execution_manifest.json)
- [preservation_report.json](manifests/preservation_report.json)
- [smoke_report.json](manifests/smoke_report.json)
- [validation_report.json](manifests/validation_report.json)

The numeric findings JSON contains supporting checkpoint values and precise aggregates. The supplemental 80% CSV only exports values from the existing plotting definition; it does not alter the analysis method.

## Interpretation and limitations

The steeper decrease does **not** produce a clearer final-best advantage for
exponential scheduling over its mean-matched baseline in these 20 seeds.
Constant mutation has the slightly lower mean (12.801966 versus 12.853573),
the lower median, and the smaller observed across-seed standard deviation.
The paired mean difference is +0.051607, but wins remain 10/10 and the
Wilcoxon result does not detect a clear systematic difference. This is not
proof that the conditions are equivalent.

Both new EA means are descriptively lower than their original counterparts:
constant improves from 12.905212 to 12.801966, and exponential from 12.902022
to 12.853573. The original paired mean was -0.003190 with p=0.869488; the
follow-up changes its sign but does not establish an advantage for either
schedule. This follow-up was chosen after seeing the original result, so its
p-value is exploratory, not a preplanned confirmatory test or a multiplicity-
adjusted result. No cross-experiment significance claim is made.

Convergence is broadly similar. Exponential is slightly ahead early, the
curves nearly meet around 1,000 evaluations, and constant is slightly ahead
at the reported later checkpoints. The 80% points move from the original
900/1,300 evaluations (constant/exponential) to 1,000/1,200. These use each
condition's own retrospective improvement, so they do not establish a
uniform speed advantage. For the common threshold 13.442064, constant reaches
it in 20/20 runs and exponential in 17/20. Their conditional medians of
1,950 and 1,900 evaluations must be read with those differing success counts;
the latter omits three non-reaching runs. Random search never reaches it.

Denisa's generation statistics distinguish current population quality from
historical best-so-far. Exponential ends with a slightly better mean current-
generation best (13.046224 versus 13.127865) and population mean (14.695926
versus 14.762258), but a worse mean current-generation worst (18.889800
versus 18.441282). Neither condition always retains its historical best:
mean generation-best worsens at 12 checkpoints for constant and 11 for
exponential. These are observations under full replacement without elitism,
not evidence that any particular mutation mechanism caused the differences.

Exponential has lower mean distance on targets 00-03, while constant is
better on target 04, the largest target. This can coexist with constant's
better composite fitness: averaged across seeds, mean target distance is
11.940 for constant and 11.795 for exponential, but the average within-body
across-target standard-deviation penalty is approximately 0.861966 and
1.058573, respectively. The composite objective therefore changes the
ordering. Per-target bars alone should not be interpreted as the objective.

Tree sizes initially shrink from 21 nodes to means near 10, then gradually
recover to 12.1765 (constant) and 12.2105 (exponential). There is no runaway
mean bloat in these runs. This does not rule out individual large trees or
show that their size distributions are identical. Population cardinality
remains 100; the depth cap is 12, not a hard node-count cap.

The selected best constant body (seed 17, 15 nodes) and exponential body
(seed 2, 13 nodes) are compact but visibly differ in branching and module
arrangement from their closest targets. They are not near-exact matches.
The random-search best (seed 7, 21 nodes) is more structurally elaborate.
These single best examples do not establish general morphology quality,
and the still-frame renders do not test locomotion or collision validity.
The inherited renderer labels its selected random-search body "best evolved
body"; this is a caption convention, not a claim that random search evolves.

## Final verification, source scope and execution note

[Validation](manifests/validation_report.json) passed for all 60 real-fitness
runs: 5,100 evaluations each, 100 individuals in each EA generation 0-50,
finite evaluated fitness, full replacement, correct schedule logs, and
2,500 crossover calls per EA run. All 20 paired initial populations also
match the original experiment; random-search first-population fitness
matches the EAs. All 20 freshly rerun random-search CSVs match their originals
byte for byte. All 8,000 initial/final EA genomes pass the depth check, and
the analysis independently re-scores each of the 60 best bodies against the
five targets. None of these checks changes raw data.

[Preservation](manifests/preservation_report.json) checked hashes of **2,668
existing files**. Only these five existing source files changed:

| Source file | Change |
|---|---|
| `ea_common.py` | Comment the original 0.2 endpoint and activate 0.01; start and schedule formula unchanged |
| `run_ea.py` | Add guarded `--extreme-decrease` routing to the separate raw root |
| `run_random_search.py` | Add the same separate-root routing for the freshly rerun baseline |
| `analyze_results.py` | Add `--experiment extreme`, preserving all current statistics and plotting functions |
| `render_bodies.py` | Add the corresponding new result-root selection |

The source diff is 31 insertions and 12 deletions across those five files.
There are no new Python source files. All non-main functions match the
current pre-edit HEAD AST, including Denisa's new analysis functions.
`results_dynamic_scheduler`, `__results__`, all historical raw data,
`src/ariel`, the assignment template, target files and fitness implementation
are unchanged. Nothing was committed or pushed.

All 60 search processes exited successfully. The **first analysis attempt**
failed at the native Windows level with exit `0xc000041d` while beginning
plotting, before producing any PNG. The precise native cause was not
established. Repeating the unchanged pipeline with the noninteractive
Matplotlib `Agg` backend succeeded; rendering also succeeded with that
backend. No search was rerun, and no old output was regenerated. Both the
failed attempt and successful postprocessing commands are recorded in the
execution manifest. Use `MPLBACKEND=Agg` as in the commands above.

All nine PNGs were visually inspected, including the two new population
analysis plots and all three best-body renders. The figures are complete
and readable. Existing Git rules ignore this new folder's PNGs and raw data,
so package them explicitly alongside the tracked CSV/JSON/README files.
