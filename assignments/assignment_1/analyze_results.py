"""Aggregate the mutation-only / mutation+crossover / random-search results.

Reads the raw per-individual logs run_ea.py and run_random_search.py wrote to
__data__/ (SQLite for the EA variants, CSV for random search) and writes
derived, report-ready output under __results__/:

    __results__/plots/      all PNG figures
    __results__/tables/     all CSV summaries
    __results__/manifests/  best_individuals.json - consumed by render_bodies.py

Raw __data__ is never modified.

Usage
-----
    python analyze_results.py
    python analyze_results.py --experiment dynamic
"""

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from tree_edit_distance import distances_to_targets

from ea_common import NUM_GENERATIONS, NUM_MODULES, POP_SIZE, TARGET_DIR, load_targets

HERE = Path(__file__).parent
DATA_DIR = HERE / "__data__"
RESULTS_DIR = HERE / "__results__"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
MANIFESTS_DIR = RESULTS_DIR / "manifests"

EA_CONDITIONS = ("mutation_only", "mutation_crossover")
RANDOM_SEARCH_FOLDER = "random_search"
SEEDS: list[int] = [0, 1, 2, 3, 4]
CHECKPOINTS: list[int] = list(range(NUM_GENERATIONS + 1))  # generation indices

LABELS: dict[str, str] = {
    "mutation_only": "Mutation-only EA",
    "mutation_crossover": "Mutation + crossover EA",
    "random_search": "Random search",
}
# Okabe-Ito colorblind-safe categorical palette, fixed assignment per condition.
COLORS: dict[str, str] = {
    "mutation_only": "#0072B2",  # blue
    "mutation_crossover": "#D55E00",  # vermillion
    "random_search": "#666666",  # neutral gray - baseline
}


def ea_generation_stats(variant: str, seed: int) -> dict[str, list[float]]:
    """Per-generation cumulative evals, population mean/std, and best-so-far."""
    db_path = DATA_DIR / "ea" / variant / f"seed_{seed}" / "database.db"
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    rows = con.execute(
        "SELECT time_of_birth, fitness_ FROM individual ORDER BY time_of_birth",
    ).fetchall()
    con.close()

    by_gen: dict[int, list[float]] = {}
    for gen, fit in rows:
        by_gen.setdefault(gen, []).append(fit)

    if sorted(by_gen) != CHECKPOINTS or any(
        len(fits) != POP_SIZE or any(f is None or not np.isfinite(f) for f in fits)
        for fits in by_gen.values()
    ):
        raise ValueError(f"Incomplete or invalid run: {db_path}")
    cumulative_evals, pop_mean, pop_std, best_so_far = [], [], [], []
    running_best = float("inf")
    for gen in sorted(by_gen):
        fits = by_gen[gen]
        running_best = min(running_best, min(fits))
        cumulative_evals.append((gen + 1) * len(fits))
        pop_mean.append(float(np.mean(fits)))
        pop_std.append(float(np.std(fits)))
        best_so_far.append(running_best)
    return {
        "cumulative_evals": cumulative_evals,
        "pop_mean": pop_mean,
        "pop_std": pop_std,
        "best_so_far": best_so_far,
    }


def random_search_generation_stats(seed: int) -> dict[str, list[float]]:
    """Bin random-search draws into pop_size-sized batches.

    This gives a like-for-like population mean/std at the same cumulative-eval
    checkpoints as the EA variants, per the shared-x-axis plan agreed for this
    experiment (see run_random_search.py's module docstring).
    """
    path = DATA_DIR / RANDOM_SEARCH_FOLDER / f"seed_{seed}" / "results.csv"
    with path.open() as f:
        fitness = [float(row["fitness"]) for row in csv.DictReader(f)]
    if len(fitness) != (NUM_GENERATIONS + 1) * POP_SIZE or not np.isfinite(fitness).all():
        raise ValueError(f"Incomplete or invalid random-search run: {path}")

    cumulative_evals, pop_mean, pop_std, best_so_far = [], [], [], []
    running_best = float("inf")
    for gen in CHECKPOINTS:
        batch = fitness[gen * POP_SIZE : (gen + 1) * POP_SIZE]
        if not batch:
            break
        running_best = min(running_best, min(batch))
        cumulative_evals.append((gen + 1) * len(batch))
        pop_mean.append(float(np.mean(batch)))
        pop_std.append(float(np.std(batch)))
        best_so_far.append(running_best)
    return {
        "cumulative_evals": cumulative_evals,
        "pop_mean": pop_mean,
        "pop_std": pop_std,
        "best_so_far": best_so_far,
    }


def evals_to_reach(stats: dict[str, list[float]], threshold: float) -> int | None:
    """First cumulative-eval count at which best_so_far <= threshold, else None."""
    for evals, best in zip(stats["cumulative_evals"], stats["best_so_far"], strict=True):
        if best <= threshold:
            return evals
    return None


def best_individual_genome(condition: str, seed: int) -> TreeGenome:
    """The best (lowest-fitness) genome found anywhere across the whole run."""
    if condition == "random_search":
        path = DATA_DIR / RANDOM_SEARCH_FOLDER / f"seed_{seed}" / "best_genome.json"
        with path.open() as f:
            return TreeGenome.from_dict(json.load(f))

    db_path = DATA_DIR / "ea" / condition / f"seed_{seed}" / "database.db"
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    row = con.execute(
        "SELECT genotype_ FROM individual ORDER BY fitness_ ASC LIMIT 1",
    ).fetchone()
    con.close()
    return TreeGenome.from_dict(json.loads(row[0]))


def plot_final_fitness_violin(
    per_seed: dict[str, dict[int, dict[str, list[float]]]],
) -> None:
    """Violin plot of final best-so-far fitness, one violin per condition."""
    conditions = ("random_search", *EA_CONDITIONS)
    data = [[per_seed[c][seed]["best_so_far"][-1] for seed in SEEDS] for c in conditions]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    positions = np.arange(1, len(conditions) + 1)
    parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[conditions[i]])
        body.set_edgecolor(COLORS[conditions[i]])
        body.set_alpha(0.35)
    for key in ("cmeans", "cmaxes", "cmins", "cbars"):
        parts[key].set_color("#333333")
        parts[key].set_linewidth(1)

    # Overlay the actual seed values as well as the smoothed distribution.
    rng = np.random.default_rng(0)
    for i, values in enumerate(data):
        jitter = rng.uniform(-0.05, 0.05, size=len(values))
        ax.scatter(
            positions[i] + jitter, values, color=COLORS[conditions[i]],
            edgecolor="white", linewidth=0.5, zorder=3, s=30,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[c] for c in conditions])
    ax.set_ylabel(f"Final best-so-far fitness ({(NUM_GENERATIONS + 1) * POP_SIZE} evals)\nlower is better")
    ax.set_title(f"Final-fitness distribution across {len(SEEDS)} independent seeds")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "final_fitness_violin.png", dpi=200)


def per_target_distance_analysis(
    per_seed: dict[str, dict[int, dict[str, list[float]]]],
) -> dict[str, dict]:
    """Per-condition, per-target distance of each seed's best genome.

    Writes the raw table and a grouped bar chart, and returns a manifest
    (also saved as JSON) of each condition's single best individual and its
    closest target - used by render_bodies.py for the qualitative renders.
    """
    targets = load_targets()
    target_names = [p.stem for p in sorted(TARGET_DIR.glob("*.json"))]

    rows_out = []
    manifest: dict[str, dict] = {}
    for condition in (*EA_CONDITIONS, "random_search"):
        per_seed_distances = []
        best_overall: tuple[float, int, TreeGenome, list[float]] | None = None
        for seed in SEEDS:
            genome = best_individual_genome(condition, seed)
            dists = list(distances_to_targets(genome.to_networkx(), targets))
            per_seed_distances.append(dists)
            for name, d in zip(target_names, dists, strict=True):
                rows_out.append([condition, seed, name, d])

            fitness = per_seed[condition][seed]["best_so_far"][-1]
            if not np.isclose(np.mean(dists) + np.std(dists), fitness, rtol=0, atol=1e-10):
                raise ValueError(f"Best genome/fitness mismatch: {condition}, seed {seed}")
            if best_overall is None or fitness < best_overall[0]:
                best_overall = (fitness, seed, genome, dists)

        arr = np.array(per_seed_distances)
        mean_dists = arr.mean(axis=0)
        std_dists = arr.std(axis=0)
        _, best_seed, best_genome, best_dists = best_overall
        closest_idx = int(np.argmin(best_dists))
        manifest[condition] = {
            "best_seed": best_seed,
            "genotype": best_genome.to_dict(),
            "closest_target": target_names[closest_idx],
            "closest_target_distance": best_dists[closest_idx],
            "mean_distances": mean_dists.tolist(),
            "std_distances": std_dists.tolist(),
            "target_names": target_names,
        }

    with (TABLES_DIR / "per_target_distance.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", "target", "distance"])
        writer.writerows(rows_out)

    with (MANIFESTS_DIR / "best_individuals.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(target_names))
    width = 0.25
    conditions = ("random_search", *EA_CONDITIONS)
    for i, condition in enumerate(conditions):
        offset = (i - 1) * width
        ax.bar(
            x + offset, manifest[condition]["mean_distances"], width,
            yerr=manifest[condition]["std_distances"], capsize=3,
            color=COLORS[condition], label=LABELS[condition],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(target_names)
    ax.set_xlabel("Target body")
    ax.set_ylabel(
        "Tree edit distance to target\n(mean ± std of each seed's best individual)",
    )
    ax.set_title("Per-target distance of each seed's best body")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "per_target_distance.png", dpi=200)
    return manifest


def tree_size_stats(variant: str, seed: int) -> dict[str, list[float]]:
    """Per-generation cumulative evals and population mean/std module count."""
    db_path = DATA_DIR / "ea" / variant / f"seed_{seed}" / "database.db"
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    rows = con.execute(
        "SELECT time_of_birth, genotype_ FROM individual ORDER BY time_of_birth",
    ).fetchall()
    con.close()

    by_gen: dict[int, list[int]] = {}
    for gen, genotype_json in rows:
        size = len(json.loads(genotype_json)["nodes"])
        by_gen.setdefault(gen, []).append(size)

    cumulative_evals, mean_size, std_size = [], [], []
    for gen in sorted(by_gen):
        sizes = by_gen[gen]
        cumulative_evals.append((gen + 1) * len(sizes))
        mean_size.append(float(np.mean(sizes)))
        std_size.append(float(np.std(sizes)))
    return {
        "cumulative_evals": cumulative_evals,
        "mean_size": mean_size,
        "std_size": std_size,
    }


def plot_tree_size() -> None:
    """Population module count over generations for the selected EA conditions.

    Random search is intentionally excluded: bloat is a selection-driven
    phenomenon, and random search resamples independently every draw, so it
    has no generation-to-generation size trend to show.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows_out = []
    for condition in EA_CONDITIONS:
        per_seed_mean = []
        cumulative_evals_ref: list[int] | None = None
        for seed in SEEDS:
            stats = tree_size_stats(condition, seed)
            if cumulative_evals_ref is None:
                cumulative_evals_ref = stats["cumulative_evals"]
            per_seed_mean.append(stats["mean_size"])
            rows = zip(
                stats["cumulative_evals"], stats["mean_size"], stats["std_size"],
                strict=True,
            )
            for gen, (evals, mean_, std_) in enumerate(rows):
                rows_out.append([condition, seed, gen, evals, mean_, std_])

        arr = np.array(per_seed_mean)
        mean_across_seeds = arr.mean(axis=0)
        std_across_seeds = arr.std(axis=0)
        evals_arr = np.array(cumulative_evals_ref)
        color = COLORS[condition]
        ax.plot(
            evals_arr, mean_across_seeds, color=color, linewidth=2,
            label=LABELS[condition],
        )
        ax.fill_between(
            evals_arr, mean_across_seeds - std_across_seeds,
            mean_across_seeds + std_across_seeds, color=color, alpha=0.15,
            linewidth=0,
        )

    with (TABLES_DIR / "tree_size_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "seed", "generation", "cumulative_evals",
            "mean_module_count", "std_module_count",
        ])
        writer.writerows(rows_out)

    ax.axhline(
        NUM_MODULES + 1, color="#999999", linestyle="--", linewidth=1,
        label=f"initial tree size ({NUM_MODULES + 1}, including core)",
    )
    ax.set_xlabel("Cumulative fitness evaluations")
    ax.set_ylabel(f"Population module count (mean ± std over {len(SEEDS)} seeds)")
    ax.set_title("Tree size over generations (bloat check)")
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tree_size_plot.png", dpi=200)


def main(argv: list[str] | None = None) -> None:
    global EA_CONDITIONS, RANDOM_SEARCH_FOLDER, SEEDS, CHECKPOINTS
    global RESULTS_DIR, PLOTS_DIR, TABLES_DIR, MANIFESTS_DIR, POP_SIZE, NUM_GENERATIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("legacy", "dynamic"), default="legacy")
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    args = parser.parse_args(argv)
    POP_SIZE, NUM_GENERATIONS = args.pop_size, args.generations
    CHECKPOINTS = list(range(NUM_GENERATIONS + 1))
    if args.experiment == "dynamic":
        EA_CONDITIONS = ("dynamic_constant", "dynamic_exponential")
        RANDOM_SEARCH_FOLDER = "random_search_dynamic_scheduler"
        SEEDS = list(range(20))
        RESULTS_DIR = HERE / "results_dynamic_scheduler"
        PLOTS_DIR, TABLES_DIR, MANIFESTS_DIR = (RESULTS_DIR / p for p in ("plots", "tables", "manifests"))
        LABELS.update(dynamic_constant="Constant mutation EA",
                      dynamic_exponential="Exponential mutation EA")
        COLORS.update(dynamic_constant="#0072B2", dynamic_exponential="#D55E00")
        # Use recorded probabilities, not a schedule reconstructed from possibly changed code.
        reference = None
        for seed in SEEDS:
            for condition in (*EA_CONDITIONS, "random_search"):
                directory = (DATA_DIR / RANDOM_SEARCH_FOLDER if condition == "random_search"
                             else DATA_DIR / "ea" / condition) / f"seed_{seed}"
                meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                if (meta["status"] != "complete" or meta["seed"] != seed
                    or meta["condition"] != condition or meta["pop_size"] != POP_SIZE
                    or meta["generations"] != NUM_GENERATIONS
                    or meta["actual_evaluations"] != (NUM_GENERATIONS + 1) * POP_SIZE):
                    raise ValueError(f"Incompatible run metadata: {directory}")
                if condition != "random_search":
                    schedule = (meta["exponential_probabilities"], meta["constant_mutation_probability"])
                    if reference is not None and schedule != reference:
                        raise ValueError(f"Different schedule settings: {directory}")
                    reference = schedule
        rates, constant = reference
        if len(rates) != NUM_GENERATIONS or not np.isclose(sum(rates), len(rates) * constant):
            raise ValueError("Invalid recorded mutation schedule")

    per_seed = {condition: {} for condition in (*EA_CONDITIONS, "random_search")}
    for seed in SEEDS:
        for condition in EA_CONDITIONS:
            per_seed[condition][seed] = ea_generation_stats(condition, seed)
        per_seed["random_search"][seed] = random_search_generation_stats(seed)

    for directory in (PLOTS_DIR, TABLES_DIR, MANIFESTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if args.experiment == "dynamic":
        with (RESULTS_DIR / "mutation_schedule.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["schedule_index", "offspring_generation",
                             "exponential_probability", "constant_probability"])
            writer.writerows((i, i + 1, rate, constant) for i, rate in enumerate(rates))
        final = {c: np.array([per_seed[c][seed]["best_so_far"][-1] for seed in SEEDS])
                 for c in per_seed}
        with (TABLES_DIR / "final_fitness_summary.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["condition", "runs", "mean", "std", "median", "min", "max"])
            for c, values in final.items():
                writer.writerow([c, len(values), values.mean(), values.std(),
                                 np.median(values), values.min(), values.max()])
        delta = final["dynamic_exponential"] - final["dynamic_constant"]
        with (TABLES_DIR / "paired_differences.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["seed", "constant", "exponential", "delta_exponential_minus_constant"])
            writer.writerows(zip(SEEDS, final["dynamic_constant"], final["dynamic_exponential"], delta))
        paired = {"mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
                  "std_delta": float(delta.std()), "exponential_wins": int(sum(delta < -1e-12)),
                  "constant_wins": int(sum(delta > 1e-12)),
                  "ties": int(sum(np.abs(delta) <= 1e-12)), "std_ddof": 0}
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            paired["test"] = "Not run: scipy unavailable"
        else:
            # Round only for the rank test to avoid floating-point pseudo-ties.
            rounded = np.round(delta, 12)
            if np.any(rounded):
                result = wilcoxon(rounded, alternative="two-sided", zero_method="wilcox", method="auto")
                paired.update(test="Wilcoxon signed-rank, two-sided, zero_method=wilcox, method=auto",
                              delta_rounding_decimals=12, statistic=float(result.statistic),
                              p_value=float(result.pvalue))
            else:
                paired["test"] = "Not run: all paired differences are zero"
        (TABLES_DIR / "paired_summary.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
        print(json.dumps(paired, indent=2))

    # --- per-seed, per-generation table (report-ready long format) --- #
    with (TABLES_DIR / "generation_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "seed", "generation", "cumulative_evals",
            "pop_mean_fitness", "pop_std_fitness", "best_so_far",
        ])
        for condition, by_seed in per_seed.items():
            for seed, stats in by_seed.items():
                rows = zip(
                    stats["cumulative_evals"], stats["pop_mean"],
                    stats["pop_std"], stats["best_so_far"], strict=True,
                )
                for gen, (evals, mean_, std_, best) in enumerate(rows):
                    writer.writerow([condition, seed, gen, evals, mean_, std_, best])

    # --- across-seed aggregation, per condition per generation --- #
    condition_summary: dict[str, dict[str, list[float]]] = {}
    for condition, by_seed in per_seed.items():
        n_gens = min(len(s["cumulative_evals"]) for s in by_seed.values())
        cumulative_evals = next(iter(by_seed.values()))["cumulative_evals"][:n_gens]
        mean_of_best, std_of_best, mean_of_pop_mean = [], [], []
        for gen in range(n_gens):
            bests = [by_seed[seed]["best_so_far"][gen] for seed in SEEDS]
            means = [by_seed[seed]["pop_mean"][gen] for seed in SEEDS]
            mean_of_best.append(float(np.mean(bests)))
            std_of_best.append(float(np.std(bests)))
            mean_of_pop_mean.append(float(np.mean(means)))
        condition_summary[condition] = {
            "cumulative_evals": cumulative_evals,
            "mean_best_so_far": mean_of_best,
            "std_best_so_far": std_of_best,
            "mean_pop_mean": mean_of_pop_mean,
        }

    with (TABLES_DIR / "condition_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "generation", "cumulative_evals",
            "mean_best_so_far", "std_best_so_far", "mean_pop_mean_fitness",
        ])
        for condition, stats in condition_summary.items():
            rows = zip(
                stats["cumulative_evals"], stats["mean_best_so_far"],
                stats["std_best_so_far"], stats["mean_pop_mean"], strict=True,
            )
            for gen, (evals, mbest, sbest, mmean) in enumerate(rows):
                writer.writerow([condition, gen, evals, mbest, sbest, mmean])

    # --- convergence speed: evals to reach within 5% of the best EA result --- #
    final_bests = [
        condition_summary[c]["mean_best_so_far"][-1]
        for c in EA_CONDITIONS
    ]
    threshold = min(final_bests) * 1.05  # lower fitness is better
    with (TABLES_DIR / "convergence_speed.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", f"evals_to_reach_{threshold:.3f}"])
        for condition, by_seed in per_seed.items():
            for seed, stats in by_seed.items():
                writer.writerow([condition, seed, evals_to_reach(stats, threshold)])

    # --- convergence plot --- #
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition in ("random_search", *EA_CONDITIONS):
        stats = condition_summary[condition]
        evals = np.array(stats["cumulative_evals"])
        mean_ = np.array(stats["mean_best_so_far"])
        std_ = np.array(stats["std_best_so_far"])
        color = COLORS[condition]
        ax.plot(evals, mean_, color=color, linewidth=2, label=LABELS[condition])
        ax.fill_between(
            evals, mean_ - std_, mean_ + std_, color=color, alpha=0.15, linewidth=0,
        )

    ax.set_xlabel("Cumulative fitness evaluations")
    ax.set_ylabel(f"Best-so-far fitness (mean ± std over {len(SEEDS)} seeds)\nlower is better")
    ax.set_title("Convergence: mutation schedules and random search" if args.experiment == "dynamic"
                 else "Convergence: mutation-only vs. mutation+crossover vs. random search")
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "convergence_plot.png", dpi=200)

    # --- the three additional plots --- #
    plot_final_fitness_violin(per_seed)
    per_target_distance_analysis(per_seed)
    plot_tree_size()

    print(f"wrote plots to {PLOTS_DIR}, tables to {TABLES_DIR}, manifests to {MANIFESTS_DIR}")


if __name__ == "__main__":
    main()
