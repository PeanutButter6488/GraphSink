"""
Multi-seed aggregator for the reposition (graph-token swap) experiment.

Calls compute_reposition_stats() per seed (silently), then reports mean ± std
across seeds for:
  - reposition accuracy / macro-F1
  - global change ratio
  - per-graph-token-index change ratio

Also produces a per-index bar plot with std error bars.

The baseline metrics do not vary across reposition seeds (same baseline file),
so they are reported once.

Usage:
    python evaluate_regex_reposition_multiseed.py <prefix> <test_dataset> <num_swap> \
        [--seeds 42,43,44,45,46] [--start_seed 42 --n_seeds 5] \
        [--baseline_seed S] [--num_graph_tokens N]
"""
from __future__ import annotations

import argparse
import os
import sys
from statistics import mean, stdev

from evaluate_regex_reposition import compute_reposition_stats


def _fmt_pct(values):
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{100.0 * values[0]:.2f} (n=1)"
    return f"{100.0 * mean(values):.2f} ± {100.0 * stdev(values):.2f} (n={len(values)})"


def _fmt_metric(values):
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.2f} (n=1)"
    return f"{mean(values):.2f} ± {stdev(values):.2f} (n={len(values)})"


def _parse_seeds(args) -> list[int]:
    if args.seeds:
        return [int(s) for s in args.seeds.split(",") if s.strip()]
    return list(range(args.start_seed, args.start_seed + args.n_seeds))


def aggregate_multiseed(
    prefix: str,
    test_dataset: str,
    num_swap: int,
    seeds: list[int],
    baseline_seed: int | None = None,
    num_graph_tokens: int = 20,
    task: str = "nc",
) -> None:
    per_seed_stats = []
    for seed in seeds:
        stats = compute_reposition_stats(
            prefix, test_dataset, num_swap, seed,
            baseline_seed=baseline_seed, num_graph_tokens=num_graph_tokens,
            task=task,
        )
        if stats is None:
            print(f"Skipping seed={seed} (missing files or length mismatch).")
            continue
        per_seed_stats.append((seed, stats))

    if not per_seed_stats:
        print("No seeds produced valid stats — nothing to aggregate.")
        return

    # Baseline is the same across seeds (same baseline file), so just take the
    # first run's numbers.
    first_seed, first_stats = per_seed_stats[0]
    base_metrics = first_stats["base_metrics"]
    analysis_dir = first_stats["analysis_dir"]

    rep_acc = [s["rep_metrics"]["regex_accuracy"] for _, s in per_seed_stats if s["rep_metrics"]]
    rep_f1 = [s["rep_metrics"]["macro_f1"] for _, s in per_seed_stats if s["rep_metrics"]]
    global_change_ratios = [
        (s["global_changed"] / s["global_applied"]) if s["global_applied"] > 0 else 0.0
        for _, s in per_seed_stats
    ]

    # Per-index ratios stacked: shape (num_graph_tokens, n_seeds).
    per_index_matrix = [
        [s["per_index_ratio"][i] for _, s in per_seed_stats]
        for i in range(num_graph_tokens)
    ]

    print(
        f"\n--- Reposition multi-seed analysis "
        f"(prefix={prefix}, test_dataset={test_dataset}, num_swap={num_swap}, "
        f"seeds={seeds}, baseline_seed={baseline_seed}) ---"
    )
    if base_metrics:
        print(
            f"Baseline (seed={first_seed} as reference, identical across reposition seeds):"
        )
        print(f"  acc: {base_metrics['regex_accuracy']:.2f}%   F1: {base_metrics['macro_f1']:.2f}%")
    print(f"Reposition acc: {_fmt_metric(rep_acc)}%")
    print(f"Reposition F1:  {_fmt_metric(rep_f1)}%")
    print(f"Global change ratio: {_fmt_pct(global_change_ratios)}%")

    print("\nPer-graph-token-index change ratio across seeds (mean ± std, %):")
    for i in range(num_graph_tokens):
        vals = per_index_matrix[i]
        # Skip indices that were never involved across any seed.
        if all(s["per_index_involved"][i] == 0 for _, s in per_seed_stats):
            continue
        print(f"  idx={i:>2}  {_fmt_pct(vals)}%")

    # Plot mean per-index change ratio with std error bars.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    means_pct = [100.0 * mean(per_index_matrix[i]) for i in range(num_graph_tokens)]
    stds_pct = [
        100.0 * stdev(per_index_matrix[i]) if len(per_index_matrix[i]) > 1 else 0.0
        for i in range(num_graph_tokens)
    ]

    seed_tag = f"seeds{seeds[0]}-{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) \
        else "seeds" + "_".join(str(s) for s in seeds)
    save_dir = analysis_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"{prefix}_reposition_swap_k{num_swap}_{seed_tag}_change_ratio_per_index_meanstd.png",
    )

    fig, ax = plt.subplots(figsize=(12, 4), dpi=180)
    xs = list(range(num_graph_tokens))
    ax.bar(xs, means_pct, yerr=stds_pct, width=0.75, color="tab:blue",
           ecolor="black", capsize=3)
    ax.set_title(
        f"{test_dataset}: prediction-change ratio per graph-token index\n"
        f"(swap_sink_nonsink k={num_swap}, n_seeds={len(seeds)}, mean ± std)"
    )
    ax.set_xlabel("Graph token index (original K-space)")
    ax.set_ylabel("Change ratio (%) among samples with this index swapped")
    if num_graph_tokens <= 30:
        ax.set_xticks(xs)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-seed aggregator for the reposition experiment."
    )
    parser.add_argument("prefix", help="Run prefix used in result filenames.")
    parser.add_argument("test_dataset", help="Test dataset directory under ./results/.")
    parser.add_argument("num_swap", type=int, help="Per-sample swap count (K).")
    parser.add_argument(
        "--seeds", default=None,
        help="Comma-separated seed list, e.g. '42,43,44,45,46'. Overrides --start_seed/--n_seeds.",
    )
    parser.add_argument("--start_seed", type=int, default=42)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument(
        "--baseline_seed", type=int, default=None,
        help="If set, pair against {prefix}_seed{baseline_seed}_model_results.txt; "
             "otherwise pair against {prefix}_model_results.txt.",
    )
    parser.add_argument("--num_graph_tokens", type=int, default=20)
    parser.add_argument(
        "--task", type=str, default="nc", choices=["nc", "lp"],
        help="nc: NC-style scoring on ./results/{test_dataset}/. "
             "lp: yes/no scoring on ./results/{test_dataset}_lp/ + "
             "./analysis/{test_dataset}_lp/global_stats/ (threaded through "
             "to compute_reposition_stats).",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args)
    if not seeds:
        print("No seeds specified.")
        sys.exit(1)

    aggregate_multiseed(
        args.prefix,
        args.test_dataset,
        args.num_swap,
        seeds,
        baseline_seed=args.baseline_seed,
        num_graph_tokens=args.num_graph_tokens,
        task=args.task,
    )
