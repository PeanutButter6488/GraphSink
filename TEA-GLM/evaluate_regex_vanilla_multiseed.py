"""
Aggregate evaluation for vanilla (no-pruning) multi-(seed,run) sweeps.

By default globs every `{prefix}_seed*_run*_model_results.txt` under
./results/{test_dataset}/ that is NOT a pruning output, computes the same
regex-accuracy + macro-P/R/F1 metrics as evaluate_regex.py for each
(seed, run) pair, and reports per-(seed,run) numbers plus mean ± std across
all rows.

Pass `--seed S` to filter to a single seed (e.g. only the seed=0 sweep or
only the seed=123 sweep); without it, every seed found on disk is aggregated
into one table.

Usage:
    python evaluate_regex_vanilla_multiseed.py <prefix> <test_dataset> [--seed S]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from statistics import mean, stdev

from evaluate_regex_random import compute_metrics, compute_lp_metrics  # reuse metric code


def _fmt(values):
    if len(values) == 1:
        return f"{values[0]:.2f} (n=1)"
    return f"{mean(values):.2f} ± {stdev(values):.2f} (n={len(values)})"


def aggregate(prefix: str, test_dataset: str, seed: int | None = None, task: str = "nc") -> None:
    # LP runs land under ./results/{test_dataset}_lp/ (out_ds_tag patch in
    # train_glm.py); NC keeps the bare {test_dataset}/ path.
    ds_dir = test_dataset if task == "nc" else f"{test_dataset}_{task}"
    metric_fn = compute_lp_metrics if task == "lp" else compute_metrics
    base = f"./results/{ds_dir}/{prefix}"
    if seed is not None:
        pattern = f"{base}_seed{seed}_run*_model_results.txt"
    else:
        pattern = f"{base}_seed*_run*_model_results.txt"
    candidates = sorted(glob.glob(pattern))
    # Belt-and-suspenders: exclude any pruning files even if they happen to
    # carry `_seed*_run*` in their suffix (e.g. `_prune_all_seed{S}_run{N}`).
    pred_files = [p for p in candidates if "_prune_" not in os.path.basename(p)]
    if not pred_files:
        print(f"No vanilla result files matched: {pattern}")
        return

    rows = []
    for pred_file in pred_files:
        m = re.search(r"_seed(\d+)_run(\d+)_model_results\.txt$", pred_file)
        if m is None:
            continue
        s = int(m.group(1))
        r = int(m.group(2))
        label_file = pred_file.replace("_model_results.txt", "_model_labels.txt")
        metrics = metric_fn(pred_file, label_file)
        if metrics is None:
            continue
        rows.append((s, r, metrics))

    if not rows:
        print("No valid result files to aggregate.")
        return

    rows.sort(key=lambda r: (r[0], r[1]))
    header_extra = f", seed={seed}" if seed is not None else ""
    print(
        f"\n--- Per-(seed,run) vanilla results "
        f"(prefix={prefix}, test_dataset={test_dataset}{header_extra}) ---"
    )
    print(f"{'seed':>6}  {'run':>4}  {'acc':>8}  {'prec':>8}  {'rec':>8}  {'f1':>8}  {'n':>8}")
    for s, r, metrics in rows:
        print(
            f"{s:>6}  "
            f"{r:>4}  "
            f"{metrics['regex_accuracy']:>7.2f}%  "
            f"{metrics['macro_precision']:>7.2f}%  "
            f"{metrics['macro_recall']:>7.2f}%  "
            f"{metrics['macro_f1']:>7.2f}%  "
            f"{metrics['num_samples']:>8d}"
        )

    accs = [m["regex_accuracy"] for _, _, m in rows]
    precs = [m["macro_precision"] for _, _, m in rows]
    recs = [m["macro_recall"] for _, _, m in rows]
    f1s = [m["macro_f1"] for _, _, m in rows]

    n_seeds = len({s for s, _, _ in rows})
    print(f"\n--- Aggregate across {len(rows)} (seed,run) pairs ({n_seeds} seed(s)) ---")
    print(f"Regex Accuracy : {_fmt(accs)}")
    print(f"Macro Precision: {_fmt(precs)}")
    print(f"Macro Recall   : {_fmt(recs)}")
    print(f"Macro F1       : {_fmt(f1s)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate vanilla multi-(seed,run) evaluation results."
    )
    parser.add_argument("prefix", help="Run prefix used in result filenames.")
    parser.add_argument("test_dataset", help="Test dataset directory under ./results/.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="If set, aggregate only files whose name contains _seed{S}_run{N}; "
             "otherwise aggregate every seed found on disk.",
    )
    parser.add_argument(
        "--task", type=str, default="nc", choices=["nc", "lp"],
        help="nc: free-class regex/macro-F1 over labels; reads "
             "./results/{test_dataset}/. lp: yes/no scoring via lp_pred_to_yn; "
             "reads ./results/{test_dataset}_lp/.",
    )
    args = parser.parse_args()
    aggregate(args.prefix, args.test_dataset, args.seed, task=args.task)
