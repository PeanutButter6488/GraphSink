"""
Aggregate evaluation for any sink-pruning mode (top2 | all | random).

Index convention differs by mode (set by the per-mode test scripts):

  - top2 / all : files are `{prefix}_prune_<mode>_seed{S}_run{N}_model_results.txt`
                 (test_citation_sinkanalysis.sh fixes --seed and varies the
                 run number to capture sampling variance; also re-runnable at
                 multiple seeds).
  - random     : files are `{prefix}_prune_random{K}_seed{S}_model_results.txt`
                 (test_citation_sinkanalysis_random.sh varies --seed to pick
                 different non-sink prune subsets per run; no separate run
                 dimension).

For each mode this aggregator globs the matching files, computes the same
regex-accuracy + macro-P/R/F1 metrics as evaluate_regex.py per index, and
reports per-index numbers plus mean ± std across them.

Pass `--seed S` to restrict to a single seed:
  - top2 / all : aggregates across all runs at seed S.
  - random     : aggregates the single seed S file (n=1).

Usage:
    python evaluate_regex_prune.py <prefix> <test_dataset> <mode> [num_prune] [--seed S]
    # mode ∈ {top2, all, random}; num_prune only used when mode=random (default 2).
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


def _prune_part_for(mode: str, num_prune: int) -> str:
    if mode == "top2":
        return "_prune_top2"
    if mode == "all":
        return "_prune_all"
    if mode == "random":
        return f"_prune_random{num_prune}"
    raise ValueError(f"Unknown mode: {mode!r}. Expected top2 | all | random.")


def _pattern_for(
    mode: str, prefix: str, test_dataset: str, num_prune: int, seed: int | None,
    task: str = "nc",
) -> str:
    # LP runs land under ./results/{test_dataset}_lp/ (see train_glm.py
    # out_ds_tag patch); NC keeps the bare {test_dataset}/ path.
    ds_dir = test_dataset if task == "nc" else f"{test_dataset}_{task}"
    base = f"./results/{ds_dir}/{prefix}{_prune_part_for(mode, num_prune)}"
    if mode in ("top2", "all"):
        # Per-(seed,run) sampling reruns from test_citation_sinkanalysis.sh.
        if seed is not None:
            return f"{base}_seed{seed}_run*_model_results.txt"
        return f"{base}_seed*_run*_model_results.txt"
    if mode == "random":
        # Per-seed prune-subset sweep from test_citation_sinkanalysis_random.sh
        # (no run dimension).
        if seed is not None:
            return f"{base}_seed{seed}_model_results.txt"
        return f"{base}_seed*_model_results.txt"
    raise ValueError(f"Unknown mode: {mode!r}. Expected top2 | all | random.")


def aggregate(
    prefix: str,
    test_dataset: str,
    mode: str,
    num_prune: int = 2,
    seed: int | None = None,
    task: str = "nc",
) -> None:
    pattern = _pattern_for(mode, prefix, test_dataset, num_prune, seed, task=task)
    metric_fn = compute_lp_metrics if task == "lp" else compute_metrics
    pred_files = sorted(glob.glob(pattern))
    if not pred_files:
        print(f"No result files matched: {pattern}")
        return

    has_run_index = mode in ("top2", "all")
    if has_run_index:
        index_re = re.compile(r"_seed(\d+)_run(\d+)_model_results\.txt$")
    else:
        index_re = re.compile(r"_seed(\d+)_model_results\.txt$")

    rows = []
    for pred_file in pred_files:
        m = index_re.search(pred_file)
        if m is None:
            continue
        if has_run_index:
            key = (int(m.group(1)), int(m.group(2)))
        else:
            key = (int(m.group(1)),)
        label_file = pred_file.replace("_model_results.txt", "_model_labels.txt")
        metrics = metric_fn(pred_file, label_file)
        if metrics is None:
            continue
        rows.append((key, metrics))

    if not rows:
        print("No valid result files to aggregate.")
        return

    rows.sort(key=lambda r: r[0])
    header = f"(prefix={prefix}, test_dataset={test_dataset}, mode={mode}"
    if mode == "random":
        header += f", num_prune={num_prune}"
    if seed is not None:
        header += f", seed={seed}"
    header += ")"

    if has_run_index:
        print(f"\n--- Per-(seed,run) prune results {header} ---")
        print(f"{'seed':>6}  {'run':>4}  {'acc':>8}  {'prec':>8}  {'rec':>8}  {'f1':>8}  {'n':>8}")
        for (s, r), metrics in rows:
            print(
                f"{s:>6}  "
                f"{r:>4}  "
                f"{metrics['regex_accuracy']:>7.2f}%  "
                f"{metrics['macro_precision']:>7.2f}%  "
                f"{metrics['macro_recall']:>7.2f}%  "
                f"{metrics['macro_f1']:>7.2f}%  "
                f"{metrics['num_samples']:>8d}"
            )
    else:
        print(f"\n--- Per-seed prune results {header} ---")
        print(f"{'seed':>6}  {'acc':>8}  {'prec':>8}  {'rec':>8}  {'f1':>8}  {'n':>8}")
        for (s,), metrics in rows:
            print(
                f"{s:>6}  "
                f"{metrics['regex_accuracy']:>7.2f}%  "
                f"{metrics['macro_precision']:>7.2f}%  "
                f"{metrics['macro_recall']:>7.2f}%  "
                f"{metrics['macro_f1']:>7.2f}%  "
                f"{metrics['num_samples']:>8d}"
            )

    accs = [m["regex_accuracy"] for _, m in rows]
    precs = [m["macro_precision"] for _, m in rows]
    recs = [m["macro_recall"] for _, m in rows]
    f1s = [m["macro_f1"] for _, m in rows]

    if has_run_index:
        n_seeds = len({key[0] for key, _ in rows})
        print(f"\n--- Aggregate across {len(rows)} (seed,run) pairs ({n_seeds} seed(s)) ---")
    else:
        print(f"\n--- Aggregate across {len(rows)} seeds ---")
    print(f"Regex Accuracy : {_fmt(accs)}")
    print(f"Macro Precision: {_fmt(precs)}")
    print(f"Macro Recall   : {_fmt(recs)}")
    print(f"Macro F1       : {_fmt(f1s)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate sink-pruning evaluation results.",
    )
    parser.add_argument("prefix", help="Run prefix used in result filenames.")
    parser.add_argument("test_dataset", help="Test dataset directory under ./results/.")
    parser.add_argument("mode", choices=["top2", "all", "random"])
    parser.add_argument(
        "num_prune", type=int, nargs="?", default=2,
        help="Number of non-sink tokens pruned per sample. Only used when mode=random. "
             "(default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="If set, restrict aggregation to files at this seed. For top2/all this "
             "selects the runs of one --seed sweep; for random this selects the single "
             "seed-S file.",
    )
    parser.add_argument(
        "--task", type=str, default="nc", choices=["nc", "lp"],
        help="nc: score with regex/macro-F1 over class names from labels; results "
             "live in ./results/{test_dataset}/. lp: score yes/no via "
             "compute_lp_metrics; results live in ./results/{test_dataset}_lp/.",
    )
    args = parser.parse_args()
    aggregate(
        args.prefix,
        args.test_dataset,
        args.mode,
        args.num_prune,
        args.seed,
        task=args.task,
    )
