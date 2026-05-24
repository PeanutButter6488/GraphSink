"""
Aggregate evaluation for random-pruning sweeps.

By default globs every `{prefix}_prune_random{num_prune}_seed*_model_results.txt`
under ./results/{test_dataset}/, computes the same regex-accuracy + macro-P/R/F1
metrics as evaluate_regex.py for each seed, and reports per-seed numbers plus
mean ± std across seeds.

Pass `--seed S` to restrict aggregation to a single seed (n=1).

Usage:
    python evaluate_regex_random.py <prefix> <test_dataset> [num_prune] [--seed S]
"""
import argparse
import glob
import json
import os
import re
import sys
from statistics import mean, stdev
from typing import Dict, Optional

from sklearn.metrics import f1_score, precision_score, recall_score


def _load_pair(pred_file: str, label_file: str):
    try:
        with open(pred_file, "r") as f:
            predictions_raw = json.load(f)
        with open(label_file, "r") as f:
            labels_raw = json.load(f)
    except FileNotFoundError as e:
        print(f"  skip ({e})")
        return None
    if len(predictions_raw) != len(labels_raw):
        print(
            f"  skip (predictions={len(predictions_raw)} != labels={len(labels_raw)})"
        )
        return None
    return predictions_raw, labels_raw


# Negative-then-positive phrase patterns. Order matters: every negative phrase
# contains the positive phrase as a substring, so negatives must be checked
# first. Lowercased; whitespace-normalized via str.strip().
_LP_NEGATIVE_PHRASES = (
    "may not have",
    "do not have",
    "does not have",
    "not have a citation",
    "not have citation",
    "no citation",
)
_LP_POSITIVE_PHRASES = (
    "have citation",
    "have a citation",
    "share a citation",
)


# Word-boundary regexes for the literal yes/no fallback. Naive substring
# matching mis-fires (e.g. 'no' is a substring of 'unknown', 'know', 'not',
# 'nothing'), so the fallback uses \b boundaries to require a standalone token.
_LP_YES_RE = re.compile(r"\byes\b")
_LP_NO_RE = re.compile(r"\bno\b")


def lp_pred_to_yn(pred: str) -> str:
    """Map a free-form LP prediction string to {'yes', 'no', 'none'}.

    TEA-GLM emits long sentences like "These two papers may not have citation
    relationships." for negatives and "These two papers have citation
    relationships." for positives. Neither contains a literal 'yes' / 'no', so
    the LLaGA-style substring rule misclassifies all of them.

    Resolution order (first match wins):
      1. Explicit negative phrases (must precede the positive phrase, since
         "may not have citation" is a superstring of "have citation").
      2. Explicit positive phrases.
      3. Standalone 'yes' / 'no' token (keeps the LLaGA-style "Yes." / "No."
         outputs working without false-positives on words like 'unknown').
    """
    s = str(pred).strip().lower()
    for phrase in _LP_NEGATIVE_PHRASES:
        if phrase in s:
            return "no"
    for phrase in _LP_POSITIVE_PHRASES:
        if phrase in s:
            return "yes"
    if _LP_YES_RE.search(s):
        return "yes"
    if _LP_NO_RE.search(s):
        return "no"
    return "none"


def compute_lp_metrics(pred_file: str, label_file: str) -> Optional[Dict[str, float]]:
    """Yes/no scoring for link-prediction outputs. Returns the same dict shape
    as compute_metrics so the per-row print/aggregation loops in the prune /
    random / reposition aggregators don't need to know about the task.

    Predictions are mapped to {yes, no, none} via lp_pred_to_yn (see above).
    Accuracy is the fraction whose mapped class equals the (mapped) ground
    truth; macro_precision / recall / f1 are computed over {yes, no}.
    """
    pair = _load_pair(pred_file, label_file)
    if pair is None:
        return None
    predictions_raw, labels_raw = pair

    y_true = [lp_pred_to_yn(l) for l in labels_raw]
    y_pred = [lp_pred_to_yn(p) for p in predictions_raw]
    classes = ["yes", "no"]

    total = len(y_true)
    correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
    return {
        "regex_accuracy": (correct / total) * 100.0 if total else 0.0,
        "macro_precision": precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ) * 100.0,
        "macro_recall": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ) * 100.0,
        "macro_f1": f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ) * 100.0,
        "num_samples": total,
    }


def compute_metrics(pred_file: str, label_file: str) -> Optional[Dict[str, float]]:
    pair = _load_pair(pred_file, label_file)
    if pair is None:
        return None
    predictions_raw, labels_raw = pair

    y_true = [str(l).strip().lower() for l in labels_raw]
    all_classes = sorted(set(y_true))

    # First-match class extraction from the free-form prediction string.
    y_pred = []
    for pred_text in predictions_raw:
        clean_pred = str(pred_text).strip().lower()
        found_class = None
        for class_name in all_classes:
            if re.search(re.escape(class_name), clean_pred):
                found_class = class_name
                break
        y_pred.append(found_class if found_class else "None")

    total = len(y_true)
    regex_correct = sum(
        1
        for pred_text, true_label in zip(predictions_raw, y_true)
        if re.search(re.escape(true_label), str(pred_text).strip().lower())
    )

    return {
        "regex_accuracy": (regex_correct / total) * 100.0,
        "macro_precision": precision_score(
            y_true, y_pred, labels=all_classes, average="macro", zero_division=0
        )
        * 100.0,
        "macro_recall": recall_score(
            y_true, y_pred, labels=all_classes, average="macro", zero_division=0
        )
        * 100.0,
        "macro_f1": f1_score(
            y_true, y_pred, labels=all_classes, average="macro", zero_division=0
        )
        * 100.0,
        "num_samples": total,
    }


def _fmt(values):
    if len(values) == 1:
        return f"{values[0]:.2f} (n=1)"
    return f"{mean(values):.2f} ± {stdev(values):.2f} (n={len(values)})"


def aggregate(
    prefix: str,
    test_dataset: str,
    num_prune: int = 2,
    seed: Optional[int] = None,
    task: str = "nc",
) -> None:
    ds_dir = test_dataset if task == "nc" else f"{test_dataset}_{task}"
    metric_fn = compute_lp_metrics if task == "lp" else compute_metrics
    base = f"./results/{ds_dir}/{prefix}_prune_random{num_prune}"
    if seed is not None:
        pattern = f"{base}_seed{seed}_model_results.txt"
    else:
        pattern = f"{base}_seed*_model_results.txt"
    pred_files = sorted(glob.glob(pattern))
    if not pred_files:
        print(f"No result files matched: {pattern}")
        return

    rows = []  # list of (seed, metrics)
    for pred_file in pred_files:
        m = re.search(r"_seed(\d+)_model_results\.txt$", pred_file)
        if m is None:
            print(f"  skip {pred_file} (could not parse seed)")
            continue
        s = int(m.group(1))
        label_file = pred_file.replace("_model_results.txt", "_model_labels.txt")
        metrics = metric_fn(pred_file, label_file)
        if metrics is None:
            continue
        rows.append((s, metrics))

    if not rows:
        print("No valid result files to aggregate.")
        return

    rows.sort(key=lambda x: x[0])
    header = (
        f"(prefix={prefix}, test_dataset={test_dataset}, num_prune={num_prune}"
    )
    if seed is not None:
        header += f", seed={seed}"
    header += ")"
    print(f"\n--- Per-seed results {header} ---")
    print(f"{'seed':>6}  {'acc':>8}  {'prec':>8}  {'rec':>8}  {'f1':>8}  {'n':>8}")
    for s, metrics in rows:
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

    print(f"\n--- Aggregate across {len(rows)} seeds ---")
    print(f"Regex Accuracy : {_fmt(accs)}")
    print(f"Macro Precision: {_fmt(precs)}")
    print(f"Macro Recall   : {_fmt(recs)}")
    print(f"Macro F1       : {_fmt(f1s)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate random-pruning evaluation results.",
    )
    parser.add_argument("prefix", help="Run prefix used in result filenames.")
    parser.add_argument("test_dataset", help="Test dataset directory under ./results/.")
    parser.add_argument(
        "num_prune", type=int, nargs="?", default=2,
        help="Number of non-sink tokens pruned per sample. (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="If set, restrict aggregation to the single _seed{S} file (n=1); "
             "otherwise aggregate every seed found on disk.",
    )
    parser.add_argument(
        "--task", type=str, default="nc", choices=["nc", "lp"],
        help="nc: score with regex/macro-F1 over class names from labels; results "
             "live in ./results/{test_dataset}/. lp: score yes/no via "
             "compute_lp_metrics; results live in ./results/{test_dataset}_lp/.",
    )
    args = parser.parse_args()
    aggregate(args.prefix, args.test_dataset, args.num_prune, args.seed, task=args.task)
