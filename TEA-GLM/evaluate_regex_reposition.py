"""
Aggregator for the graph-token shuffling (reposition) experiment.

For each graph-token-index i in [0, K-1]:
    change_ratio[i] = (# samples where i was involved in the swap AND the
                       reposition prediction differs from the baseline prediction)
                      / (# samples where i was involved in the swap)

Also reports overall accuracy (vs labels) for baseline and reposition, plus
the global change ratio (fraction of samples whose prediction flipped).

Inputs:
    ./results/{test_dataset}/{prefix}_model_results.txt                    (baseline preds, unseeded)
        or the seeded variant {prefix}_seed{seed}_model_results.txt        (preferred if available)
    ./results/{test_dataset}/{prefix}_reposition_swap_k{K}_seed{S}_model_results.txt
    ./results/{test_dataset}/{prefix}_reposition_swap_k{K}_seed{S}_model_labels.txt
    ./analysis/{test_dataset}/global_stats/
        {prefix}_reposition_swap_k{K}_seed{S}_reposition_records.jsonl

Usage:
    python evaluate_regex_reposition.py <prefix> <test_dataset> <num_swap> <reposition_seed> [baseline_seed]
    # baseline_seed optional: if given, pair against {prefix}_seed{baseline_seed}_model_results.txt
"""
import glob
import json
import os
import re
import sys
from statistics import mean, stdev

from evaluate_regex_random import compute_metrics, compute_lp_metrics, lp_pred_to_yn  # reuse metric code


def _load_json_list(path):
    with open(path, "r") as f:
        return json.load(f)


def _load_records(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (int(rec["step"]), int(rec["batch_index"]))
            out[key] = rec
    return out


def _extract_class(pred_text, all_classes):
    clean = str(pred_text).strip().lower()
    for cls in all_classes:
        if re.search(re.escape(cls), clean):
            return cls
    return None


def _extract_yn(pred_text):
    """Reuse the shared LP normalizer so per-row baseline-vs-reposition
    comparisons agree with the metric reported by compute_lp_metrics. Returns
    None for unknown phrasings so they don't collide with a real class.
    """
    cls = lp_pred_to_yn(pred_text)
    return cls if cls in ("yes", "no") else None


def compute_reposition_stats(
    prefix, test_dataset, num_swap, reposition_seed,
    baseline_seed=None, num_graph_tokens=20, task="nc",
):
    """Compute reposition stats for a single seed. Returns a dict, or None on
    error. Silent (no printing/plotting) so it can be reused by the multi-seed
    aggregator. The returned dict has:
        base_pred_path, rep_pred_path, rep_label_path, rep_suffix,
        base_metrics, rep_metrics,
        global_applied, global_changed, pair_count,
        per_index_involved, per_index_changed, per_index_ratio,
        num_graph_tokens
    """
    # LP runs land under the _lp-suffixed dirs (out_ds_tag patch in train_glm.py).
    ds_dir = test_dataset if task == "nc" else f"{test_dataset}_{task}"
    results_dir = f"./results/{ds_dir}"
    analysis_dir = f"./analysis/{ds_dir}/global_stats"

    rep_suffix = f"_reposition_swap_k{num_swap}_seed{reposition_seed}"
    rep_pred_path = os.path.join(results_dir, f"{prefix}{rep_suffix}_model_results.txt")
    rep_label_path = os.path.join(results_dir, f"{prefix}{rep_suffix}_model_labels.txt")
    rep_records_path = os.path.join(analysis_dir, f"{prefix}{rep_suffix}_reposition_records.jsonl")

    if baseline_seed is not None:
        base_pred_path = os.path.join(results_dir, f"{prefix}_seed{baseline_seed}_model_results.txt")
    else:
        base_pred_path = os.path.join(results_dir, f"{prefix}_model_results.txt")

    for p in (base_pred_path, rep_pred_path, rep_label_path, rep_records_path):
        if not os.path.exists(p):
            print(f"Missing: {p}")
            return None

    baseline_preds = _load_json_list(base_pred_path)
    rep_preds = _load_json_list(rep_pred_path)
    labels = _load_json_list(rep_label_path)
    rep_records = _load_records(rep_records_path)

    if not (len(baseline_preds) == len(rep_preds) == len(labels)):
        print(
            f"Length mismatch: baseline={len(baseline_preds)} rep={len(rep_preds)} "
            f"labels={len(labels)}"
        )
        return None

    n = len(labels)
    y_true = [str(l).strip().lower() for l in labels]
    all_classes = sorted(set(y_true))
    extract_pred = (lambda p, _cls=None: _extract_yn(p)) if task == "lp" else (lambda p, cls=all_classes: _extract_class(p, cls))
    metric_fn = compute_lp_metrics if task == "lp" else compute_metrics

    per_index_involved = [0] * num_graph_tokens
    per_index_changed = [0] * num_graph_tokens
    global_changed = 0
    global_applied = 0

    sorted_keys = sorted(rep_records.keys())
    if len(sorted_keys) != n:
        print(
            f"Warning: {len(sorted_keys)} reposition records but {n} prediction rows. "
            f"Falling back to positional match on min(len)."
        )
    pair_count = min(len(sorted_keys), n)

    for i in range(pair_count):
        b_pred = extract_pred(baseline_preds[i])
        r_pred = extract_pred(rep_preds[i])
        rec = rep_records[sorted_keys[i]]
        applied = bool(rec.get("reposition_applied", False))
        if not applied:
            continue
        global_applied += 1
        changed = b_pred != r_pred
        if changed:
            global_changed += 1
        for k_idx in rec.get("graph_token_indices", []):
            k_idx = int(k_idx)
            if 0 <= k_idx < num_graph_tokens:
                per_index_involved[k_idx] += 1
                if changed:
                    per_index_changed[k_idx] += 1

    per_index_ratio = [
        (per_index_changed[i] / per_index_involved[i]) if per_index_involved[i] > 0 else 0.0
        for i in range(num_graph_tokens)
    ]

    base_metrics = metric_fn(base_pred_path, rep_label_path)
    rep_metrics = metric_fn(rep_pred_path, rep_label_path)

    return {
        "base_pred_path": base_pred_path,
        "rep_pred_path": rep_pred_path,
        "rep_label_path": rep_label_path,
        "rep_suffix": rep_suffix,
        "base_metrics": base_metrics,
        "rep_metrics": rep_metrics,
        "global_applied": global_applied,
        "global_changed": global_changed,
        "pair_count": pair_count,
        "per_index_involved": per_index_involved,
        "per_index_changed": per_index_changed,
        "per_index_ratio": per_index_ratio,
        "num_graph_tokens": num_graph_tokens,
        "analysis_dir": analysis_dir,
    }


def aggregate(prefix, test_dataset, num_swap, reposition_seed, baseline_seed=None, num_graph_tokens=20, task="nc"):
    stats = compute_reposition_stats(
        prefix, test_dataset, num_swap, reposition_seed,
        baseline_seed=baseline_seed, num_graph_tokens=num_graph_tokens, task=task,
    )
    if stats is None:
        return
    base_metrics = stats["base_metrics"]
    rep_metrics = stats["rep_metrics"]
    global_applied = stats["global_applied"]
    global_changed = stats["global_changed"]
    pair_count = stats["pair_count"]
    per_index_involved = stats["per_index_involved"]
    per_index_changed = stats["per_index_changed"]
    per_index_ratio = stats["per_index_ratio"]
    rep_suffix = stats["rep_suffix"]
    analysis_dir = stats["analysis_dir"]

    print(
        f"\n--- Reposition change analysis "
        f"(prefix={prefix}, test_dataset={test_dataset}, num_swap={num_swap}, "
        f"reposition_seed={reposition_seed}, baseline_seed={baseline_seed}) ---"
    )
    if base_metrics and rep_metrics:
        print(f"Baseline acc:    {base_metrics['regex_accuracy']:.2f}%   F1: {base_metrics['macro_f1']:.2f}%")
        print(f"Reposition acc:  {rep_metrics['regex_accuracy']:.2f}%   F1: {rep_metrics['macro_f1']:.2f}%")
    if global_applied > 0:
        print(
            f"Samples with swap applied: {global_applied}/{pair_count} "
            f"(global change ratio = {100.0 * global_changed / global_applied:.2f}%)"
        )

    print("\nPer-graph-token-index change ratio (index, involved, changed, ratio):")
    for i in range(num_graph_tokens):
        print(
            f"  idx={i:>2}  involved={per_index_involved[i]:>6}  "
            f"changed={per_index_changed[i]:>6}  "
            f"ratio={100.0 * per_index_ratio[i]:>6.2f}%"
        )

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    save_dir = os.path.join(analysis_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"{prefix}{rep_suffix}_change_ratio_per_index.png"
    )
    fig, ax = plt.subplots(figsize=(12, 4), dpi=180)
    xs = list(range(num_graph_tokens))
    ax.bar(xs, [100.0 * r for r in per_index_ratio], width=0.75, color="tab:blue")
    ax.set_title(
        f"{test_dataset}: prediction-change ratio per graph-token index\n"
        f"(swap_sink_nonsink k={num_swap}, seed={reposition_seed})"
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
    import argparse
    parser = argparse.ArgumentParser(
        description="Aggregate reposition (graph-token swap) eval results.",
    )
    parser.add_argument("prefix")
    parser.add_argument("test_dataset")
    parser.add_argument("num_swap", type=int)
    parser.add_argument("reposition_seed", type=int)
    parser.add_argument("baseline_seed", type=int, nargs="?", default=None)
    parser.add_argument(
        "--task", type=str, default="nc", choices=["nc", "lp"],
        help="nc: NC scoring on ./results/{test_dataset}/. "
             "lp: yes/no scoring on ./results/{test_dataset}_lp/ + "
             "./analysis/{test_dataset}_lp/global_stats/.",
    )
    args = parser.parse_args()
    aggregate(
        args.prefix, args.test_dataset, args.num_swap, args.reposition_seed,
        baseline_seed=args.baseline_seed, task=args.task,
    )
