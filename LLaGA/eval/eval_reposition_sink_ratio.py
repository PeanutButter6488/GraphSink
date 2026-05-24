import json
import os
import argparse

import numpy as np


def load_sink_indices(path, key="top2_sink_token_indices"):
    by_id = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_id[r["question_id"]] = r.get(key, [])
    return by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--template", default="ND")
    parser.add_argument(
        "--reposition_tag",
        required=True,
        help="e.g. front_top2 | front_all | swap_sink_nonsink_k2_seed0",
    )
    parser.add_argument("--num_graph_tokens", type=int, default=111)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    base = f"analysis/{args.dataset}_{args.template}"
    before_path = f"{base}/sink_records.jsonl"
    after_path = f"{base}/reposition_{args.reposition_tag}_sink_records.jsonl"

    if not os.path.exists(before_path):
        raise FileNotFoundError(before_path)
    if not os.path.exists(after_path):
        raise FileNotFoundError(after_path)

    before = load_sink_indices(before_path)
    after = load_sink_indices(after_path)

    shared = sorted(set(before.keys()) & set(after.keys()))
    n = len(shared)
    K = args.num_graph_tokens
    cnt_before = np.zeros(K, dtype=np.int64)
    cnt_after = np.zeros(K, dtype=np.int64)
    for qid in shared:
        for i in set(before[qid]):
            if 0 <= int(i) < K:
                cnt_before[int(i)] += 1
        for i in set(after[qid]):
            if 0 <= int(i) < K:
                cnt_after[int(i)] += 1

    ratio_before = cnt_before / max(n, 1)
    ratio_after = cnt_after / max(n, 1)

    save_path = args.save_path or f"{base}/reposition_{args.reposition_tag}_sink_ratio.png"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    import matplotlib.pyplot as plt

    delta = ratio_after - ratio_before
    x = np.arange(K)
    w = 0.4

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 6), dpi=180, sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    ax_top.bar(x - w / 2, ratio_before, width=w, label=f"Before (n={n})", color="#1f77b4")
    ax_top.bar(x + w / 2, ratio_after, width=w, label=f"After: {args.reposition_tag}", color="#d62728")
    ax_top.set_ylabel("Sink ratio across samples")
    ax_top.set_title(f"{args.dataset} {args.template}: sink ratio per graph-token index (top-2 sinks)")
    ax_top.legend()
    ax_top.grid(True, axis="y", alpha=0.25)

    delta_colors = ["#2ca02c" if d >= 0 else "#9467bd" for d in delta]
    ax_bot.bar(x, delta, width=0.9, color=delta_colors)
    ax_bot.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax_bot.set_xlabel("Graph token index")
    ax_bot.set_ylabel(r"$\Delta$ ratio (after - before)")
    ax_bot.set_xticks(list(range(0, K, 5)))
    ax_bot.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = save_path.rsplit(".", 1)[0] + ".json"
    summary = {
        "dataset": args.dataset,
        "template": args.template,
        "reposition_tag": args.reposition_tag,
        "num_samples": n,
        "num_graph_tokens": K,
        "count_before": cnt_before.tolist(),
        "count_after": cnt_after.tolist(),
        "ratio_before": ratio_before.tolist(),
        "ratio_after": ratio_after.tolist(),
        "top10_before": [int(i) for i in np.argsort(-ratio_before)[:10]],
        "top10_after": [int(i) for i in np.argsort(-ratio_after)[:10]],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Samples (paired by question_id): {n}")
    print(f"Top-10 indices (before): {summary['top10_before']}")
    print(f"Top-10 indices (after):  {summary['top10_after']}")
    print(f"Saved plot:    {save_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
