import argparse
import json
import os

import numpy as np


def normalize(text):
    return " ".join(str(text).strip().lower().split())


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True,
                        help="prediction jsonl (question_id, text, gt, and echoed sink fields)")
    parser.add_argument("--sink_path", required=True,
                        help="sink jsonl from analysis/{dataset}_{template}/sink_records.jsonl")
    parser.add_argument("--num_graph_tokens", type=int, default=111)
    parser.add_argument("--which", choices=["top2", "all"], default="top2")
    parser.add_argument("--save_path", required=True,
                        help="output prefix; writes <prefix>.json and <prefix>.png")
    args = parser.parse_args()

    preds = load_jsonl(args.pred_path)
    labels_norm = {normalize(r.get("gt", "")) for r in preds}
    labels_norm.discard("")

    # correctness: strict match OR (gt appears in answer AND it's the only label that does)
    correct_ids, wrong_ids = set(), set()
    for r in preds:
        ans = normalize(r.get("text", ""))
        gt = normalize(r.get("gt", ""))
        qid = r["question_id"]
        if ans == gt:
            correct_ids.add(qid)
            continue
        matched = [lab for lab in labels_norm if lab and lab in ans]
        if gt and gt in ans and len(matched) == 1:
            correct_ids.add(qid)
        else:
            wrong_ids.add(qid)

    key = "top2_sink_token_indices" if args.which == "top2" else "all_sink_indices"
    sink_by_id = {r["question_id"]: r.get(key, []) for r in load_jsonl(args.sink_path)}

    K = args.num_graph_tokens
    cnt_c = np.zeros(K, dtype=np.int64)
    cnt_w = np.zeros(K, dtype=np.int64)
    n_c = n_w = 0
    for qid, idxs in sink_by_id.items():
        if qid in correct_ids:
            bucket, n = cnt_c, "c"
            n_c += 1
        elif qid in wrong_ids:
            bucket, n = cnt_w, "w"
            n_w += 1
        else:
            continue
        for i in set(int(x) for x in idxs):
            if 0 <= i < K:
                bucket[i] += 1

    if n_c == 0 or n_w == 0:
        raise SystemExit(f"no samples in one bucket (n_correct={n_c}, n_wrong={n_w}) — "
                         f"check that pred_path and sink_path share question_ids")
    if cnt_c.sum() == 0 or cnt_w.sum() == 0:
        raise SystemExit(f"sink lists are empty for one bucket — check --sink_path "
                         f"({args.sink_path}) actually contains non-empty '{key}'")

    ratio_c = cnt_c / n_c
    ratio_w = cnt_w / n_w
    delta = ratio_w - ratio_c

    from scipy.stats import spearmanr, kendalltau
    sp = spearmanr(ratio_c, ratio_w)
    kt = kendalltau(ratio_c, ratio_w)

    top10_wc = [int(i) for i in np.argsort(-delta)[:10]]

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    summary = {
        "pred_path": args.pred_path,
        "sink_path": args.sink_path,
        "which": args.which,
        "num_graph_tokens": K,
        "n_correct": n_c,
        "n_wrong": n_w,
        "accuracy": n_c / max(n_c + n_w, 1),
        "spearman": {"statistic": float(sp.statistic), "pvalue": float(sp.pvalue)},
        "kendall": {"statistic": float(kt.statistic), "pvalue": float(kt.pvalue)},
        "ratio_correct": ratio_c.tolist(),
        "ratio_wrong": ratio_w.tolist(),
        "delta_wrong_minus_correct": delta.tolist(),
        "top10_wrong_over_correct": top10_wc,
        "top10_delta": [float(delta[i]) for i in top10_wc],
    }
    json_path = args.save_path + ".json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    import matplotlib.pyplot as plt
    x = np.arange(K)
    w = 0.4
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 6), dpi=180, sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )
    ax_top.bar(x - w / 2, ratio_c, width=w, label=f"correct (n={n_c})", color="#1f77b4")
    ax_top.bar(x + w / 2, ratio_w, width=w, label=f"wrong (n={n_w})", color="#d62728")
    ax_top.set_ylabel("sink ratio across samples")
    ax_top.set_title(
        f"sink-position frequency: correct vs wrong ({args.which})  "
        f"spearman={sp.statistic:.3f} (p={sp.pvalue:.2g})  "
        f"kendall={kt.statistic:.3f}"
    )
    ax_top.legend()
    ax_top.grid(True, axis="y", alpha=0.25)

    colors = ["#d62728" if d >= 0 else "#1f77b4" for d in delta]
    ax_bot.bar(x, delta, width=0.9, color=colors)
    ax_bot.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax_bot.set_xlabel("graph token index")
    ax_bot.set_ylabel(r"$\Delta$ (wrong - correct)")
    ax_bot.set_xticks(list(range(0, K, 5)))
    ax_bot.grid(True, axis="y", alpha=0.25)

    png_path = args.save_path + ".png"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"n_correct={n_c}  n_wrong={n_w}  accuracy={summary['accuracy']:.4f}")
    print(f"spearman={sp.statistic:.4f} (p={sp.pvalue:.3g})  "
          f"kendall={kt.statistic:.4f} (p={kt.pvalue:.3g})")
    print(f"top-10 positions where wrong > correct: {top10_wc}")
    print(f"saved: {json_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
