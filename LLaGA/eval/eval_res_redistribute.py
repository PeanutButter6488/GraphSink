import os
import glob
import json
import re
import argparse
import torch
import numpy as np


def compute_cora_nc_accuracy(res_path, sample=-1):
    data = torch.load("dataset/cora/processed_data.pt", weights_only=False)
    labels = data.label_texts
    short_labels = [l.split('_')[0] for l in labels]
    ys = data.y.numpy().tolist()

    all_sample = 0
    correct = 0
    with open(res_path, "r") as f:
        for line in f:
            all_sample += 1
            res = json.loads(line)
            ans = res["text"]
            y = ys[res["question_id"]]
            short_label = short_labels[y]

            if (
                short_label.strip().lower() in ans.strip().lower()
                and sum(l.strip().lower() in ans.strip().lower() for l in short_labels) == 1
            ):
                correct += 1

            if sample > 0 and all_sample >= sample:
                break

    acc = correct / all_sample if all_sample > 0 else 0.0
    return acc, all_sample, correct


def compute_pubmed_nc_accuracy(res_path, sample=-1):
    data = torch.load("dataset/pubmed/processed_data.pt", weights_only=False)
    labels = data.label_texts
    short_labels = [l[18:] for l in labels]
    ys = data.y.numpy().tolist()

    all_sample = 0
    correct = 0
    with open(res_path, "r") as f:
        for line in f:
            all_sample += 1
            res = json.loads(line)
            ans = res["text"]
            y = ys[res["question_id"]]
            short_label = short_labels[y]
            label = labels[y]

            if ans.lower().strip() == label.lower().strip():
                correct += 1
            elif (
                short_label.lower().strip() in ans.lower().strip()
                and sum(la.lower().strip() in ans.lower().strip() for la in short_labels) == 1
            ):
                correct += 1

            if sample > 0 and all_sample >= sample:
                break

    acc = correct / all_sample if all_sample > 0 else 0.0
    return acc, all_sample, correct


def compute_arxiv_nc_accuracy(res_path, sample=-1):
    data = torch.load("dataset/arxiv/processed_data.pt", weights_only=False)
    labels = data.label_texts
    short_labels = [l[0:5] for l in labels]
    ys = data.y.numpy().tolist()

    all_sample = 0
    correct = 0
    with open(res_path, "r") as f:
        for line in f:
            all_sample += 1
            res = json.loads(line)
            ans = res["text"]
            y = ys[res["question_id"]]
            short_label = short_labels[y]
            label = labels[y]
            if label.lower().strip() == ans.lower().strip():
                correct += 1
            elif (
                short_label.lower() in ans.lower()
                and sum(la.lower() in ans.lower() for la in short_labels) == 1
            ):
                correct += 1
            if sample > 0 and all_sample >= sample:
                break
    acc = correct / all_sample if all_sample > 0 else 0.0
    return acc, all_sample, correct


def compute_nc_accuracy(res_path, dataset, sample=-1):
    if dataset == "cora":
        return compute_cora_nc_accuracy(res_path, sample=sample)
    if dataset == "pubmed":
        return compute_pubmed_nc_accuracy(res_path, sample=sample)
    if dataset == "arxiv":
        return compute_arxiv_nc_accuracy(res_path, sample=sample)
    raise ValueError(f"Unsupported dataset for redistribute evaluation: {dataset}")


def eval_redistribute(prefix, dataset, sample=-1, verbose=True):
    pattern = f"{prefix}_redistribute_*_run*.jsonl"
    base = re.escape(os.path.basename(prefix))
    regex_src = re.compile(
        rf"^{base}_redistribute_src_to_sinks_pct([0-9.]+)_src(\d+)_run(\d+)\.jsonl$"
    )
    regex_sinks = re.compile(
        rf"^{base}_redistribute_(sinks_to_top_nonsink|sinks_to_nonsink_even|sinks_to_nonsink_value_sim)_pct([0-9.]+)_run(\d+)\.jsonl$"
    )

    matched = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        m = regex_src.match(name)
        if m is not None:
            matched.append({
                "file": path,
                "filename": name,
                "direction": "src_to_sinks",
                "fraction": float(m.group(1)),
                "source_idx": int(m.group(2)),
                "run_idx": int(m.group(3)),
            })
            continue
        m = regex_sinks.match(name)
        if m is not None:
            matched.append({
                "file": path,
                "filename": name,
                "direction": m.group(1),
                "fraction": float(m.group(2)),
                "source_idx": None,
                "run_idx": int(m.group(3)),
            })

    matched.sort(key=lambda x: (x["direction"], x["fraction"], x["source_idx"] or -1, x["run_idx"]))

    if len(matched) == 0:
        print(f"No redistribute files found with prefix '{prefix}'.")
        return None

    by_key = {}
    for meta in matched:
        by_key.setdefault((meta["direction"], meta["fraction"], meta["source_idx"]), []).append(meta)

    all_stats = []
    for (direction, frac, src), metas in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or -1)):
        accs = []
        for meta in metas:
            acc, n, correct = compute_nc_accuracy(meta["file"], dataset=dataset, sample=sample)
            meta["acc"] = acc
            meta["all_sample"] = n
            meta["correct"] = correct
            accs.append(acc)
            if verbose:
                src_str = f"src={src}" if src is not None else "src=-"
                print(
                    f"[{dataset}] {meta['filename']} | dir={direction} | pct={frac} | {src_str} | "
                    f"run_idx={meta['run_idx']} | acc={acc:.4f}"
                )

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=0))
        src_str = f"source_idx={src}" if src is not None else "source_idx=-"
        print(
            f"\n[{dataset}] direction={direction} | fraction={frac} | {src_str} | "
            f"n_runs={len(accs)} | mean={mean_acc:.4f} | std={std_acc:.4f}\n"
        )
        all_stats.append({
            "dataset": dataset,
            "direction": direction,
            "fraction": frac,
            "source_idx": src,
            "n_runs": len(accs),
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "results": metas,
        })

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Path prefix of prediction files, e.g. 'results_phc3mn/cora_nc_ND_predictions'",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Prefix: {args.prefix}")
    print(f"Dataset: {args.dataset}\n")

    stats = eval_redistribute(
        prefix=args.prefix,
        dataset=args.dataset,
        sample=args.sample,
        verbose=not args.quiet,
    )

    if stats is not None:
        print("=" * 80)
        print("Final summary")
        print("=" * 80)
        for s in stats:
            src_str = f"src={s['source_idx']}" if s['source_idx'] is not None else "src=-"
            print(
                f"{s['dataset']} | dir={s['direction']} | pct={s['fraction']} | {src_str} | "
                f"n_runs={s['n_runs']} | mean={s['mean_acc']:.4f} | std={s['std_acc']:.4f}"
            )
