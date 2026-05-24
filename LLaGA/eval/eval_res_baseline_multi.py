import os
import glob
import json
import re
import argparse
import numpy as np
import torch


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
    raise ValueError(f"Unsupported dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Path prefix (without _seed{S}_run{R}.jsonl), "
             "e.g. 'results_phc3mn/cora_nc_ND_predictions'",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--seed", type=int, default=None,
                        help="If given, only files with this seed are used.")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    pattern = f"{args.prefix}_seed*_run*.jsonl"
    regex = re.compile(
        rf"^{re.escape(os.path.basename(args.prefix))}_seed(\d+)_run(\d+)\.jsonl$"
    )

    matched = []
    for path in glob.glob(pattern):
        m = regex.match(os.path.basename(path))
        if m is None:
            continue
        s, r = int(m.group(1)), int(m.group(2))
        if args.seed is not None and s != args.seed:
            continue
        matched.append({"file": path, "filename": os.path.basename(path), "seed": s, "run_idx": r})
    matched.sort(key=lambda x: (x["seed"], x["run_idx"]))

    if not matched:
        print(f"No baseline run files found with prefix '{args.prefix}'"
              f"{f' and seed={args.seed}' if args.seed is not None else ''}.")
        return

    by_seed = {}
    for meta in matched:
        by_seed.setdefault(meta["seed"], []).append(meta)

    final_summary = []
    for seed, metas in sorted(by_seed.items()):
        accs = []
        for meta in metas:
            acc, n, correct = compute_nc_accuracy(meta["file"], dataset=args.dataset, sample=args.sample)
            meta["acc"] = acc
            meta["all_sample"] = n
            meta["correct"] = correct
            accs.append(acc)
            if not args.quiet:
                print(f"[{args.dataset}] {meta['filename']} | seed={seed} | run={meta['run_idx']} | acc={acc:.4f}")

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=0))
        print(
            f"\n[{args.dataset}] seed={seed} | n_runs={len(accs)} | "
            f"mean={mean_acc:.4f} | std={std_acc:.4f}\n"
        )
        final_summary.append({
            "dataset": args.dataset,
            "seed": seed,
            "n_runs": len(accs),
            "mean_acc": mean_acc,
            "std_acc": std_acc,
        })

    print("=" * 80)
    print("Final summary")
    print("=" * 80)
    for s in final_summary:
        print(
            f"{s['dataset']} | seed={s['seed']} | n_runs={s['n_runs']} | "
            f"mean={s['mean_acc']:.4f} | std={s['std_acc']:.4f}"
        )


if __name__ == "__main__":
    main()
