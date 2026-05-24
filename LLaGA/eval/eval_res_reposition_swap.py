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
    raise ValueError(f"Unsupported dataset for reposition evaluation: {dataset}")


def compute_lp_accuracy(res_path, sample=-1):
    """LP yes/no accuracy. Mirrors eval_res.eval_lp: counts a sample as correct
    when the answer says "yes" and gt contains "yes", or the answer doesn't say
    "yes" and gt contains "no". gt is read from the JSONL — no per-dataset helper
    needed."""
    all_sample = 0
    correct = 0
    with open(res_path, "r") as f:
        for line in f:
            all_sample += 1
            res = json.loads(line)
            ans = res["text"].strip().lower()
            label = res["gt"].strip().lower()
            if ("yes" in ans and "yes" in label) or ("yes" not in ans and "no" in label):
                correct += 1
            if sample > 0 and all_sample >= sample:
                break
    acc = correct / all_sample if all_sample > 0 else 0.0
    return acc, all_sample, correct


def compute_accuracy(res_path, dataset, task, sample=-1):
    """Dispatch to NC or LP scoring based on task."""
    if task == "nc":
        return compute_nc_accuracy(res_path, dataset=dataset, sample=sample)
    if task == "lp":
        return compute_lp_accuracy(res_path, sample=sample)
    raise ValueError(f"Unsupported task for reposition evaluation: {task}")


def eval_reposition_swap(prefix, dataset, sample=-1, verbose=True, task="nc"):
    # matches files like {prefix}_reposition_swap_sink_nonsink_k{K}_run{R}.jsonl
    pattern = f"{prefix}_reposition_swap_sink_nonsink_k*_run*.jsonl"
    regex = re.compile(
        rf"^{re.escape(os.path.basename(prefix))}_reposition_swap_sink_nonsink_k(\d+)_run(\d+)\.jsonl$"
    )

    matched = []
    for path in glob.glob(pattern):
        m = regex.match(os.path.basename(path))
        if m is None:
            continue
        matched.append({
            "file": path,
            "filename": os.path.basename(path),
            "num_swap": int(m.group(1)),
            "run_idx": int(m.group(2)),
        })
    matched.sort(key=lambda x: (x["num_swap"], x["run_idx"]))

    if not matched:
        print(f"[{dataset}] No swap files found with prefix '{prefix}'.")
        return None

    by_k = {}
    for meta in matched:
        by_k.setdefault(meta["num_swap"], []).append(meta)

    all_stats = []
    for k, metas in sorted(by_k.items()):
        accs = []
        for meta in metas:
            acc, n, correct = compute_accuracy(meta["file"], dataset=dataset, task=task, sample=sample)
            meta["acc"] = acc
            meta["all_sample"] = n
            meta["correct"] = correct
            accs.append(acc)
            if verbose:
                print(f"[{dataset}] {meta['filename']} | k={k} | run_idx={meta['run_idx']} | acc={acc:.4f}")

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=0))
        print(
            f"\n[{dataset}] swap | num_swap={k} | n_runs={len(accs)} | "
            f"mean={mean_acc:.4f} | std={std_acc:.4f}\n"
        )
        all_stats.append({
            "dataset": dataset,
            "target": "swap",
            "num_swap": k,
            "n_runs": len(accs),
            "mean_acc": mean_acc,
            "std_acc": std_acc,
        })
    return all_stats


def eval_reposition_front(prefix, dataset, target, sample=-1, verbose=True, task="nc"):
    # target ∈ {"front_top2", "front_all"}
    # Match both `..._reposition_{target}.jsonl` and `..._reposition_{target}_run{N}.jsonl`
    pattern = f"{prefix}_reposition_{target}*.jsonl"
    regex = re.compile(
        rf"^{re.escape(os.path.basename(prefix))}_reposition_{re.escape(target)}(?:_run(\d+))?\.jsonl$"
    )

    matched = []
    for path in glob.glob(pattern):
        m = regex.match(os.path.basename(path))
        if m is None:
            continue
        run_str = m.group(1)
        matched.append({
            "file": path,
            "filename": os.path.basename(path),
            "run_idx": int(run_str) if run_str is not None else 0,
        })
    matched.sort(key=lambda x: x["run_idx"])

    if not matched:
        print(f"[{dataset}] No {target} files found with prefix '{prefix}'.")
        return None

    accs = []
    for meta in matched:
        acc, n, correct = compute_accuracy(meta["file"], dataset=dataset, task=task, sample=sample)
        meta["acc"] = acc
        meta["all_sample"] = n
        meta["correct"] = correct
        accs.append(acc)
        if verbose:
            print(f"[{dataset}] {meta['filename']} | run_idx={meta['run_idx']} | acc={acc:.4f}")

    if len(accs) == 1:
        print(f"\n[{dataset}] {target} | acc={accs[0]:.4f}\n")
        return [{
            "dataset": dataset,
            "target": target,
            "n_runs": 1,
            "mean_acc": float(accs[0]),
            "std_acc": 0.0,
        }]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs, ddof=0))
    print(
        f"\n[{dataset}] {target} | n_runs={len(accs)} | "
        f"mean={mean_acc:.4f} | std={std_acc:.4f}\n"
    )
    return [{
        "dataset": dataset,
        "target": target,
        "n_runs": len(accs),
        "mean_acc": mean_acc,
        "std_acc": std_acc,
    }]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Path prefix of prediction files, e.g. 'results_phc3mn/cora_nc_ND_predictions'",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["cora", "pubmed", "arxiv"])
    parser.add_argument(
        "--target",
        type=str,
        default="swap",
        choices=["swap", "front_top2", "front_all"],
        help="Which reposition mode to aggregate.",
    )
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--task", type=str, default="nc", choices=["nc", "lp"],
                        help="nc: scores against data.label_texts (per-dataset helpers). "
                             "lp: scores yes/no against the gt field in each prediction record. "
                             "File-matching glob is unchanged — pass --prefix pointing to your "
                             "LP base output (e.g. results_phc3mn/cora_lp_ND_predictions).")
    args = parser.parse_args()

    print(f"Prefix:  {args.prefix}")
    print(f"Dataset: {args.dataset}")
    print(f"Task:    {args.task}")
    print(f"Target:  {args.target}\n")

    if args.target == "swap":
        stats = eval_reposition_swap(
            prefix=args.prefix,
            dataset=args.dataset,
            sample=args.sample,
            verbose=not args.quiet,
            task=args.task,
        )
    else:
        stats = eval_reposition_front(
            prefix=args.prefix,
            dataset=args.dataset,
            target=args.target,
            sample=args.sample,
            verbose=not args.quiet,
            task=args.task,
        )

    if stats:
        print("=" * 80)
        print("Final summary")
        print("=" * 80)
        for s in stats:
            tag = (
                f"num_swap={s['num_swap']}, "
                if s["target"] == "swap"
                else f"target={s['target']}, "
            )
            print(
                f"{s['dataset']} | {tag}n_runs={s['n_runs']} | "
                f"mean={s['mean_acc']:.4f} | std={s['std_acc']:.4f}"
            )
