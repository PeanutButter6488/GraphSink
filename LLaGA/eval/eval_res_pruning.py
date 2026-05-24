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


def compute_lp_accuracy(res_path, sample=-1):
    """LP yes/no accuracy. Mirrors eval_res.eval_lp: counts a sample as correct
    when the answer says "yes" and gt contains "yes", or the answer doesn't say
    "yes" and gt contains "no". Works for any dataset (gt comes from the JSONL,
    not from data.y), so no dataset-specific helper is needed."""
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


def compute_nc_accuracy(res_path, dataset, sample=-1):
    if dataset == "cora":
        return compute_cora_nc_accuracy(res_path, sample=sample)
    if dataset == "pubmed":
        return compute_pubmed_nc_accuracy(res_path, sample=sample)
    if dataset == "arxiv":
        return compute_arxiv_nc_accuracy(res_path, sample=sample)
    raise ValueError(f"Unsupported dataset for nc pruning evaluation: {dataset}")


def compute_accuracy(res_path, dataset, task, sample=-1):
    """Dispatch to NC or LP scoring based on task."""
    if task == "nc":
        return compute_nc_accuracy(res_path, dataset=dataset, sample=sample)
    if task == "lp":
        return compute_lp_accuracy(res_path, sample=sample)
    raise ValueError(f"Unsupported task for pruning evaluation: {task}")


def get_pruning_dir(result_dir, dataset, task="nc"):
    """LP scripts write to pruning_lp_{dataset}/, NC scripts to pruning_{dataset}/."""
    sub = f"pruning_{dataset}" if task == "nc" else f"pruning_{task}_{dataset}"
    base = os.path.basename(os.path.normpath(result_dir))
    if base == sub:
        return result_dir
    return os.path.join(result_dir, sub)


def extract_nonsink_metadata(path, dataset, task="nc", template="ND"):
    name = os.path.basename(path)
    match = re.match(
        rf"^{re.escape(dataset)}_{re.escape(task)}_{re.escape(template)}.*_prune_nonsinktoken_(\d+)_run(\d+)\.jsonl$",
        name,
    )
    if match is None:
        return None

    return {
        "file": path,
        "filename": name,
        "num_pruned": int(match.group(1)),
        "run_idx": int(match.group(2)),
    }


def find_nonsink_result_files(result_dir, dataset, num_pruned=None, task="nc", template="ND"):
    pruning_dir = get_pruning_dir(result_dir, dataset, task=task)
    pattern = os.path.join(pruning_dir, f"{dataset}_{task}_{template}*_prune_nonsinktoken_*_run*.jsonl")

    matched = []
    for path in glob.glob(pattern):
        meta = extract_nonsink_metadata(path, dataset, task=task, template=template)
        if meta is None:
            continue
        if num_pruned is not None and meta["num_pruned"] != num_pruned:
            continue
        matched.append((path, meta))

    matched.sort(key=lambda item: item[1]["run_idx"])
    return matched


def extract_sink_metadata(path, dataset, task="nc", template="ND"):
    name = os.path.basename(path)
    match = re.match(
        rf"^{re.escape(dataset)}_{re.escape(task)}_{re.escape(template)}.*_prune_sinktoken_(top2|all)_run(\d+)\.jsonl$",
        name,
    )
    if match is None:
        return None

    return {
        "file": path,
        "filename": name,
        "pruning_mode": match.group(1),
        "run_idx": int(match.group(2)),
    }


def find_sink_result_files(result_dir, dataset, pruning_mode=None, task="nc", template="ND"):
    pruning_dir = get_pruning_dir(result_dir, dataset, task=task)
    pattern = os.path.join(pruning_dir, f"{dataset}_{task}_{template}*_prune_sinktoken_*_run*.jsonl")

    matched = []
    for path in glob.glob(pattern):
        meta = extract_sink_metadata(path, dataset, task=task, template=template)
        if meta is None:
            continue
        if pruning_mode is not None and meta["pruning_mode"] != pruning_mode:
            continue
        matched.append((path, meta))

    matched.sort(key=lambda item: item[1]["run_idx"])
    return matched


def eval_sink_pruning(result_dir, dataset, pruning_mode="all", sample=-1, verbose=True,
                      task="nc", template="ND"):
    matched = find_sink_result_files(
        result_dir=result_dir,
        dataset=dataset,
        pruning_mode=pruning_mode,
        task=task,
        template=template,
    )

    if len(matched) == 0:
        print(f"No sink pruning files found for {dataset} (pruning_mode={pruning_mode}).")
        return {
            "dataset": dataset,
            "pruning_mode": pruning_mode,
            "num_files": 0,
            "num_skipped": 0,
            "mean_acc": None,
            "std_acc": None,
            "results": [],
        }

    results = []
    accs = []
    skipped = []
    for res_path, meta in matched:
        try:
            acc, all_sample, correct = compute_accuracy(res_path, dataset=dataset, task=task, sample=sample)
        except Exception as e:
            skipped.append((meta["filename"], str(e)))
            if verbose:
                print(f"[{dataset}] skipping {meta['filename']} | error={e}")
            continue

        meta["acc"] = acc
        meta["all_sample"] = all_sample
        meta["correct"] = correct
        results.append(meta)
        accs.append(acc)

        if verbose:
            print(
                f"[{dataset}] {meta['filename']} | "
                f"pruning_mode={meta['pruning_mode']} | "
                f"run_idx={meta['run_idx']} | "
                f"acc={acc:.4f}"
            )

    if len(results) == 0:
        print(f"All sink pruning files failed for {dataset} (pruning_mode={pruning_mode}).")
        return {
            "dataset": dataset,
            "pruning_mode": pruning_mode,
            "num_files": 0,
            "num_skipped": len(skipped),
            "mean_acc": None,
            "std_acc": None,
            "results": [],
        }

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs, ddof=0))

    print(f"\nSink pruning summary for {dataset}")
    print(f"Number of valid files: {len(results)}")
    print(f"Number of skipped files: {len(skipped)}")
    print(f"Pruning mode: {pruning_mode}")
    print(f"Mean accuracy: {mean_acc:.4f}")
    print(f"Std accuracy: {std_acc:.4f}")

    return {
        "dataset": dataset,
        "pruning_mode": pruning_mode,
        "num_files": len(results),
        "num_skipped": len(skipped),
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "results": results,
    }


def eval_nonsink_pruning(result_dir, dataset, num_pruned=1, sample=-1, verbose=True,
                         task="nc", template="ND"):
    matched = find_nonsink_result_files(
        result_dir=result_dir,
        dataset=dataset,
        num_pruned=num_pruned,
        task=task,
        template=template,
    )

    if len(matched) == 0:
        print(f"No nonsink pruning files found for {dataset} (num_pruned={num_pruned}).")
        return {
            "dataset": dataset,
            "num_pruned": num_pruned,
            "num_files": 0,
            "num_skipped": 0,
            "mean_acc": None,
            "std_acc": None,
            "results": [],
        }

    results = []
    accs = []
    skipped = []
    for res_path, meta in matched:
        try:
            acc, all_sample, correct = compute_accuracy(res_path, dataset=dataset, task=task, sample=sample)
        except Exception as e:
            skipped.append((meta["filename"], str(e)))
            if verbose:
                print(f"[{dataset}] skipping {meta['filename']} | error={e}")
            continue

        meta["acc"] = acc
        meta["all_sample"] = all_sample
        meta["correct"] = correct
        results.append(meta)
        accs.append(acc)

        if verbose:
            print(
                f"[{dataset}] {meta['filename']} | "
                f"num_pruned={meta['num_pruned']} | "
                f"run_idx={meta['run_idx']} | "
                f"acc={acc:.4f}"
            )

    if len(results) == 0:
        print(f"All nonsink pruning files failed for {dataset} (num_pruned={num_pruned}).")
        return {
            "dataset": dataset,
            "num_pruned": num_pruned,
            "num_files": 0,
            "num_skipped": len(skipped),
            "mean_acc": None,
            "std_acc": None,
            "results": [],
        }

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs, ddof=0))

    print(f"\nNonsink pruning summary for {dataset}")
    print(f"Number of valid files: {len(results)}")
    print(f"Number of skipped files: {len(skipped)}")
    print(f"Num pruned: {num_pruned}")
    print(f"Mean accuracy: {mean_acc:.4f}")
    print(f"Std accuracy: {std_acc:.4f}")

    return {
        "dataset": dataset,
        "num_pruned": num_pruned,
        "num_files": len(results),
        "num_skipped": len(skipped),
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results_phc3mn")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["cora", "pubmed", "arxiv", "all"])
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--target", type=str, default="nonsink", choices=["nonsink", "sink"],
                        help="nonsink: aggregate _prune_nonsinktoken_* files. "
                             "sink: aggregate _prune_sinktoken_* files.")
    parser.add_argument("--num_pruned", type=int, default=1,
                        help="Filter for nonsink target.")
    parser.add_argument("--pruning_mode", type=str, default="all", choices=["top2", "all"],
                        help="Filter for sink target.")
    parser.add_argument("--task", type=str, default="nc", choices=["nc", "lp"],
                        help="nc: scores against data.label_texts (per-dataset helpers). "
                             "lp: scores yes/no against the gt field in each prediction record. "
                             "Also selects the result subdir (pruning_{dataset} vs pruning_lp_{dataset}) "
                             "and the {dataset}_{task}_{template}* filename pattern.")
    parser.add_argument("--template", type=str, default="ND",
                        help="LLaGA encoding template, used in the result filename pattern "
                             "{dataset}_{task}_{template}*_prune_*. Default ND matches the NC scripts.")
    args = parser.parse_args()

    verbose = not args.quiet
    datasets = ["cora", "pubmed", "arxiv"] if args.dataset == "all" else [args.dataset]

    print(f"Scanning result directory: {args.result_dir}")
    print(f"Task: {args.task} | Template: {args.template}")
    print(f"Target: {args.target}")
    if args.target == "nonsink":
        print(f"Num pruned: {args.num_pruned}")
    else:
        print(f"Pruning mode: {args.pruning_mode}")
    print(f"Datasets: {', ' .join(datasets)}\n")

    all_stats = []
    for i, dataset in enumerate(datasets):
        if args.target == "nonsink":
            stats = eval_nonsink_pruning(
                result_dir=args.result_dir,
                dataset=dataset,
                num_pruned=args.num_pruned,
                sample=args.sample,
                verbose=verbose,
                task=args.task,
                template=args.template,
            )
        else:
            stats = eval_sink_pruning(
                result_dir=args.result_dir,
                dataset=dataset,
                pruning_mode=args.pruning_mode,
                sample=args.sample,
                verbose=verbose,
                task=args.task,
                template=args.template,
            )
        all_stats.append(stats)
        if i != len(datasets) - 1:
            print("\n" + "=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("Final summary")
    print("=" * 80)
    for stats in all_stats:
        if stats["num_files"] > 0:
            print(
                f"{stats['dataset']}: mean acc {stats['mean_acc']:.4f}, "
                f"std {stats['std_acc']:.4f}, "
                f"files {stats['num_files']}, "
                f"skipped {stats['num_skipped']}"
            )
        else:
            print(f"{stats['dataset']}: no valid files found")
