"""
Single-run evaluation for TEA-GLM inference outputs.

NC (default): regex-accuracy + macro-P/R/F1 over the class names that appear in
the labels file. Reads ./results/{test_dataset}/{prefix}_model_{results,labels}.txt.

LP (--task lp): yes/no scoring via lp_pred_to_yn() (handles TEA-GLM's
free-form "These two papers (may not) have citation relationships." outputs in
addition to LLaGA-style "Yes." / "No."). Reads
./results/{test_dataset}_lp/{prefix}_model_{results,labels}.txt.

Usage:
    python evaluate_regex.py <prefix> <test_dataset> [--task {nc,lp}]
"""
import argparse
import json
import re
from sklearn.metrics import f1_score, precision_score, recall_score

from evaluate_regex_random import lp_pred_to_yn


def evaluate_metrics_nc(pred_file, label_file):
    """
    NC scoring (unchanged): "regex accuracy" treats a prediction as correct if
    the true label string appears as a substring of the prediction; macro
    metrics bucket each prediction to the first known class string it contains.
    """
    try:
        with open(pred_file, 'r') as f:
            predictions_raw = json.load(f)
        with open(label_file, 'r') as f:
            labels_raw = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure your paths are correct.")
        print("Pred File Path:", pred_file)
        print("Label File Path:", label_file)
        return

    if len(predictions_raw) != len(labels_raw):
        print(f"Error: Mismatch in number of samples!")
        print(f"Predictions: {len(predictions_raw)}, Labels: {len(labels_raw)}")
        return

    # --- Data Cleaning and Preparation ---

    # 1. Get clean list of true labels
    y_true = [str(l).strip().lower() for l in labels_raw]

    # 2. Get a sorted list of all unique possible classes
    all_classes = sorted(list(set(y_true)))

    # 3. Extract predicted class for each prediction
    y_pred = []

    for pred_text in predictions_raw:
        clean_pred = str(pred_text).strip().lower()
        found_class = None

        # Search for any known class in the prediction string
        # We search in order to find a match.
        # Note: This is a simple "first-match" extraction.
        for class_name in all_classes:
            # Use re.escape to handle special chars like 'cs.ai'
            if re.search(re.escape(class_name), clean_pred):
                found_class = class_name
                break  # Stop at the first class we find

        # If no known class is found in the output, mark as 'None'
        y_pred.append(found_class if found_class else "None")


    # --- Metric Calculation ---

    total = len(y_true)

    # 1. Calculate "Regex Accuracy" (Your original metric)
    regex_correct = 0
    print("\n--- First 10 Mismatches (Prediction | Label) ---")
    mismatch_count = 0

    for pred_text, true_label in zip(predictions_raw, y_true):
        clean_pred = str(pred_text).strip().lower()

        if re.search(re.escape(true_label), clean_pred):
            regex_correct += 1
            print("clean pred", clean_pred, 'next one: \n')
        else:
            if mismatch_count < 10:
                # Show the *raw* prediction and the *true* label for mismatch analysis
                #print(f'"{str(pred_text).strip()}"  |  "{true_label}"')
                mismatch_count += 1

    if mismatch_count == 0:
        print("No mismatches found!")

    regex_accuracy = (regex_correct / total) * 100

    # 2. Calculate true multi-class metrics
    # We tell the metrics functions to only use the labels we know
    # and to handle divisions by zero gracefully.
    macro_precision = precision_score(
        y_true, y_pred, labels=all_classes, average='macro', zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, labels=all_classes, average='macro', zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, labels=all_classes, average='macro', zero_division=0
    )

    # --- Report Results ---
    print(f"\n--- Evaluation Results ---")
    print(f"Total Samples: {total}")
    print(f"Known Classes: {len(all_classes)}")
    print("----------------------------")
    print(f"Regex Accuracy: {regex_accuracy:.2f}%")
    print("----------------------------")
    print(f"Macro Precision: {macro_precision * 100:.2f}%")
    print(f"Macro Recall: {macro_recall * 100:.2f}%")
    print(f"Macro F1-Score: {macro_f1 * 100:.2f}%")
    print("----------------------------\n")


def evaluate_metrics_lp(pred_file, label_file):
    """LP scoring: predictions and labels are mapped to {yes, no, none} by
    lp_pred_to_yn so TEA-GLM's free-form citation phrasings score correctly.
    """
    try:
        with open(pred_file, 'r') as f:
            predictions_raw = json.load(f)
        with open(label_file, 'r') as f:
            labels_raw = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure your paths are correct.")
        print("Pred File Path:", pred_file)
        print("Label File Path:", label_file)
        return

    if len(predictions_raw) != len(labels_raw):
        print(f"Error: Mismatch in number of samples!")
        print(f"Predictions: {len(predictions_raw)}, Labels: {len(labels_raw)}")
        return

    y_true = [lp_pred_to_yn(l) for l in labels_raw]
    y_pred = [lp_pred_to_yn(p) for p in predictions_raw]
    classes = ["yes", "no"]
    total = len(y_true)
    correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
    accuracy = (correct / total) * 100 if total else 0.0

    macro_precision = precision_score(
        y_true, y_pred, labels=classes, average='macro', zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, labels=classes, average='macro', zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, labels=classes, average='macro', zero_division=0
    )

    # Show a small sample of mapped (pred -> yn, label -> yn) pairs for sanity.
    print("\n--- First 10 (mapped pred | mapped label | raw pred) ---")
    for i in range(min(10, total)):
        print(f"  {y_pred[i]:>4}  |  {y_true[i]:>4}  |  {str(predictions_raw[i]).strip()[:80]!r}")

    # Highlight any predictions that the normalizer couldn't classify, so
    # untracked phrasings surface instead of silently being marked wrong.
    none_count = sum(1 for c in y_pred if c == "none")
    if none_count:
        print(f"\nWARNING: {none_count} prediction(s) mapped to 'none' (unknown phrasing).")

    print(f"\n--- Evaluation Results (LP) ---")
    print(f"Total Samples: {total}")
    print("----------------------------")
    print(f"Yes/No Accuracy: {accuracy:.2f}%")
    print("----------------------------")
    print(f"Macro Precision: {macro_precision * 100:.2f}%")
    print(f"Macro Recall: {macro_recall * 100:.2f}%")
    print(f"Macro F1-Score: {macro_f1 * 100:.2f}%")
    print("----------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a single TEA-GLM inference run (NC or LP)."
    )
    parser.add_argument("prefix", help="Run prefix used in result filenames.")
    parser.add_argument("test_dataset", help="Bare dataset name (e.g. 'arxiv'); "
                        "the LP path automatically appends '_lp'.")
    parser.add_argument(
        "--task", type=str, default="nc", choices=["nc", "lp"],
        help="nc: free-class regex/macro-F1 over labels; reads "
             "./results/{test_dataset}/. lp: yes/no scoring via lp_pred_to_yn; "
             "reads ./results/{test_dataset}_lp/.",
    )
    args = parser.parse_args()

    ds_dir = args.test_dataset if args.task == "nc" else f"{args.test_dataset}_{args.task}"
    pred_file_path = f'./results/{ds_dir}/{args.prefix}_model_results.txt'
    label_file_path = f'./results/{ds_dir}/{args.prefix}_model_labels.txt'

    if args.task == "lp":
        evaluate_metrics_lp(pred_file_path, label_file_path)
    else:
        evaluate_metrics_nc(pred_file_path, label_file_path)
