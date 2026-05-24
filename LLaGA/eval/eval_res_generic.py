import argparse
import json


def normalize(text):
    return " ".join(str(text).strip().lower().split())


def load_records(res_path, sample=-1):
    records = []
    with open(res_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if sample > 0 and len(records) >= sample:
                break
    return records


def build_label_space(records):
    labels = []
    seen = set()
    for record in records:
        label = record.get("gt", "")
        norm = normalize(label)
        if not norm or norm in seen:
            continue
        labels.append(label.strip())
        seen.add(norm)
    return labels


def eval_generic_nc(records):
    if not records:
        raise ValueError("No records found in result file.")

    labels = build_label_space(records)
    normalized_labels = [normalize(label) for label in labels]

    strict_correct = 0
    overall_correct = 0
    for record in records:
        answer = normalize(record.get("text", ""))
        gt = normalize(record.get("gt", ""))

        if answer == gt:
            strict_correct += 1
            overall_correct += 1
            continue

        matched_labels = [label for label in normalized_labels if label and label in answer]
        if gt and gt in answer and len(matched_labels) == 1:
            overall_correct += 1

    total = len(records)
    print(f"Test samples: {total}")
    print(f"Strict accuracy: {strict_correct / total:.4f}")
    print(f"Overall accuracy: {overall_correct / total:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="nc", choices=["nc"])
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    records = load_records(args.res_path, sample=args.sample)
    eval_generic_nc(records)


if __name__ == "__main__":
    main()
