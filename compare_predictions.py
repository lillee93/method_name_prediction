import argparse
import json

from utils import load_prediction_records


def main():
    parser = argparse.ArgumentParser(
        description="Find methods that were previously predicted wrong but now predicted correctly."
    )
    parser.add_argument(
        "--old-predictions",
        required=True,
        help="JSONL predictions file from the first run (before).",
    )
    parser.add_argument(
        "--new-predictions",
        required=True,
        help="JSONL predictions file from the second run (after).",
    )
    parser.add_argument(
        "--output-file",
        default="improved_methods.jsonl",
        help="Where to save improved cases as JSONL (default: improved_methods.jsonl).",
    )
    args = parser.parse_args()

    old_records = load_prediction_records(args.old_predictions)
    new_records = load_prediction_records(args.new_predictions)

    if len(old_records) != len(new_records):
        print(
            f"Warning: different number of records "
            f"(old={len(old_records)}, new={len(new_records)}). "
            f"Will only compare up to the shorter length."
        )

    n = min(len(old_records), len(new_records))
    improved = []

    for i in range(n):
        old_rec = old_records[i]
        new_rec = new_records[i]

        old_correct = bool(old_rec.get("is_correct", False))
        new_correct = bool(new_rec.get("is_correct", False))

        # we only care about cases that flipped from wrong -> correct
        if (not old_correct) and new_correct:
            improved.append({
                "true_name": new_rec.get("true_name"),
                "old_pred": old_rec.get("pred_name"),
                "new_pred": new_rec.get("pred_name"),
            })

    print(f"Found {len(improved)} methods that were previously wrong and are now correct.\n")

    # Print a few examples to the terminal
    for ex in improved[:100]:
        print("=" * 80)
        print("True name:", ex["true_name"])
        print("Old prediction:", ex["old_pred"])
        print("New prediction:", ex["new_pred"])

    # Save all improved cases to file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for ex in improved:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved all improved cases to {args.output_file}")


if __name__ == "__main__":
    main()