import argparse
import json

from utils import split_camel_and_underscore, name_similarity


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prediction results and error patterns."
    )
    parser.add_argument(
        "--predictions-file",
        default="predictions.jsonl",
        help="JSONL file containing predictions (default: predictions.jsonl).",
    )
    args = parser.parse_args()

    # Load prediction records
    results = []
    with open(args.predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    total = len(results)
    if total == 0:
        print("No predictions found.")
        return

    correct = [r for r in results if r.get("is_correct", False)]
    wrong = [r for r in results if not r.get("is_correct", False)]

    print(f"Total evaluated examples: {total}")
    print(f"Correct: {len(correct)}, Wrong: {len(wrong)}")
    print(f"Accuracy: {len(correct) / total * 100:.2f}%\n")

    # Accuracy by true method name length
    buckets = {
        "short (<=10 chars)": {"total": 0, "wrong": 0},
        "medium (11–20 chars)": {"total": 0, "wrong": 0},
        "long (>20 chars)": {"total": 0, "wrong": 0},
    }

    for r in results:
        name = r["true_name"]
        L = len(name)
        if L <= 10:
            key = "short (<=10 chars)"
        elif L <= 20:
            key = "medium (11–20 chars)"
        else:
            key = "long (>20 chars)"
        buckets[key]["total"] += 1
        if not r.get("is_correct", False):
            buckets[key]["wrong"] += 1

    print("Accuracy by true method name length:")
    for key, stats in buckets.items():
        if stats["total"] == 0:
            acc = 0.0
        else:
            acc = (1 - stats["wrong"] / stats["total"]) * 100
        print(f"  {key}: {acc:.2f}%  ({stats['total']} examples)")
    print()

    # Accuracy by number of subtokens in the name
    token_buckets = {
        "1 token": {"total": 0, "wrong": 0},
        "2–3 tokens": {"total": 0, "wrong": 0},
        ">3 tokens": {"total": 0, "wrong": 0},
    }

    for r in results:
        name = r["true_name"]
        tokens = split_camel_and_underscore(name)
        tlen = len(tokens)
        if tlen <= 1:
            key = "1 token"
        elif tlen <= 3:
            key = "2–3 tokens"
        else:
            key = ">3 tokens"
        token_buckets[key]["total"] += 1
        if not r.get("is_correct", False):
            token_buckets[key]["wrong"] += 1

    print("Accuracy by number of subtokens in true name:")
    for key, stats in token_buckets.items():
        if stats["total"] == 0:
            acc = 0.0
        else:
            acc = (1 - stats["wrong"] / stats["total"]) * 100
        print(f"  {key}: {acc:.2f}%  ({stats['total']} examples)")
    print()

    # For wrong predictions, measure how close they are
    if wrong:
        wrong_with_sim = []
        for r in wrong:
            sim = name_similarity(r["true_name"], r["pred_name"])
            r["similarity"] = sim
            wrong_with_sim.append(r)

        avg_sim = sum(r["similarity"] for r in wrong_with_sim) / len(wrong_with_sim)
        print(f"Average similarity for wrong predictions (0–1): {avg_sim:.3f}\n")

        wrong_sorted_high = sorted(wrong_with_sim, key=lambda x: x["similarity"], reverse=True)
        wrong_sorted_low = sorted(wrong_with_sim, key=lambda x: x["similarity"])

        print("Top 5 near-miss wrong predictions (high similarity):")
        print("-" * 80)
        for r in wrong_sorted_high[:5]:
            print(
                f"True: {r['true_name']}  | Pred: {r['pred_name']}  "
                f"| sim = {r['similarity']:.3f}"
            )
        print("-" * 80)

        print("\nTop 5 most off wrong predictions (low similarity):")
        print("-" * 80)
        for r in wrong_sorted_low[:5]:
            print(
                f"True: {r['true_name']}  | Pred: {r['pred_name']}  "
                f"| sim = {r['similarity']:.3f}"
            )
        print("-" * 80)
        print()

    # Subtokens of true name appearing in method body
    count_all_present = 0
    count_any_present = 0
    count_none_present = 0

    stats_correct = {"total": 0, "all": 0, "any": 0, "none": 0}
    stats_wrong = {"total": 0, "all": 0, "any": 0, "none": 0}

    for r in results:
        true_name = r["true_name"]
        body = r.get("method_body", "")
        body_lower = body.lower()

        subtokens = split_camel_and_underscore(true_name)
        subtokens_lower = [t.lower() for t in subtokens]

        if not subtokens_lower:
            present_flags = []
        else:
            present_flags = [tok in body_lower for tok in subtokens_lower]

        if subtokens_lower and all(present_flags):
            category = "all"
            count_all_present += 1
        elif subtokens_lower and any(present_flags):
            category = "any"
            count_any_present += 1
        else:
            category = "none"
            count_none_present += 1

        if r.get("is_correct", False):
            stats_correct["total"] += 1
            stats_correct[category] += 1
        else:
            stats_wrong["total"] += 1
            stats_wrong[category] += 1

    print("Subtoken coverage of TRUE name in method body (all examples):")
    print(
        f"  all subtokens appear:  {count_all_present} "
        f"({count_all_present / total * 100:.2f}%)"
    )
    print(
        f"  some subtokens appear: {count_any_present} "
        f"({count_any_present / total * 100:.2f}%)"
    )
    print(
        f"  no subtokens appear:   {count_none_present} "
        f"({count_none_present / total * 100:.2f}%)"
    )
    print()

    if stats_correct["total"] > 0:
        tc = stats_correct["total"]
        print("For CORRECT predictions:")
        print(f"  total: {tc}")
        print(
            f"  all subtokens appear:  {stats_correct['all']} "
            f"({stats_correct['all'] / tc * 100:.2f}%)"
        )
        print(
            f"  some subtokens appear: {stats_correct['any']} "
            f"({stats_correct['any'] / tc * 100:.2f}%)"
        )
        print(
            f"  no subtokens appear:   {stats_correct['none']} "
            f"({stats_correct['none'] / tc * 100:.2f}%)"
        )
        print()

    if stats_wrong["total"] > 0:
        tw = stats_wrong["total"]
        print("For WRONG predictions:")
        print(f"  total: {tw}")
        print(
            f"  all subtokens appear:  {stats_wrong['all']} "
            f"({stats_wrong['all'] / tw * 100:.2f}%)"
        )
        print(
            f"  some subtokens appear: {stats_wrong['any']} "
            f"({stats_wrong['any'] / tw * 100:.2f}%)"
        )
        print(
            f"  no subtokens appear:   {stats_wrong['none']} "
            f"({stats_wrong['none'] / tw * 100:.2f}%)"
        )
        print()
        
    # compare true vs false-only subtokens in method body for WRONG predictions
    if wrong:
        more_false_than_true = 0
        more_true_than_false = 0
        equal_or_zero = 0

        for r in wrong:
            body = r.get("method_body", "")
            body_lower = body.lower()

            true_tokens = [t.lower() for t in split_camel_and_underscore(r["true_name"])]
            pred_tokens = [t.lower() for t in split_camel_and_underscore(r["pred_name"])]

            true_token_set = set(true_tokens)
            pred_token_set = set(pred_tokens)

            false_only_tokens = [t for t in pred_token_set if t and t not in true_token_set]

            true_occurrences = sum(body_lower.count(tok) for tok in true_token_set if tok)
            false_occurrences = sum(body_lower.count(tok) for tok in false_only_tokens)

            if false_occurrences > true_occurrences:
                more_false_than_true += 1
            elif true_occurrences > false_occurrences:
                more_true_than_false += 1
            else:
                equal_or_zero += 1

        total_wrong = len(wrong)
        print("For WRONG predictions: comparison of subtoken occurrences in method body")
        print(
            f"  more FALSE-only subtokens than TRUE subtokens: "
            f"{more_false_than_true} ({more_false_than_true / total_wrong * 100:.2f}%)"
        )
        print(
            f"  more TRUE subtokens than FALSE-only subtokens: "
            f"{more_true_than_false} ({more_true_than_false / total_wrong * 100:.2f}%)"
        )
        print(
            f"  equal or both zero: "
            f"{equal_or_zero} ({equal_or_zero / total_wrong * 100:.2f}%)"
        )
        print()

if __name__ == "__main__":
    main()
