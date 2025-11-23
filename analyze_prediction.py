import argparse
import json

from utils import split_camel_and_underscore, name_similarity, are_wordnet_synonyms


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

    # For wrong predictions, measure how close they are + WordNet-based semantics
    if wrong:
        wrong_with_sim = []
        for r in wrong:
            sim = name_similarity(r["true_name"], r["pred_name"])
            r["similarity"] = sim
            wrong_with_sim.append(r)

        avg_sim = sum(r["similarity"] for r in wrong_with_sim) / len(wrong_with_sim)
        print(f"Average similarity for wrong predictions (0–1): {avg_sim:.3f}\n")

        wrong_sorted_high = sorted(
            wrong_with_sim, key=lambda x: x["similarity"], reverse=True
        )
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

        # Semantic-style breakdown for WRONG predictions using WordNet (treat all subtokens symmetrically)
        def find_synonym_pair(true_tokens_set, pred_tokens_set):
            for t in true_tokens_set:
                for p in pred_tokens_set:
                    if are_wordnet_synonyms(t, p):
                        return t, p
            return None, None

        synonym_like = 0
        ambiguous_like = 0
        other_wrong_semantic = 0
        synonym_examples = []

        for r in wrong_with_sim:
            true_tokens = split_camel_and_underscore(r["true_name"])
            pred_tokens = split_camel_and_underscore(r["pred_name"])

            true_tokens_lower = [t.lower() for t in true_tokens]
            pred_tokens_lower = [t.lower() for t in pred_tokens]

            if not true_tokens_lower or not pred_tokens_lower:
                other_wrong_semantic += 1
                continue

            true_set = set(true_tokens_lower)
            pred_set = set(pred_tokens_lower)

            common_exact = true_set.intersection(pred_set)
            min_len = max(1, min(len(true_set), len(pred_set)))
            overlap_ratio = len(common_exact) / min_len

            full_sim = r["similarity"]

            syn_t, syn_p = find_synonym_pair(true_set, pred_set)
            has_synonyms = syn_t is not None

            if full_sim >= 0.7 and overlap_ratio >= 0.5 and has_synonyms:
                synonym_like += 1
                if len(synonym_examples) < 5:
                    synonym_examples.append(
                        {
                            "true_name": r["true_name"],
                            "pred_name": r["pred_name"],
                            "syn_pair": (syn_t, syn_p),
                            "similarity": full_sim,
                        }
                    )
                continue

            if full_sim >= 0.7 and overlap_ratio >= 0.5:
                ambiguous_like += 1
            else:
                other_wrong_semantic += 1

        total_wrong_semantic = synonym_like + ambiguous_like + other_wrong_semantic
        if total_wrong_semantic > 0:
            print(
                "Semantic-style breakdown for WRONG predictions (WordNet, symmetric subtokens):"
            )
            print(
                f"  synonym-like (high similarity, overlap, and some WordNet synonym subtokens): "
                f"{synonym_like} ({synonym_like / total_wrong_semantic * 100:.2f}%)"
            )
            print(
                f"  ambiguous-like (high similarity and overlap, but no synonym pair): "
                f"{ambiguous_like} ({ambiguous_like / total_wrong_semantic * 100:.2f}%)"
            )
            print(
                f"  other wrong cases: "
                f"{other_wrong_semantic} ({other_wrong_semantic / total_wrong_semantic * 100:.2f}%)"
            )
            print()

            if synonym_examples:
                print("Example synonym-like WRONG predictions (WordNet):")
                print("-" * 80)
                for ex in synonym_examples:
                    print(
                        f"True: {ex['true_name']}  | Pred: {ex['pred_name']}  "
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

    # compare true vs false-only subtokens in method body for WRONG predictions + token order patterns
    if wrong:
        more_false_than_true = 0
        more_true_than_false = 0
        equal_or_zero = 0

        same_tokens_diff_order = 0
        true_subset_pred = 0
        pred_subset_true = 0
        mixed_or_disjoint = 0

        for r in wrong:
            body = r.get("method_body", "")
            body_lower = body.lower()

            true_tokens = [t.lower() for t in split_camel_and_underscore(r["true_name"])]
            pred_tokens = [t.lower() for t in split_camel_and_underscore(r["pred_name"])]

            true_token_set = set(true_tokens)
            pred_token_set = set(pred_tokens)

            # token frequency comparison
            false_only_tokens = [t for t in pred_token_set if t and t not in true_token_set]

            true_occurrences = sum(
                body_lower.count(tok) for tok in true_token_set if tok
            )
            false_occurrences = sum(
                body_lower.count(tok) for tok in false_only_tokens
            )

            if false_occurrences > true_occurrences:
                more_false_than_true += 1
            elif true_occurrences > false_occurrences:
                more_true_than_false += 1
            else:
                equal_or_zero += 1

            # token order / subset–superset patterns
            if true_tokens and pred_tokens:
                if sorted(true_tokens) == sorted(pred_tokens) and true_tokens != pred_tokens:
                    same_tokens_diff_order += 1
                elif true_token_set.issubset(pred_token_set) and true_token_set != pred_token_set:
                    true_subset_pred += 1
                elif pred_token_set.issubset(true_token_set) and pred_token_set != true_token_set:
                    pred_subset_true += 1
                else:
                    mixed_or_disjoint += 1

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

        print("For WRONG predictions: token order and subset/superset patterns")
        print(
            f"  same tokens but different order: "
            f"{same_tokens_diff_order} ({same_tokens_diff_order / total_wrong * 100:.2f}%)"
        )
        print(
            f"  TRUE tokens subset of PRED tokens (over-specified prediction): "
            f"{true_subset_pred} ({true_subset_pred / total_wrong * 100:.2f}%)"
        )
        print(
            f"  PRED tokens subset of TRUE tokens (under-specified prediction): "
            f"{pred_subset_true} ({pred_subset_true / total_wrong * 100:.2f}%)"
        )
        print(
            f"  mixed / disjoint token sets: "
            f"{mixed_or_disjoint} ({mixed_or_disjoint / total_wrong * 100:.2f}%)"
        )
        print()


if __name__ == "__main__":
    main()