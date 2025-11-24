import argparse

from utils import (
    load_prediction_records,
    group_accuracy_by_name_length,
    group_accuracy_by_subtoken_count,
    compute_true_name_subtoken_coverage,
    attach_similarity_scores_to_wrong_predictions,
    classify_wrong_predictions_with_wordnet,
    compare_true_and_false_only_subtokens_in_body,
    analyze_token_order_and_subset_patterns,
    classify_totally_wrong_predictions,
    semantic_similarity_wordnet, 
)


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

    # Load predictions and basic accuracy
    prediction_records = load_prediction_records(args.predictions_file)
    total_examples = len(prediction_records)

    if total_examples == 0:
        print("No predictions found.")
        return

    correct_predictions = [
        record for record in prediction_records
        if record.get("is_correct", False)
    ]
    wrong_predictions = [
        record for record in prediction_records
        if not record.get("is_correct", False)
    ]

    total_correct = len(correct_predictions)
    total_wrong = len(wrong_predictions)
    overall_accuracy = (total_correct / total_examples) * 100.0

    print(f"Total evaluated examples: {total_examples}")
    print(f"Correct: {total_correct}, Wrong: {total_wrong}")
    print(f"Accuracy: {overall_accuracy:.2f}%\n")

    # Accuracy by true method name length
    length_buckets = group_accuracy_by_name_length(prediction_records)
    print("Accuracy by true method name length:")
    for bucket_label, counts in length_buckets.items():
        bucket_total = counts["total"]
        bucket_wrong = counts["wrong"]
        bucket_accuracy = 0.0
        if bucket_total > 0:
            bucket_accuracy = (1.0 - bucket_wrong / bucket_total) * 100.0
        print(f"  {bucket_label}: {bucket_accuracy:.2f}%  ({bucket_total} examples)")
    print()

    # Accuracy by number of subtokens in the true name
    subtoken_buckets = group_accuracy_by_subtoken_count(prediction_records)
    print("Accuracy by number of subtokens in true name:")
    for bucket_label, counts in subtoken_buckets.items():
        bucket_total = counts["total"]
        bucket_wrong = counts["wrong"]
        bucket_accuracy = 0.0
        if bucket_total > 0:
            bucket_accuracy = (1.0 - bucket_wrong / bucket_total) * 100.0
        print(f"  {bucket_label}: {bucket_accuracy:.2f}%  ({bucket_total} examples)")
    print()

    # Coverage of true-name subtokens in method body
    overall_coverage, coverage_correct, coverage_wrong = compute_true_name_subtoken_coverage(
        prediction_records
    )

    print("Subtoken coverage of TRUE name in method body (all examples):")
    print(
        f"  all subtokens appear:  {overall_coverage['all']} "
        f"({overall_coverage['all'] / total_examples * 100:.2f}%)"
    )
    print(
        f"  some subtokens appear: {overall_coverage['any']} "
        f"({overall_coverage['any'] / total_examples * 100:.2f}%)"
    )
    print(
        f"  no subtokens appear:   {overall_coverage['none']} "
        f"({overall_coverage['none'] / total_examples * 100:.2f}%)"
    )
    print()

    if coverage_correct["total"] > 0:
        total_correct_cov = coverage_correct["total"]
        print("For CORRECT predictions:")
        print(f"  total: {total_correct_cov}")
        print(
            f"  all subtokens appear:  {coverage_correct['all']} "
            f"({coverage_correct['all'] / total_correct_cov * 100:.2f}%)"
        )
        print(
            f"  some subtokens appear: {coverage_correct['any']} "
            f"({coverage_correct['any'] / total_correct_cov * 100:.2f}%)"
        )
        print(
            f"  no subtokens appear:   {coverage_correct['none']} "
            f"({coverage_correct['none'] / total_correct_cov * 100:.2f}%)"
        )
        print()

    if coverage_wrong["total"] > 0:
        total_wrong_cov = coverage_wrong["total"]
        print("For WRONG predictions:")
        print(f"  total: {total_wrong_cov}")
        print(
            f"  all subtokens appear:  {coverage_wrong['all']} "
            f"({coverage_wrong['all'] / total_wrong_cov * 100:.2f}%)"
        )
        print(
            f"  some subtokens appear: {coverage_wrong['any']} "
            f"({coverage_wrong['any'] / total_wrong_cov * 100:.2f}%)"
        )
        print(
            f"  no subtokens appear:   {coverage_wrong['none']} "
            f"({coverage_wrong['none'] / total_wrong_cov * 100:.2f}%)"
        )
        print()

    # Wrong prediction analysis
    if not wrong_predictions:
        print("No wrong predictions to analyze further.")
        return

    wrong_with_similarity = attach_similarity_scores_to_wrong_predictions(
        wrong_predictions
    )
    average_similarity = (
        sum(record["similarity"] for record in wrong_with_similarity)
        / len(wrong_with_similarity)
    )
    print(
        f"Average similarity for wrong predictions (0–1): "
        f"{average_similarity:.3f}\n"
    )

    wrong_sorted_high = sorted(
        wrong_with_similarity,
        key=lambda record: record["similarity"],
        reverse=True,
    )
    wrong_sorted_low = sorted(
        wrong_with_similarity,
        key=lambda record: record["similarity"],
    )

    print("Top 5 near-miss wrong predictions (high similarity):")
    print("-" * 80)
    for record in wrong_sorted_high[:5]:
        print(
            f"True: {record['true_name']}  | Pred: {record['pred_name']}  "
            f"| sim = {record['similarity']:.3f}"
        )
    print("-" * 80)

    print("\nTop 5 most off wrong predictions (low similarity):")
    print("-" * 80)
    for record in wrong_sorted_low[:5]:
        semantic_sim = semantic_similarity_wordnet(
            record["true_name"], record["pred_name"]
        )
        print(
            f"True: {record['true_name']}  | Pred: {record['pred_name']}  "
            f"| string_sim = {record['similarity']:.3f}  "
            f"| semantic_sim = {semantic_sim:.3f}"
        )
    print("-" * 80)
    print()

    # WordNet-based semantic breakdown
    (
        synonym_like_count,
        ambiguous_like_count,
        other_wrong_count,
        synonym_examples,
    ) = classify_wrong_predictions_with_wordnet(wrong_with_similarity)
    total_semantic_cases = (
        synonym_like_count + ambiguous_like_count + other_wrong_count
    )

    if total_semantic_cases > 0:
        print(
            "Semantic-style breakdown for WRONG predictions "
            "(WordNet, symmetric subtokens):"
        )
        print(
            f"  synonym-like (high similarity, overlap, and some WordNet synonym subtokens): "
            f"{synonym_like_count} "
            f"({synonym_like_count / total_semantic_cases * 100:.2f}%)"
        )
        print(
            f"  ambiguous-like (high similarity and overlap, but no synonym pair): "
            f"{ambiguous_like_count} "
            f"({ambiguous_like_count / total_semantic_cases * 100:.2f}%)"
        )
        print(
            f"  other wrong cases: {other_wrong_count} "
            f"({other_wrong_count / total_semantic_cases * 100:.2f}%)"
        )
        print()

        if synonym_examples:
            print("Example synonym-like WRONG predictions (WordNet):")
            print("-" * 80)
            for example in synonym_examples:
                print(
                    f"True: {example['true_name']}  | Pred: {example['pred_name']}"
                )
            print("-" * 80)
            print()

    # Frequency comparison: true vs predicted-only subtokens in method body
    more_false_only, more_true, equal_or_zero = compare_true_and_false_only_subtokens_in_body(
        wrong_predictions
    )
    print("For WRONG predictions: comparison of subtoken occurrences in method body")
    print(
        f"  more FALSE-only subtokens than TRUE subtokens: "
        f"{more_false_only} ({more_false_only / total_wrong * 100:.2f}%)"
    )
    print(
        f"  more TRUE subtokens than FALSE-only subtokens: "
        f"{more_true} ({more_true / total_wrong * 100:.2f}%)"
    )
    print(
        f"  equal or both zero: "
        f"{equal_or_zero} ({equal_or_zero / total_wrong * 100:.2f}%)"
    )
    print()

    # Token order / subset / superset patterns
    (
        same_tokens_different_order,
        true_subset_predicted,
        predicted_subset_true,
        mixed_or_disjoint,
    ) = analyze_token_order_and_subset_patterns(wrong_predictions)

    print("For WRONG predictions: token order and subset/superset patterns")
    print(
        f"  same tokens but different order: "
        f"{same_tokens_different_order} "
        f"({same_tokens_different_order / total_wrong * 100:.2f}%)"
    )
    print(
        f"  TRUE tokens subset of PRED tokens (over-specified prediction): "
        f"{true_subset_predicted} "
        f"({true_subset_predicted / total_wrong * 100:.2f}%)"
    )
    print(
        f"  PRED tokens subset of TRUE tokens (under-specified prediction): "
        f"{predicted_subset_true} "
        f"({predicted_subset_true / total_wrong * 100:.2f}%)"
    )
    print(
        f"  mixed / disjoint token sets: "
        f"{mixed_or_disjoint} "
        f"({mixed_or_disjoint / total_wrong * 100:.2f}%)"
    )
    print()

    # Totally wrong cases: no overlapping subtokens and no synonym pair
    totally_wrong_count, totally_wrong_examples = classify_totally_wrong_predictions(
        wrong_with_similarity
    )
    if totally_wrong_count > 0:
        print(
            "Totally wrong predictions (no overlapping subtokens and no WordNet synonym pair):"
        )
        print(
            f"  count: {totally_wrong_count} "
            f"({totally_wrong_count / total_wrong * 100:.2f}% of all wrong predictions)"
        )
        print()

        print("Example totally wrong predictions:")
        print("-" * 80)
        for example in totally_wrong_examples[:5]:
            semantic_sim = semantic_similarity_wordnet(
                example["true_name"], example["pred_name"]
            )
            print(
                f"True: {example['true_name']}  | Pred: {example['pred_name']}  "
                f"| string_sim = {example.get('similarity', 0.0):.3f}  "
                f"| semantic_sim = {semantic_sim:.3f}"
            )
        print("-" * 80)
        print()


if __name__ == "__main__":
    main()