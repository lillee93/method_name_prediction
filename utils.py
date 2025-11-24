import json
import javalang
import re
import nltk
from nltk.corpus import wordnet as wn

def extract_methods_from_file(file_path):
    methods = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except PermissionError:
        return methods
    try:
        tree = javalang.parse.parse(code)
        # print(f"Parsed {file_path} successfully.")
    except (javalang.parser.JavaSyntaxError,
            javalang.tokenizer.LexerError) as e:
        return methods

    try:
        # Traverse AST to find method declarations
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            name = node.name

            if node.position:
                start_line = node.position[0] - 1
            else:
                continue
            lines = code.splitlines()
            # Starting from method declaration line, collect lines until matching braces
            brace_count = 0
            method_lines = []
            for i in range(start_line, len(lines)):
                line = lines[i]
                brace_count += line.count('{') - line.count('}')
                method_lines.append(line)

                if brace_count == 0 and '{' in lines[start_line]:
                    break
            method_code = "\n".join(method_lines).strip()

            tokens = [tok for tok in method_code.replace(';', ' ').replace('(', ' ').replace(')', ' ').split()]
            if len(tokens) > 256:
                continue
            methods.append((method_code, name))
    except RecursionError as e:
        return methods        

    return methods

def save_pairs(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for code, name in pairs:
            json_line = {"method_body": code, "method_name": name}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

def load_pairs(path: str):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["method_body"], obj["method_name"]))
    return pairs

def split_code_prefix_suffix(method_code: str, method_name: str):
    idx = method_code.find(method_name)
    if idx == -1:
        return None
    prefix = method_code[:idx]
    suffix = method_code[idx + len(method_name):]
    return prefix, suffix

def show_fim_debug_examples(train_pairs, num_examples, tokenizer):
    print(f"Showing up to {num_examples} FIM debug examples.")
    shown = 0

    for idx, (code, name) in enumerate(train_pairs):
        if shown >= num_examples:
            break

        result = split_code_prefix_suffix(code, name)
        if result is None:
            continue

        prefix, suffix = result

        fim_sequence = (
            f"<|fim_prefix|>{prefix}"
            f"<|fim_suffix|>{suffix}"
            f"<|fim_middle|>{name}<|endoftext|>"
        )

        print("=" * 80)
        print(f"Example {shown}")
        print("Original method code:\n")
        print(code)
        print("\nMethod name:", name)
        print("\nFIM sequence (with special tokens):\n")
        print(fim_sequence)
        print("=" * 80)

        shown += 1

    if shown == 0:
        print("No valid FIM examples to show (method name not found in code).")

def load_prediction_records(predictions_file_path):
    prediction_records = []
    with open(predictions_file_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            prediction_records.append(json.loads(stripped_line))
    return prediction_records
    
def save_prediction_results(results, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def split_camel_and_underscore(name: str):
    parts = []
    for chunk in name.split("_"):
        parts.extend(re.findall(r"[A-Z]?[a-z0-9]+", chunk))
    return [p for p in parts if p]

def edit_distance(first_string, second_string):
    len_first = len(first_string)
    len_second = len(second_string)

    if len_first == 0:
        return len_second
    if len_second == 0:
        return len_first

    dp = [[0] * (len_second + 1) for _ in range(len_first + 1)]

    for i in range(len_first + 1):
        dp[i][0] = i
    for j in range(len_second + 1):
        dp[0][j] = j

    for i in range(1, len_first + 1):
        char_first = first_string[i - 1]
        for j in range(1, len_second + 1):
            char_second = second_string[j - 1]
            substitution_cost = 0 if char_first == char_second else 1

            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + substitution_cost)

    return dp[len_first][len_second]

def name_similarity(true_name, predicted_name):
    if not true_name and not predicted_name:
        return 1.0

    distance = edit_distance(true_name, predicted_name)
    longest_length = max(len(true_name), len(predicted_name))

    if longest_length == 0:
        return 1.0

    return 1.0 - distance / longest_length


def are_wordnet_synonyms(word1, word2):
    if not word1 or not word2:
        return False

    word1 = word1.lower()
    word2 = word2.lower()

    if word1 == word2:
        return True

    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    if not synsets1 or not synsets2:
        return False

    lemmas1 = {lemma.name().lower() for syn in synsets1 for lemma in syn.lemmas()}
    lemmas2 = {lemma.name().lower() for syn in synsets2 for lemma in syn.lemmas()}

    return len(lemmas1.intersection(lemmas2)) > 0



def load_prediction_records(predictions_file_path):
    prediction_records = []
    with open(predictions_file_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            prediction_records.append(json.loads(stripped_line))
    return prediction_records


def group_accuracy_by_name_length(prediction_records):
    grouped_counts = {
        "short (<=10 chars)": {"total": 0, "wrong": 0},
        "medium (11–20 chars)": {"total": 0, "wrong": 0},
        "long (>20 chars)": {"total": 0, "wrong": 0},
    }

    for record in prediction_records:
        true_name = record["true_name"]
        name_length = len(true_name)

        if name_length <= 10:
            bucket_key = "short (<=10 chars)"
        elif name_length <= 20:
            bucket_key = "medium (11–20 chars)"
        else:
            bucket_key = "long (>20 chars)"

        grouped_counts[bucket_key]["total"] += 1
        if not record.get("is_correct", False):
            grouped_counts[bucket_key]["wrong"] += 1

    return grouped_counts


def group_accuracy_by_subtoken_count(prediction_records):
    grouped_counts = {
        "1 token": {"total": 0, "wrong": 0},
        "2–3 tokens": {"total": 0, "wrong": 0},
        ">3 tokens": {"total": 0, "wrong": 0},
    }

    for record in prediction_records:
        true_name = record["true_name"]
        true_subtokens = split_camel_and_underscore(true_name)
        subtoken_count = len(true_subtokens)

        if subtoken_count <= 1:
            bucket_key = "1 token"
        elif subtoken_count <= 3:
            bucket_key = "2–3 tokens"
        else:
            bucket_key = ">3 tokens"

        grouped_counts[bucket_key]["total"] += 1
        if not record.get("is_correct", False):
            grouped_counts[bucket_key]["wrong"] += 1

    return grouped_counts


def compute_true_name_subtoken_coverage(prediction_records):
    overall_counts = {"all": 0, "any": 0, "none": 0}
    correct_counts = {"total": 0, "all": 0, "any": 0, "none": 0}
    wrong_counts = {"total": 0, "all": 0, "any": 0, "none": 0}

    for record in prediction_records:
        true_name = record["true_name"]
        method_body = record.get("method_body", "")
        method_body_lower = method_body.lower()

        true_subtokens = split_camel_and_underscore(true_name)
        true_subtokens_lower = [token.lower() for token in true_subtokens]

        if not true_subtokens_lower:
            present_flags = []
        else:
            present_flags = [
                subtoken in method_body_lower for subtoken in true_subtokens_lower
            ]

        if true_subtokens_lower and all(present_flags):
            coverage_category = "all"
            overall_counts["all"] += 1
        elif true_subtokens_lower and any(present_flags):
            coverage_category = "any"
            overall_counts["any"] += 1
        else:
            coverage_category = "none"
            overall_counts["none"] += 1

        if record.get("is_correct", False):
            correct_counts["total"] += 1
            correct_counts[coverage_category] += 1
        else:
            wrong_counts["total"] += 1
            wrong_counts[coverage_category] += 1

    return overall_counts, correct_counts, wrong_counts


def attach_similarity_scores_to_wrong_predictions(wrong_prediction_records):
    records_with_similarity = []
    for record in wrong_prediction_records:
        true_name = record["true_name"]
        predicted_name = record["pred_name"]
        similarity_score = name_similarity(true_name, predicted_name)

        record_copy = dict(record)
        record_copy["similarity"] = similarity_score
        records_with_similarity.append(record_copy)

    return records_with_similarity


def find_wordnet_synonym_pair(true_token_set, predicted_token_set):
    for true_token in true_token_set:
        for predicted_token in predicted_token_set:
            if are_wordnet_synonyms(true_token, predicted_token):
                return true_token, predicted_token
    return None, None


def classify_wrong_predictions_with_wordnet(wrong_records_with_similarity):
    synonym_like_count = 0
    ambiguous_like_count = 0
    other_wrong_count = 0
    synonym_examples = []

    for record in wrong_records_with_similarity:
        true_name = record["true_name"]
        predicted_name = record["pred_name"]
        similarity_score = record["similarity"]

        true_tokens = split_camel_and_underscore(true_name)
        predicted_tokens = split_camel_and_underscore(predicted_name)

        true_tokens_lower = [token.lower() for token in true_tokens]
        predicted_tokens_lower = [token.lower() for token in predicted_tokens]

        if not true_tokens_lower or not predicted_tokens_lower:
            other_wrong_count += 1
            continue

        true_token_set = set(true_tokens_lower)
        predicted_token_set = set(predicted_tokens_lower)

        common_tokens = true_token_set.intersection(predicted_token_set)
        min_token_count = max(1, min(len(true_token_set), len(predicted_token_set)))
        overlap_ratio = float(len(common_tokens)) / float(min_token_count)

        synonym_true_token, synonym_predicted_token = find_wordnet_synonym_pair(
            true_token_set, predicted_token_set
        )
        has_synonym_pair = synonym_true_token is not None

        is_high_similarity = similarity_score >= 0.7
        has_decent_overlap = overlap_ratio >= 0.5

        if is_high_similarity and has_decent_overlap and has_synonym_pair:
            synonym_like_count += 1
            if len(synonym_examples) < 5:
                synonym_examples.append(
                    {
                        "true_name": true_name,
                        "pred_name": predicted_name,
                        "syn_pair": (synonym_true_token, synonym_predicted_token),
                        "similarity": similarity_score,
                    }
                )
        elif is_high_similarity and has_decent_overlap:
            ambiguous_like_count += 1
        else:
            other_wrong_count += 1

    return synonym_like_count, ambiguous_like_count, other_wrong_count, synonym_examples


def compare_true_and_false_only_subtokens_in_body(wrong_prediction_records):
    more_false_only = 0
    more_true = 0
    equal_or_zero = 0

    for record in wrong_prediction_records:
        method_body = record.get("method_body", "")
        method_body_lower = method_body.lower()

        true_tokens = [
            token.lower()
            for token in split_camel_and_underscore(record["true_name"])
        ]
        predicted_tokens = [
            token.lower()
            for token in split_camel_and_underscore(record["pred_name"])
        ]

        true_token_set = set(true_tokens)
        predicted_token_set = set(predicted_tokens)

        false_only_tokens = [
            token
            for token in predicted_token_set
            if token and token not in true_token_set
        ]

        true_occurrences = sum(
            method_body_lower.count(token) for token in true_token_set if token
        )
        false_only_occurrences = sum(
            method_body_lower.count(token) for token in false_only_tokens
        )

        if false_only_occurrences > true_occurrences:
            more_false_only += 1
        elif true_occurrences > false_only_occurrences:
            more_true += 1
        else:
            equal_or_zero += 1

    return more_false_only, more_true, equal_or_zero


def analyze_token_order_and_subset_patterns(wrong_prediction_records):
    """
    For wrong predictions, classify token-level differences between true and predicted names
    into four categories (order / subset / superset / mixed).
    """
    same_tokens_different_order = 0
    true_subset_predicted = 0
    predicted_subset_true = 0
    mixed_or_disjoint = 0

    for record in wrong_prediction_records:
        true_tokens = [
            token.lower()
            for token in split_camel_and_underscore(record["true_name"])
        ]
        predicted_tokens = [
            token.lower()
            for token in split_camel_and_underscore(record["pred_name"])
        ]

        if not true_tokens or not predicted_tokens:
            mixed_or_disjoint += 1
            continue

        true_token_set = set(true_tokens)
        predicted_token_set = set(predicted_tokens)

        if sorted(true_tokens) == sorted(predicted_tokens) and true_tokens != predicted_tokens:
            same_tokens_different_order += 1
        elif true_token_set.issubset(predicted_token_set) and true_token_set != predicted_token_set:
            true_subset_predicted += 1
        elif predicted_token_set.issubset(true_token_set) and predicted_token_set != true_token_set:
            predicted_subset_true += 1
        else:
            mixed_or_disjoint += 1

    return (
        same_tokens_different_order,
        true_subset_predicted,
        predicted_subset_true,
        mixed_or_disjoint,
    )


def classify_totally_wrong_predictions(wrong_records_with_similarity):
    """
    Identify 'totally wrong' cases among wrong predictions:
    - no overlapping subtokens between true and predicted names
    - no WordNet synonym pair between any true and predicted subtokens
    """
    totally_wrong_count = 0
    totally_wrong_examples = []

    for record in wrong_records_with_similarity:
        true_tokens = [
            token.lower()
            for token in split_camel_and_underscore(record["true_name"])
        ]
        predicted_tokens = [
            token.lower()
            for token in split_camel_and_underscore(record["pred_name"])
        ]

        if not true_tokens or not predicted_tokens:
            continue

        true_token_set = set(true_tokens)
        predicted_token_set = set(predicted_tokens)

        common_tokens = true_token_set.intersection(predicted_token_set)
        if common_tokens:
            continue

        synonym_true_token, synonym_predicted_token = find_wordnet_synonym_pair(
            true_token_set, predicted_token_set
        )
        if synonym_true_token is not None:
            continue

        totally_wrong_count += 1
        if len(totally_wrong_examples) < 10:
            example_copy = dict(record)
            example_copy["syn_pair"] = None
            totally_wrong_examples.append(example_copy)

    return totally_wrong_count, totally_wrong_examples

def semantic_similarity_wordnet(true_name, predicted_name):
    """
    Compute semantic similarity between the true and predicted
    method names
    """
    true_tokens = [token.lower() for token in split_camel_and_underscore(true_name)]
    predicted_tokens = [
        token.lower() for token in split_camel_and_underscore(predicted_name)
    ]

    if not true_tokens or not predicted_tokens:
        return 0.0

    def best_synset_similarity(word, other_word):
        synsets1 = wn.synsets(word)
        synsets2 = wn.synsets(other_word)
        if not synsets1 or not synsets2:
            return 0.0

        best_score = 0.0
        for synset1 in synsets1:
            for synset2 in synsets2:
                score = synset1.wup_similarity(synset2)
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
        return best_score

    per_token_scores = []
    for true_token in true_tokens:
        best_for_true_token = 0.0
        for predicted_token in predicted_tokens:
            score = best_synset_similarity(true_token, predicted_token)
            if score > best_for_true_token:
                best_for_true_token = score
        if best_for_true_token > 0.0:
            per_token_scores.append(best_for_true_token)

    if not per_token_scores:
        return 0.0

    return sum(per_token_scores) / float(len(per_token_scores))