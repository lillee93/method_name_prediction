import json
import javalang
import re

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