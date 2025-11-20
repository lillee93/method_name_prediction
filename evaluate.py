import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_pairs, split_code_prefix_suffix, save_prediction_results

MAX_NEW_TOKENS = 8

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a FIM-trained model on method name prediction."
    )
    parser.add_argument(
        "--model-dir",
        default="./qwen_methodname_finetune",
        help="Directory of the model to evaluate (default: ./qwen_methodname_finetune).",
    )
    parser.add_argument(
        "--test-file",
        default="./test_dataset.jsonl",
        help="JSONL test file with method_body and method_name (default: ./test_dataset.jsonl).",
    )
    parser.add_argument(
        "--predictions-file",
        default=None,
        help="Optional path to save all prediction results as JSONL (no file if omitted).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()

    # Load test data
    test_pairs = load_pairs(args.test_file)
    print(f"Loaded {len(test_pairs)} test examples.")

    correct = 0
    total = 0

    results = []

    # Evaluate
    for code, true_name in test_pairs:
        res = split_code_prefix_suffix(code, true_name)
        if res is None:
            continue
        prefix, suffix = res

        prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )[0]

        gen_tokens = output_ids[input_ids.shape[1]:]
        pred_name = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        total += 1
        is_correct = (pred_name == true_name)
        if is_correct:
            correct += 1

        results.append({
            "true_name": true_name,
            "pred_name": pred_name,
            "is_correct": is_correct,
            "method_body": code,
        })

    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    print(f"Exact-match accuracy on test set: {accuracy:.2f}%  ({correct}/{total})")

    if args.predictions_file is not None:
        save_prediction_results(results, args.predictions_file)
        print(f"Saved all predictions to {args.predictions_file}")
        
if __name__ == "__main__":
    main()