import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from utils import load_pairs, split_code_prefix_suffix, show_fim_debug_examples

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-Coder on method name prediction with FIM."
    )
    parser.add_argument(
        "--train-file",
        default="train_dataset.jsonl",
        help="Input JSONL file containing method_body and method_name (default: train_dataset.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        default="qwen_methodname_finetune",
        help="Output directory for the fine-tuned model (default: qwen_methodname_finetune).",
    )
    parser.add_argument(
        "--model-dir",
        default="./qwen2.5-Coder-0.5B",
        help="Base model directory (default: ./qwen2.5-Coder-0.5B).",
    )
    parser.add_argument(
        "--debug-examples",
        type=int,
        default=2,
        help="Number of FIM debug examples to print (0 to disable, default: 2).",
    )
    args = parser.parse_args()

    model_name = args.model_dir
    output_dir = args.output_dir

    # Load tokenizer & add FIM tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    special_tokens = {"additional_special_tokens": ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]}
    tokenizer.add_special_tokens(special_tokens)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FIM_MIDDLE_ID = tokenizer.convert_tokens_to_ids("<|fim_middle|>")

    train_pairs = load_pairs(args.train_file)

    # Debug examples if requested
    if args.debug_examples > 0 and len(train_pairs) > 0:
        show_fim_debug_examples(train_pairs, args.debug_examples, tokenizer)

    # Build training sequences
    train_encodings = []
    skipped = 0

    for code, name in train_pairs:
        result = split_code_prefix_suffix(code, name)
        if result is None:
            skipped += 1
            continue

        prefix, suffix = result
        sequence = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{name}<|endoftext|>"

        enc = tokenizer(sequence, truncation=True, max_length=512)
        input_ids = enc["input_ids"]
        labels = input_ids.copy()

        try:
            middle_index = input_ids.index(FIM_MIDDLE_ID)
        except ValueError:
            skipped += 1
            continue

        for i in range(0, middle_index + 1):
            labels[i] = -100

        train_encodings.append({"input_ids": input_ids, "labels": labels})

    print(f"Prepared {len(train_encodings)} training sequences for FIM, skipped {skipped}.")
    
    # Read training data
    train_dataset = Dataset.from_list(train_encodings)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,   
        gradient_accumulation_steps=4,
        num_train_epochs=3,              
        learning_rate=5e-5,
        weight_decay=0.01, 
        warmup_ratio=0.05,  
        fp16=torch.cuda.is_available(),                     
        bf16=False,
        optim="adamw_torch",
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    print("Start training...")
    trainer.train()
    print("Training done.")

    # Save model + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved fine-tuned model + tokenizer to {output_dir}")

if __name__ == "__main__":
    main()
