# build_dataset.py
import argparse
import glob
import random
import sys
from pathlib import Path

from utils import extract_methods_from_file, save_pairs

FORBIDDEN_SUBSTRINGS = [
    "accesssecret",
    "access_secret",
    "secretkey",
    "accesskey",
    "_id",
    "user_id",
    "client_id",
]
def main():
    parser = argparse.ArgumentParser(
        description="Extract Java methods and build train/test JSONL datasets."
    )
    parser.add_argument(
        "--repos-dir",
        default="repos",
        help="Root directory of cloned repos (default: ./repos)",
    )
    parser.add_argument(
        "--max-methods",
        type=int,
        default=2000000,
        help="Maximum number of raw methods to collect before dedup (default: 2000000)",
    )
    parser.add_argument(
        "--max-unique",
        type=int,
        default=50000,
        help="Maximum number of unique methods to keep (default: 50000)",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=10000,
        help="Python recursion limit for javalang (default: 10000)",
    )
    parser.add_argument(
        "--train-file",
        default="train_datasets.jsonl",
        help="Output path for training file (default: train_datasets.jsonl)",
    )
    parser.add_argument(
        "--test-file",
        default="test_datasets.jsonl",
        help="Output path for test file (default: test_datasets.jsonl)",
    )
    args = parser.parse_args()

    # set recursion limit
    sys.setrecursionlimit(args.recursion_limit)

    repos_dir = Path(args.repos_dir)
    print(f"Scanning Java files under: {repos_dir.resolve()}")

    java_glob = str(repos_dir / "**" / "*.java")
    java_files = glob.glob(java_glob, recursive=True)
    print(f"Found {len(java_files)} .java files.")

    # collect raw methods
    all_methods = []
    for idx, java_file in enumerate(java_files, start=1):
        if len(all_methods) >= args.max_methods:
            print(f"Reached max_methods = {args.max_methods}, stopping collection.")
            break

        methods = extract_methods_from_file(java_file)
        if not methods:
            continue

        all_methods.extend(methods)

        if len(all_methods) % 10_000 < len(methods):
            print(f"[{idx}/{len(java_files)}] Collected methods so far: {len(all_methods)}")

    print(f"Total collected methods (before dedup): {len(all_methods)}")

    # dedup and enforce max_unique, also skip 'test' names again for safety
    unique_methods = {}
    for code, name in all_methods:
        lname = name.lower()
        lcode = code.lower()

        # skip test methods
        if "test" in lname:
            continue

        # skip methods containing sensitive patterns in name or body
        if any(s in lname or s in lcode for s in FORBIDDEN_SUBSTRINGS):
            continue
        
        if code not in unique_methods:
            unique_methods[code] = name
            if len(unique_methods) >= args.max_unique:
                print(f"Reached max_unique = {args.max_unique}, stopping dedup.")
                break

    methods_dataset = list(unique_methods.items())
    print(f"Total unique methods collected (capped): {len(methods_dataset)}")

    # shuffle + split 80/20
    random.shuffle(methods_dataset)
    split_idx = int(0.8 * len(methods_dataset))
    train_pairs = methods_dataset[:split_idx]
    test_pairs = methods_dataset[split_idx:]

    print(f"Train examples: {len(train_pairs)}, Test examples: {len(test_pairs)}")

    # save to JSONL
    save_pairs(train_pairs, args.train_file)
    save_pairs(test_pairs, args.test_file)

    print(f"Saved train dataset to: {args.train_file}")
    print(f"Saved test dataset  to: {args.test_file}")


if __name__ == "__main__":
    main()
