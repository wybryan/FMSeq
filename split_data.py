import json
import argparse
import random
import os


def split_jsonl(input_file, output_dir, valid_ratio=0.05, seed=42):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Loaded {len(lines)} samples from {input_file}")

    random.seed(seed)
    indices = list(range(len(lines)))
    random.shuffle(indices)

    valid_size = int(len(lines) * valid_ratio)
    valid_indices = set(indices[:valid_size])

    train_lines = [lines[i] for i in range(len(lines)) if i not in valid_indices]
    valid_lines = [lines[i] for i in range(len(lines)) if i in valid_indices]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    valid_path = os.path.join(output_dir, "valid.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(valid_path, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)

    print(f"Train: {len(train_lines)} samples -> {train_path}")
    print(f"Valid: {len(valid_lines)} samples -> {valid_path}")


def main():
    parser = argparse.ArgumentParser(description="Split a JSONL file into train and valid sets")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for train.jsonl and valid.jsonl")
    parser.add_argument("--valid_ratio", type=float, default=0.05, help="Fraction of data for validation (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    split_jsonl(args.input, args.output_dir, args.valid_ratio, args.seed)


if __name__ == "__main__":
    main()
