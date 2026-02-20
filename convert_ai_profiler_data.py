import json
import os
import argparse
from transformers import AutoTokenizer


def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def process_item(item, tokenizer):
    conversations = item.get("conversations", [])
    model_output = item.get("model_output", "")

    system_value = ""
    human_value = ""
    gpt_value = ""

    for msg in conversations:
        role = msg.get("from", "")
        value = msg.get("value", "")

        if role == "system":
            system_value = value
        elif role == "human":
            human_value = value
        elif role == "gpt":
            gpt_value = value

    # messages_src = [
    #     {"role": "system", "content": system_value},
    #     {"role": "user", "content": human_value},
    #     {"role": "user", "content": model_output},
    # ]
    # src_text = tokenizer.apply_chat_template(
    #     messages_src, tokenize=False, add_generation_prompt=False
    # )

    # messages_trg = [{"role": "assistant", "content": gpt_value}]
    # trg_text = tokenizer.apply_chat_template(
    #     messages_trg, tokenize=False, add_generation_prompt=False
    # )
    src_text = system_value + "\n\n" + human_value + "\n\n原始输出：\n" + model_output + "\n\n修正输出：\n" 
    trg_text = gpt_value

    return {"src": src_text.strip(), "trg": trg_text.strip()}


def convert_dataset(input_file, output_file, tokenizer):
    print(f"Loading data from {input_file}...")
    data = load_json_data(input_file)
    print(f"Loaded {len(data)} samples")

    results = []
    for i, item in enumerate(data):
        processed = process_item(item, tokenizer)
        results.append(processed)
        if (i + 1) % 100 == 0 or (i + 1) == len(data):
            print(f"Processed {i + 1}/{len(data)} samples")

    print(f"Writing to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done! Wrote {len(results)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ai_profiler data to FMSeq format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen3-4B",
        help="Path to Qwen model or HuggingFace model name (default: Qwen/Qwen2.5-32B)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/ai_profiler",
        help="Input directory containing JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/ai_profiler",
        help="Output directory for JSONL files",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        print("Local model not found, trying to download from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(
            "models/Qwen3-4B", trust_remote_code=True
        )

    train_input = os.path.join(args.input_dir, "combined_train_v2_qwen32b.json")
    train_output = os.path.join(args.output_dir, "train.jsonl")
    convert_dataset(train_input, train_output, tokenizer)

    test_input = os.path.join(args.input_dir, "test_v2_qwen32b.json")
    test_output = os.path.join(args.output_dir, "test.jsonl")
    convert_dataset(test_input, test_output, tokenizer)

    print("\nConversion complete!")
    print(f"Train data: {train_output}")
    print(f"Test data: {test_output}")


if __name__ == "__main__":
    main()
