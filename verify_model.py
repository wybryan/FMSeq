import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def verify_model(model_path: str = "/Projects/FMSeq/models/Qwen3-0.6B"):
    print(f"Verifying model at: {model_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("-" * 50)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Tokenizer loaded: {type(tokenizer).__name__}")
    print(f"Vocab size: {len(tokenizer)}")
    print("-" * 50)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    print(f"Model loaded successfully!")
    print(f"Model type: {model.config.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print("-" * 50)

    print("Testing inference...")
    test_prompt = "Hello, how are you?"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_prompt}")
    print(f"Output: {result}")
    print("-" * 50)

    print("Verification PASSED!")


if __name__ == "__main__":
    verify_model()
