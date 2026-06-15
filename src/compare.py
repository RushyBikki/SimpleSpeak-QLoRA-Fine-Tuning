"""
Compare base Qwen3 output with the SimpleSpeak LoRA adapter.
"""

import gc
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen3-1.7B"
TEST_PROMPTS_FILE = "data/test_prompts.txt"
ADAPTER_DIR = "adapters/qwen3-1.7b-simple-speak"
RESULTS_FILE = "results/results.json"


def apply_qwen_chat_template(tokenizer, messages, add_generation_prompt):
    """Apply Qwen chat template, disabling thinking mode when supported."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def remove_thinking(text):
    """Remove Qwen thinking blocks if the model emits them."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(tokenizer):
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    model.eval()
    return model


def generate_response(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = apply_qwen_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return remove_thinking(answer)


print("=" * 60)
print("SimpleSpeak Qwen3 1.7B Comparison")
print("=" * 60)

if not Path(TEST_PROMPTS_FILE).exists():
    raise FileNotFoundError(f"Test prompts file not found: {TEST_PROMPTS_FILE}")

if not Path(ADAPTER_DIR).exists():
    print("Adapter not found. Please run python src/train.py first.")
    sys.exit(1)

Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)

with open(TEST_PROMPTS_FILE, "r", encoding="utf-8") as file:
    prompts = [line.strip() for line in file if line.strip()]

print(f"[OK] Loaded {len(prompts)} prompts")

print("\nLoading tokenizer...")
tokenizer = load_tokenizer()
print("[OK] Tokenizer loaded")

print("\nStep A: Generating base model outputs...")
base_model = load_base_model(tokenizer)
base_outputs = [generate_response(base_model, tokenizer, prompt) for prompt in prompts]
del base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("[OK] Base outputs complete")

print("\nStep B: Generating fine-tuned adapter outputs...")
model = load_base_model(tokenizer)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()
fine_tuned_outputs = [generate_response(model, tokenizer, prompt) for prompt in prompts]
print("[OK] Fine-tuned outputs complete")

results = []
for prompt, base_output, fine_tuned_output in zip(
    prompts,
    base_outputs,
    fine_tuned_outputs,
):
    results.append(
        {
            "prompt": prompt,
            "base_output": base_output,
            "fine_tuned_output": fine_tuned_output,
        }
    )

with open(RESULTS_FILE, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("\n" + "=" * 60)
print("Comparison Results")
print("=" * 60)
for index, result in enumerate(results, 1):
    print(f"\nPrompt {index}: {result['prompt']}")
    print(f"Base model:\n{result['base_output']}")
    print(f"Fine-tuned model:\n{result['fine_tuned_output']}")

print(f"\nResults saved to: {RESULTS_FILE}")
