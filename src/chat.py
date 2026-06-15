"""
Interactive chat interface for the SimpleSpeak Qwen3 LoRA adapter.
"""

import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen3-1.7B"
ADAPTER_DIR = "adapters/qwen3-1.7b-simple-speak"
STYLE_INSTRUCTION = (
    "Please explain this for a middle school student. "
    "Use simple words, short sentences, and helpful analogies. "
    "Give a complete but simple explanation, and explain any technical terms."
)


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


def generate_response(model, tokenizer, prompt):
    user_content = f"{prompt}\n\n{STYLE_INSTRUCTION}"
    messages = [{"role": "user", "content": user_content}]
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
            max_new_tokens=300,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return remove_thinking(answer)


print("=" * 60)
print("SimpleSpeak Qwen3 1.7B LoRA Chat")
print("=" * 60)

if not Path(ADAPTER_DIR).exists():
    print("Adapter not found. Please run python src/train.py first.")
    sys.exit(1)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("[OK] Tokenizer loaded")

print("\nLoading model...")
if torch.cuda.is_available():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

base_model.config.pad_token_id = tokenizer.eos_token_id
base_model.config.use_cache = False
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
print("[OK] Model and adapter loaded")

print("\nSimpleSpeak Qwen3 1.7B LoRA Chat")
print("Type a Computer Science concept or question.")
print("Type 'quit', 'exit', or 'q' to stop.")

while True:
    user_input = input("\nEnter your question:\n> ").strip()

    if user_input.lower() in {"quit", "exit", "q"}:
        print("\nGoodbye.")
        break

    if not user_input:
        print("Please enter a question.")
        continue

    answer = generate_response(model, tokenizer, user_input)
    print("\nAnswer:")
    print(answer)
