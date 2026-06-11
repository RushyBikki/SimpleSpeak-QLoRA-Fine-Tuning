"""
Interactive chat interface for the fine-tuned SimpleSpeak model.

Loads the QLoRA adapter and provides a command-line interface to
simplify technical or complicated text in real-time.
"""

import os
import torch
from pathlib import Path

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "adapters/qwen-simple-speak"

# ============================================================
# SETUP AND VALIDATION
# ============================================================

print("=" * 60)
print("SimpleSpeak QLoRA Chat")
print("=" * 60)

# Check if adapter exists
if not Path(ADAPTER_DIR).exists():
    print("\n" + "!" * 60)
    print("ERROR: Adapter not found.")
    print("=" * 60)
    print(f"Adapter folder not found: {ADAPTER_DIR}")
    print("Please run: python src/train.py")
    print("=" * 60 + "\n")
    exit(1)

print(f"✓ Adapter found: {ADAPTER_DIR}\n")

# Check for CUDA
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"✓ CUDA available. Device: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ CUDA not available. Using CPU (inference will be slower).")

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"
print("✓ Tokenizer loaded")

# ============================================================
# LOAD BASE MODEL AND ADAPTER
# ============================================================

print("\nLoading model and adapter...")

if cuda_available:
    # 4-bit quantization for GPU
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
    # CPU loading without quantization
    print("⚠ Loading on CPU without quantization (may be slow and use more memory)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()  # Set to evaluation mode

print("✓ Model and adapter loaded")

# ============================================================
# CHAT LOOP
# ============================================================

print("\n" + "=" * 60)
print("Ready for chat!")
print("=" * 60)
print("Type text to simplify.")
print("Type 'quit', 'exit', or 'q' to stop.\n")

while True:
    # Get user input
    user_input = input("Enter text to simplify:\n> ").strip()
    
    # Check for exit commands
    if user_input.lower() in ["quit", "exit", "q"]:
        print("\nGoodbye!\n")
        break
    
    # Skip empty input
    if not user_input:
        print("Please enter some text.\n")
        continue
    
    # Prepare the prompt with the task instruction
    prompt = f"Rewrite this in simpler words: {user_input}"
    
    # Create message in chat format
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template and tokenize
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    # Generate output
    print("\nSimplified answer:")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the newly generated tokens (skip the input prompt)
    generated_ids = output_ids[0][input_ids.shape[-1]:]
    simplified_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"{simplified_text.strip()}\n")

print("=" * 60 + "\n")
