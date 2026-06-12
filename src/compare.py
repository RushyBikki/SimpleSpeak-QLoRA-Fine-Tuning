"""
Comparison script: base model vs. fine-tuned adapter.

Loads prompts from data/test_prompts.txt, generates outputs from both
the base GPT-2 and the fine-tuned adapter, then saves results to
results/results.json for comparison.
"""

import os
import json
import torch
from pathlib import Path

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai-community/gpt2"
TEST_PROMPTS_FILE = "data/test_prompts.txt"
ADAPTER_DIR = "adapters/gpt2-simple-speak"
RESULTS_FILE = "results/results.json"

# ============================================================
# SETUP AND VALIDATION
# ============================================================

print("=" * 60)
print("SimpleSpeak Model Comparison Script")
print("=" * 60)

# Create results directory
Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)

# Check if test prompts file exists
if not Path(TEST_PROMPTS_FILE).exists():
    raise FileNotFoundError(
        f"Test prompts file not found: {TEST_PROMPTS_FILE}\n"
        "Please ensure data/test_prompts.txt exists with test prompts (one per line)."
    )

print(f"[OK] Test prompts file found: {TEST_PROMPTS_FILE}")

# Check if adapter exists
if not Path(ADAPTER_DIR).exists():
    print("\n" + "!" * 60)
    print("ERROR: Adapter not found.")
    print("=" * 60)
    print(f"Adapter folder not found: {ADAPTER_DIR}")
    print("Please run: python src/train.py")
    print("=" * 60 + "\n")
    exit(1)

print(f"[OK] Adapter found: {ADAPTER_DIR}")

# Check for CUDA
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"[OK] CUDA available. Device: {torch.cuda.get_device_name(0)}")
else:
    print("[WARN] CUDA not available. Using CPU (inference will be slower).")

# ============================================================
# LOAD TEST PROMPTS
# ============================================================

print("\nLoading test prompts...")
with open(TEST_PROMPTS_FILE, "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

print(f"[OK] Loaded {len(prompts)} test prompts")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_tokenizer():
    """Load tokenizer from model."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def format_prompt(user_text):
    """Format prompt for GPT-2 (no chat template)."""
    return f"User: {user_text}\nAssistant:"

def generate_output(model, tokenizer, user_text):
    """
    Generate simplified output from model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        user_text: The user's input text
        
    Returns:
        The generated simplified text (without the input prompt)
    """
    # Format prompt for GPT-2
    prompt = format_prompt(user_text)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the newly generated tokens (skip the input prompt)
    generated_ids = output_ids[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_text = generated_text.split("\nUser:", 1)[0]
    generated_text = generated_text.split("\nAssistant:", 1)[0]
    
    return generated_text.strip()

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")
tokenizer = load_tokenizer()
print("[OK] Tokenizer loaded")

# ============================================================
# STEP A: GENERATE BASE MODEL OUTPUTS
# ============================================================

print("\n" + "-" * 60)
print("STEP 1: Base Model Inference")
print("-" * 60)

if cuda_available:
    try:
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
    except Exception as e:
        print(f"[WARN] 4-bit loading failed, using normal loading: {e}")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
else:
    # CPU loading without quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

print("[OK] Base model loaded")

base_outputs = []
for i, prompt in enumerate(prompts, 1):
    output = generate_output(base_model, tokenizer, prompt)
    base_outputs.append(output)
    print(f"  [{i}/{len(prompts)}] Generated base output")

print("[OK] Base model inference complete")

# Clean up
del base_model
if cuda_available:
    torch.cuda.empty_cache()
print("[OK] Base model cleared from memory")

# ============================================================
# STEP B: GENERATE FINE-TUNED MODEL OUTPUTS
# ============================================================

print("\n" + "-" * 60)
print("STEP 2: Fine-tuned Adapter Inference")
print("-" * 60)

if cuda_available:
    try:
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
    except Exception as e:
        print(f"[WARN] 4-bit loading failed, using normal loading: {e}")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
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

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()  # Set to evaluation mode

print("[OK] Base model + adapter loaded")

fine_tuned_outputs = []
for i, prompt in enumerate(prompts, 1):
    output = generate_output(model, tokenizer, prompt)
    fine_tuned_outputs.append(output)
    print(f"  [{i}/{len(prompts)}] Generated fine-tuned output")

print("[OK] Fine-tuned model inference complete")

# ============================================================
# STEP C: COMBINE AND SAVE RESULTS
# ============================================================

print("\n" + "-" * 60)
print("STEP 3: Saving Results")
print("-" * 60)

results = []
for prompt, base_output, fine_tuned_output in zip(
    prompts, base_outputs, fine_tuned_outputs
):
    results.append({
        "prompt": prompt,
        "base_output": base_output,
        "fine_tuned_output": fine_tuned_output,
    })

# Save to JSON
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"[OK] Results saved to: {RESULTS_FILE}")

# ============================================================
# PRINT RESULTS TO TERMINAL
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60 + "\n")

for i, result in enumerate(results, 1):
    print("=" * 60)
    print(f"PROMPT {i}")
    print("=" * 60)
    print(f"\nPrompt:\n{result['prompt']}\n")
    print(f"BASE MODEL OUTPUT:\n{result['base_output']}\n")
    print(f"FINE-TUNED MODEL OUTPUT:\n{result['fine_tuned_output']}\n")

# ============================================================
# FINAL MESSAGE
# ============================================================

print("=" * 60)
print("Comparison complete!")
print("=" * 60)
print(f"Results saved to: {RESULTS_FILE}")
print("\nNext step:")
print(f"  Run: python src/chat.py")
print("=" * 60 + "\n")
