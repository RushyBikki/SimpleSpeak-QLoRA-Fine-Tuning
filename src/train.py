"""
LoRA fine-tuning script for GPT-2 on simple-speak task.

This script trains a LoRA adapter to explain technical concepts in simpler words
suitable for 12-13 year old students. The adapter is saved to adapters/gpt2-simple-speak
and can be used with compare.py or chat.py.
"""

import os
import json
from inspect import signature
import torch
from pathlib import Path

# Hugging Face imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai-community/gpt2"
TRAIN_FILE = "data/train.jsonl"
ADAPTER_DIR = "adapters/gpt2-simple-speak"
OUTPUT_DIR = "results/training_output"

# Disable Weights & Biases reporting
os.environ["WANDB_DISABLED"] = "true"

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================================
# SETUP AND VALIDATION
# ============================================================

print("=" * 60)
print("SimpleSpeak GPT-2 LoRA Training Script")
print("=" * 60)

# Create necessary directories
Path(ADAPTER_DIR).parent.mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).parent.mkdir(parents=True, exist_ok=True)

# Check if training data exists
if not Path(TRAIN_FILE).exists():
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}\n"
        "Please ensure data/train.jsonl exists with training examples."
    )

print(f"[OK] Training data found: {TRAIN_FILE}")

# Check for CUDA/GPU (optional for GPT-2)
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"[OK] CUDA available. Device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("[WARN] CUDA not available. Will use CPU (training will be slower).")

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# Ensure tokenizer has a pad token (GPT-2 doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set padding side to "right"
tokenizer.padding_side = "right"
print("[OK] Tokenizer loaded")

# ============================================================
# LOAD MODEL WITH 4-BIT QUANTIZATION (if available)
# ============================================================

print("\nLoading base model...")
model = None
using_4bit = False

# Try 4-bit quantization if CUDA is available
if cuda_available:
    try:
        print("Attempting 4-bit quantization...")
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
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        using_4bit = True
        print("[OK] Model loaded in 4-bit")
    except Exception as e:
        print(f"[WARN] 4-bit loading failed: {e}")
        print("Falling back to normal LoRA loading...")
        model = None

# Fall back to normal loading if 4-bit failed or no CUDA
if model is None:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32 if not cuda_available else torch.float16,
        device_map="auto" if cuda_available else "cpu",
        trust_remote_code=True,
    )
    print("[OK] Model loaded (normal LoRA)")

# Disable cache for training
model.config.use_cache = False
model.config.pad_token_id = tokenizer.eos_token_id

# ============================================================
# PREPARE FOR KBIT TRAINING (if using 4-bit)
# ============================================================

if using_4bit:
    print("\nPreparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    print("[OK] Model prepared for k-bit training")

# ============================================================
# ADD LORA ADAPTER
# ============================================================

print("\nSetting up LoRA adapter...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "c_fc"],
    fan_in_fan_out=True,
)

model = get_peft_model(model, lora_config)

# Keep trainable adapter weights in FP32 (for 4-bit training stability)
if using_4bit:
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

model.print_trainable_parameters()
print("[OK] LoRA adapter added")

# ============================================================
# LOAD AND PREPROCESS DATASET
# ============================================================

print("\nLoading dataset...")
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

# Validate dataset structure
if "messages" not in dataset.column_names:
    raise ValueError(
        f"Dataset must contain 'messages' field.\n"
        f"Found columns: {dataset.column_names}\n"
        f"Expected format:\n"
        f'{{"messages":[{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}]}}'
    )

print(f"[OK] Dataset loaded: {len(dataset)} examples")

# Convert messages to plain text (no chat template for GPT-2)
def format_training_example(example):
    """Convert messages to plain text format for GPT-2."""
    messages = example["messages"]
    user_message = ""
    assistant_message = ""
    
    for message in messages:
        if message.get("role") == "user":
            user_message = message.get("content", "")
        elif message.get("role") == "assistant":
            assistant_message = message.get("content", "")
    
    return {
        "text": f"User: {user_message}\nAssistant: {assistant_message}"
    }

dataset = dataset.map(format_training_example)
print("[OK] Dataset formatted as plain text")

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print("\nConfiguring training...")

training_config_kwargs = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "linear",
    "optim": "paged_adamw_8bit" if using_4bit else "adamw_torch",
    "logging_steps": 5,
    "save_strategy": "epoch",
    "max_grad_norm": 0.0,
    "fp16": cuda_available,
    "bf16": False,
    "report_to": "none",
    "seed": 42,
}

# TRL renamed max_seq_length to max_length in newer releases.
sft_config_params = signature(SFTConfig).parameters
if "max_length" in sft_config_params:
    training_config_kwargs["max_length"] = 512
elif "max_seq_length" in sft_config_params:
    training_config_kwargs["max_seq_length"] = 512
else:
    print("Warning: This TRL version does not expose max_length/max_seq_length.")

training_config = SFTConfig(**training_config_kwargs)

# Initialize trainer
trainer_kwargs = {
    "model": model,
    "train_dataset": dataset,
    "args": training_config,
}

# Newer TRL uses processing_class; older releases used tokenizer.
sft_trainer_params = signature(SFTTrainer.__init__).parameters
if "processing_class" in sft_trainer_params:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in sft_trainer_params:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = SFTTrainer(**trainer_kwargs)

print("✓ Training config ready")

# ============================================================
# TRAIN MODEL
# ============================================================

print("\nStarting training...")
print("-" * 60)

trainer.train()

print("-" * 60)
print("[OK] Training complete")

# ============================================================
# SAVE ADAPTER
# ============================================================

print("\nSaving adapter...")

# Save only the LoRA adapter (not the base model)
model.save_pretrained(ADAPTER_DIR)

# Save tokenizer to adapter folder
tokenizer.save_pretrained(ADAPTER_DIR)

print(f"[OK] Adapter saved to: {ADAPTER_DIR}")
print(f"[OK] Tokenizer saved to: {ADAPTER_DIR}")

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
print(f"Adapter location: {ADAPTER_DIR}/")
print("\nNext steps:")
print(f"  1. Run: python src/compare.py")
print(f"  2. Run: python src/chat.py")
print("=" * 60 + "\n")
