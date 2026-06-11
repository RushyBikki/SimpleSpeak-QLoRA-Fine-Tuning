"""
QLoRA fine-tuning script for Qwen2.5-0.5B-Instruct on simple-speak task.

This script trains a LoRA adapter to rewrite technical text in simpler words.
The adapter is saved to adapters/qwen-simple-speak and can be used with compare.py or chat.py.
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

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "data/train.jsonl"
ADAPTER_DIR = "adapters/qwen-simple-speak"
OUTPUT_DIR = "results/training_output"

# Disable Weights & Biases reporting
os.environ["WANDB_DISABLED"] = "true"

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================================
# SETUP AND VALIDATION
# ============================================================

print("=" * 60)
print("SimpleSpeak QLoRA Training Script")
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

print(f"✓ Training data found: {TRAIN_FILE}")

# Check for CUDA/GPU
if not torch.cuda.is_available():
    print("\n" + "!" * 60)
    print("ERROR: CUDA/GPU was not found.")
    print("=" * 60)
    print("This QLoRA training script is intended to run in Google Colab with GPU enabled.")
    print("\nIn Colab, go to: Runtime > Change runtime type > GPU")
    print("=" * 60 + "\n")
    exit(1)

print(f"✓ CUDA available. Device: {torch.cuda.get_device_name(0)}")
print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set padding side to "right"
tokenizer.padding_side = "right"
print("✓ Tokenizer loaded")

# ============================================================
# LOAD MODEL WITH 4-BIT QUANTIZATION
# ============================================================

print("\nLoading base model in 4-bit...")

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

# Disable cache for training
model.config.use_cache = False
print("✓ Model loaded in 4-bit")

# ============================================================
# PREPARE FOR KBIT TRAINING
# ============================================================

print("\nPreparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)
print("✓ Model prepared for k-bit training")

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
    target_modules="all-linear",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✓ LoRA adapter added")

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

print(f"✓ Dataset loaded: {len(dataset)} examples")

# Convert messages to text using chat template
def format_chat_template(example):
    """Convert messages to text using Qwen's chat template."""
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = dataset.map(format_chat_template)
print("✓ Dataset formatted with chat template")

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
    "optim": "paged_adamw_8bit",
    "logging_steps": 5,
    "save_strategy": "epoch",
    "fp16": True,
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
print("✓ Training complete")

# ============================================================
# SAVE ADAPTER
# ============================================================

print("\nSaving adapter...")

# Save only the LoRA adapter (not the base model)
model.save_pretrained(ADAPTER_DIR)

# Save tokenizer to adapter folder
tokenizer.save_pretrained(ADAPTER_DIR)

print(f"✓ Adapter saved to: {ADAPTER_DIR}")
print(f"✓ Tokenizer saved to: {ADAPTER_DIR}")

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
