"""
QLoRA fine-tuning script for Qwen3 1.7B on SimpleSpeak.

The script trains a small LoRA adapter so Qwen/Qwen3-1.7B explains
Computer Science concepts in simple language for 12-13 year old students.
"""

import os
import sys
from inspect import signature
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None


MODEL_NAME = "Qwen/Qwen3-1.7B"
TRAIN_FILE = "data/train.jsonl"
ADAPTER_DIR = "adapters/qwen3-1.7b-simple-speak"
TRAINING_OUTPUT_DIR = "results/training_output"

os.environ["WANDB_DISABLED"] = "true"
torch.manual_seed(42)


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


def build_training_args():
    """Create SFT training arguments while tolerating TRL version differences."""
    config_kwargs = {
        "output_dir": TRAINING_OUTPUT_DIR,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "fp16": True,
        "report_to": "none",
        "optim": "paged_adamw_8bit",
    }

    config_cls = SFTConfig or TrainingArguments
    supported_params = signature(config_cls).parameters

    if SFTConfig and "max_length" in supported_params:
        config_kwargs["max_length"] = 512
    if SFTConfig and "dataset_text_field" in supported_params:
        config_kwargs["dataset_text_field"] = "text"

    filtered_kwargs = {
        key: value for key, value in config_kwargs.items() if key in supported_params
    }
    return config_cls(**filtered_kwargs)


print("=" * 60)
print("SimpleSpeak Qwen3 1.7B QLoRA Training")
print("=" * 60)

Path("adapters").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(parents=True, exist_ok=True)
Path(TRAINING_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

if not Path(TRAIN_FILE).exists():
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}\n"
        "Please ensure data/train.jsonl exists."
    )

if not torch.cuda.is_available():
    print("\nCUDA/GPU was not found.")
    print("This QLoRA training script is intended to run in Google Colab with GPU enabled.")
    print("In Colab, go to Runtime > Change runtime type > GPU.")
    sys.exit(1)

print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("[OK] Tokenizer loaded")

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
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = False
print("[OK] Model loaded")

print("\nPreparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)
print("[OK] Model prepared")

print("\nAdding LoRA adapter...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
if hasattr(model, "print_trainable_parameters"):
    model.print_trainable_parameters()
print("[OK] LoRA adapter added")

print("\nLoading dataset...")
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
if "messages" not in dataset.column_names:
    raise ValueError("Dataset must contain a 'messages' field.")


def validate_messages(example):
    if not example.get("messages"):
        raise ValueError("Each training example must contain a non-empty 'messages' list.")
    return example


def format_training_example(example):
    text = apply_qwen_chat_template(
        tokenizer,
        example["messages"],
        add_generation_prompt=False,
    )
    return {"text": text}


dataset = dataset.map(validate_messages)
dataset = dataset.map(
    format_training_example,
    remove_columns=[col for col in dataset.column_names if col != "text"],
)
print(f"[OK] Dataset formatted: {len(dataset)} examples")

print("\nConfiguring training...")
training_args = build_training_args()

trainer_kwargs = {
    "model": model,
    "train_dataset": dataset,
    "args": training_args,
}

trainer_params = signature(SFTTrainer.__init__).parameters
if "processing_class" in trainer_params:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in trainer_params:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = SFTTrainer(**trainer_kwargs)
print("[OK] Training config ready")

print("\nStarting training...")
trainer.train()

print("\nSaving adapter...")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

print("\nTraining complete.")
print(f"Adapter saved to: {ADAPTER_DIR}")
