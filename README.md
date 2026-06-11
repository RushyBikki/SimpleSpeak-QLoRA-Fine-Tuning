# SimpleSpeak QLoRA

A beginner-friendly proof-of-concept project that fine-tunes **Qwen/Qwen2.5-0.5B-Instruct** using QLoRA to rewrite technical or complicated text into simpler, easier-to-understand language.

---

## Project Goal

This project demonstrates how to fine-tune a small language model (Qwen2.5-0.5B) to simplify technical text. After training, the model rewrites complicated content using:

- **Simpler wording**
- **Shorter sentences**
- **Less jargon**
- **Clear, beginner-friendly explanations**
- **The same original meaning**
- **A professional but easy-to-understand tone**

The output should sound like a helpful tutor explaining something clearly—not childish or condescending.

---

## What This Project Demonstrates

- Loading models from Hugging Face
- QLoRA/LoRA fine-tuning on custom datasets
- PEFT (Parameter-Efficient Fine-Tuning) adapters
- JSONL dataset formatting
- Before/after model comparison
- Command-line inference with the fine-tuned adapter
- Working with 4-bit quantization on GPU

---

## Model Used

- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Size**: 0.5 Billion parameters (very small, fits on standard GPUs)
- **License**: Check Hugging Face model card for license details

---

## What is LoRA / QLoRA?

**LoRA** (Low-Rank Adaptation) is a technique that trains a small adapter instead of changing the entire model. This makes fine-tuning faster and requires less memory.

**QLoRA** adds 4-bit quantization on top of LoRA, which further reduces memory usage. This allows training on smaller GPUs (or even free Colab GPUs).

In this project:
- The base model weights stay frozen
- Only a small LoRA adapter (~2-3% of model size) is trained
- The adapter can be saved and loaded separately
- Multiple adapters can be created for different tasks

---

## Folder Structure

```
simple-speak-qlora/
│
├── data/
│   ├── train.jsonl          # Training examples (conversational format)
│   └── test_prompts.txt     # Test prompts (one per line)
│
├── src/
│   ├── train.py             # Training script
│   ├── compare.py           # Base vs. fine-tuned comparison
│   └── chat.py              # Interactive chat interface
│
├── adapters/
│   └── qwen-simple-speak/   # Saved LoRA adapter (created during training)
│
├── results/
│   ├── training_output/     # Training logs and checkpoints
│   └── results.json         # Comparison results (created by compare.py)
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## Dataset Format

The training data in `data/train.jsonl` uses conversational format with one JSON object per line:

```json
{"messages":[{"role":"user","content":"Rewrite this in simpler words: Fine-tuning updates a pretrained model on a smaller task-specific dataset."},{"role":"assistant","content":"Fine-tuning means taking an AI model that already knows a lot and training it a little more for one specific job."}]}
```

Each line contains:
- **user message**: The instruction to simplify text
- **assistant message**: The simplified version

---

## How to Run (Google Colab)

### Step 1: Setup in Colab

1. **Upload the project** to Colab or clone from GitHub
2. **Enable GPU**:
   - Go to: `Runtime > Change runtime type > GPU`
   - Select GPU (Tesla T4 or better)

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First-time installation may take 3-5 minutes. `bitsandbytes` compilation is normal.

### Step 3: Train the Model

```bash
python src/train.py
```

- Training takes **10-30 minutes** depending on GPU and dataset size
- Progress is printed every 5 training steps
- Adapter is saved to `adapters/qwen-simple-speak/`

### Step 4: Compare Base vs. Fine-tuned

```bash
python src/compare.py
```

- Loads test prompts from `data/test_prompts.txt`
- Generates outputs from base model and fine-tuned model
- Prints side-by-side comparison
- Saves results to `results/results.json`

### Step 5: Interactive Chat

```bash
python src/chat.py
```

- Type text you want simplified
- The fine-tuned model rewrites it
- Type `quit` to exit

---

## Expected Output Files

### After Training
```
adapters/qwen-simple-speak/
├── adapter_config.json       # LoRA configuration
├── adapter_model.bin         # LoRA weights (~2-3 MB)
└── tokenizer_config.json     # Tokenizer files
```

### After Comparison
```
results/results.json          # JSON file with prompt/output pairs
```

Example structure:
```json
[
  {
    "prompt": "Gradient descent is an optimization algorithm...",
    "base_output": "Gradient descent is a way to change a model...",
    "fine_tuned_output": "Gradient descent is a method to slowly improve a model..."
  }
]
```

---

## Scripts Overview

### `src/train.py`
- Loads Qwen2.5-0.5B-Instruct
- Sets up QLoRA with 4-bit quantization
- Adds LoRA adapter with r=8, alpha=16
- Trains on `data/train.jsonl` for 3 epochs
- Saves adapter to `adapters/qwen-simple-speak/`

**Run**: `python src/train.py`

**Time**: 10-30 minutes on GPU

### `src/compare.py`
- Loads base model, generates outputs for test prompts
- Loads fine-tuned adapter, generates outputs for same prompts
- Prints side-by-side comparison in terminal
- Saves structured results to `results/results.json`

**Run**: `python src/compare.py`

**Time**: 2-5 minutes depending on number of test prompts

### `src/chat.py`
- Loads the fine-tuned adapter
- Provides interactive terminal loop
- User types text → Model rewrites it → User sees simplified version
- Type `quit`, `exit`, or `q` to stop

**Run**: `python src/chat.py`

---

## Common Google Colab Issues & Fixes

### Issue 1: "GPU Not Enabled"
**Error**: `CUDA/GPU was not found`

**Fix**:
1. Go to: `Runtime > Change runtime type`
2. Select `GPU` from the dropdown
3. Click `SAVE`
4. Restart the notebook

---

### Issue 2: "`bitsandbytes` Installation Fails"
**Error**: `RuntimeError: (libcuda.so.1: cannot open shared object file)`

**Fix**:
```bash
pip install -U bitsandbytes
# If still fails, try pre-compiled binary:
pip install --no-deps bitsandbytes
```

---

### Issue 3: "Out of Memory" During Training
**Error**: `CUDA out of memory`

**Fixes** (in order):
1. Reduce batch size (already set to 1)
2. Reduce the max sequence length in `src/train.py` from 512 to 256
3. Reduce dataset size for testing
4. Use a different GPU in Colab (Runtime > Change runtime type)

---

### Issue 4: "Adapter Not Found"
**Error**: `Adapter not found. Please run python src/train.py first.`

**Fix**: Ensure `src/train.py` completed successfully and adapter was saved.

---

### Issue 5: "TRL Version Incompatibility"
**Error**: `SFTConfig not found`, `unexpected keyword argument 'max_seq_length'`, or similar

**Fix**: The code handles multiple TRL versions automatically. If issues persist:
```bash
pip install --upgrade trl transformers
```

---

### Issue 6: Very Slow Inference
**Symptom**: `compare.py` or `chat.py` runs on CPU and takes several minutes per prompt

**Cause**: GPU not enabled or not recognized

**Fix**: 
1. Verify GPU is enabled (see Issue 1)
2. In Colab, run `!nvidia-smi` to confirm GPU
3. Restart runtime and retry

---

## Training Parameters

The training script uses these settings:

| Parameter | Value | Reason |
|-----------|-------|--------|
| Epochs | 3 | Small model, small dataset (adjust if needed) |
| Batch Size | 1 | Fits in Colab GPU memory |
| Gradient Accumulation | 8 | Simulates larger batch size |
| Learning Rate | 2e-4 | Standard for LoRA fine-tuning |
| Max Sequence Length | 512 | Balances quality and speed |
| LoRA Rank (r) | 8 | Small rank for small model |
| LoRA Alpha | 16 | Standard ratio (alpha = 2 * r) |

---

## Customization

### To adjust training:
Edit `src/train.py`:
- Change `num_train_epochs` (e.g., 5 for longer training)
- Change `per_device_train_batch_size` (if GPU memory allows)
- Change `learning_rate` (try 1e-4 or 3e-4)

### To adjust generation:
Edit `src/compare.py` or `src/chat.py`:
- Change `max_new_tokens` (more tokens = longer responses)
- Change `temperature` (lower = more deterministic, 0.3 is quite low)
- Change `top_p` (nucleus sampling threshold)

### To use different data:
Replace `data/train.jsonl` and `data/test_prompts.txt` with your own examples in the same format.

---

## Notes

- This is a **proof-of-concept** project, not a production model
- The model output quality depends heavily on training data quality
- Results improve with more training examples and longer training
- The 0.5B model is small—for better results, use larger models (3B, 7B) but they require more GPU memory
- Fine-tuning on task-specific data typically outperforms prompting alone

---

## Next Steps for Improvement

1. **Collect more training data**: More diverse examples improve generalization
2. **Use a larger model**: 3B or 7B parameters would improve quality
3. **Experiment with hyperparameters**: Try different learning rates, LoRA ranks, etc.
4. **Evaluate systematically**: Use metrics like BLEU, ROUGE, or human feedback
5. **Try different base models**: Llama, Mistral, etc.
6. **Deploy to production**: Use frameworks like vLLM or Ollama for serving

---

## References

- [Qwen Model Card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

**Enjoy fine-tuning!** 🚀

