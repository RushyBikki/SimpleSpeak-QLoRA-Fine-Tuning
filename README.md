# SimpleSpeak GPT-2 LoRA

A beginner-friendly proof-of-concept project that fine-tunes **GPT-2** using LoRA to explain technical concepts in simpler language suitable for 12–13 year old students.

---

## Project Goal

This project demonstrates how to fine-tune GPT-2 (a causal language model) with LoRA to teach the model to explain complex topics using:

- **Simpler wording**
- **Shorter sentences**
- **Less jargon**
- **Clear, beginner-friendly explanations**
- **Useful analogies**
- **Accurate but accessible information**

The output should sound like a helpful tutor explaining something to a young student—not childish or condescending.

---

## What This Project Demonstrates

- Loading models from Hugging Face
- LoRA/QLoRA fine-tuning on custom datasets
- PEFT (Parameter-Efficient Fine-Tuning) adapters
- JSONL dataset formatting with plain-text (not chat templates)
- Before/after model comparison
- Command-line inference with the fine-tuned adapter
- Working with 4-bit quantization on GPU (with CPU fallback for GPT-2)

---

## Model Used

- **Model**: `openai-community/gpt2`
- **Type**: Causal Language Model (not a chat model)
- **Size**: 124 Million parameters (small, fits easily on GPUs and CPUs)
- **License**: OpenAI/MIT (check Hugging Face model card)

---

## What is LoRA / QLoRA?

**LoRA** (Low-Rank Adaptation) is a technique that trains a small adapter instead of changing the entire model. This makes fine-tuning faster and requires less memory.

**QLoRA** adds 4-bit quantization on top of LoRA, which further reduces memory usage.

In this project:
- The base model weights stay frozen
- Only a small LoRA adapter (~1-2% of model size) is trained
- QLoRA (4-bit) loading is attempted if CUDA is available, but normal LoRA works on CPU too
- The adapter can be saved and loaded separately
- Multiple adapters can be created for different tasks

---

## Folder Structure

```
simple-speak-gpt2-lora/
│
├── data/
│   ├── train.jsonl          # Training examples (JSONL format)
│   └── test_prompts.txt     # Test prompts (one per line)
│
├── src/
│   ├── train.py             # Training script
│   ├── compare.py           # Base vs. fine-tuned comparison
│   └── chat.py              # Interactive chat interface
│
├── adapters/
│   └── gpt2-simple-speak/   # Saved LoRA adapter (created during training)
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

**Important**: GPT-2 is a causal language model, NOT a chat model. It doesn't have a chat template, so training examples are formatted as plain text.

The training data in `data/train.jsonl` uses JSONL format with messages objects, but each example is converted to plain text during training:

```json
{"messages":[{"role":"user","content":"Can you explain binary search?"},{"role":"assistant","content":"Sure! Binary search is like looking for a word in a dictionary. You do not start from the first page and check every word. You open near the middle, see if your word comes before or after that page, and then only search that half. It keeps cutting the search area in half until it finds the answer."}]}
```

During training, this is converted to:
```
User: Can you explain binary search?
Assistant: Sure! Binary search is like looking for a word in a dictionary...
```

For inference (chat.py and compare.py), prompts are formatted as:
```
User: Can you explain binary search?
Assistant:
```

Then the model generates the assistant's answer.

---

## How to Run

### Step 1: Setup in Colab (or Local)

**For Google Colab**:
1. **Upload the project** or clone from GitHub
2. **Enable GPU** (optional, but recommended):
   - Go to: `Runtime > Change runtime type > GPU`
   - Select GPU (Tesla T4 or better)

**For Local Machine**:
- No special setup needed; works on CPU too (slower, but works)

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First-time installation may take 3-5 minutes. `bitsandbytes` may not compile on CPU, but the code handles that fallback.

### Step 3: Train the Model

```bash
python src/train.py
```

- Training takes **5-15 minutes** on GPU, **30-60 minutes** on CPU
- Progress is printed every 5 training steps
- Adapter is saved to `adapters/gpt2-simple-speak/`

### Step 4: Compare Base vs. Fine-tuned

```bash
python src/compare.py
```

- Loads test prompts from `data/test_prompts.txt`
- Generates outputs from base GPT-2 and fine-tuned adapter
- Prints side-by-side comparison
- Saves results to `results/results.json`

### Step 5: Interactive Chat

```bash
python src/chat.py
```

- Type a question or concept
- The fine-tuned model explains it in simple language
- Type `quit`, `exit`, or `q` to stop

---

## Expected Output Files

### After Training
```
adapters/gpt2-simple-speak/
├── adapter_config.json       # LoRA configuration
├── adapter_model.bin         # LoRA weights (~500 KB - 2 MB)
├── tokenizer_config.json     # Tokenizer files
└── special_tokens_map.json   # Special tokens mapping
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
- Loads GPT-2 from `openai-community/gpt2`
- Sets up LoRA adapter with r=8, alpha=16, target modules `["c_attn", "c_proj", "c_fc"]`
- Attempts 4-bit quantization (QLoRA) if CUDA is available; falls back to normal LoRA on CPU
- Trains on `data/train.jsonl` for 3 epochs with plain-text format (`User: ... Assistant: ...`)
- Saves adapter to `adapters/gpt2-simple-speak/`

**Run**: `python src/train.py`

**Time**: 
- GPU (4-bit): 5-15 minutes
- GPU (normal): 10-20 minutes
- CPU: 30-60 minutes

### `src/compare.py`
- Loads base GPT-2, generates outputs for test prompts
- Loads fine-tuned adapter, generates outputs for same prompts
- Prints side-by-side comparison in terminal
- Saves structured results to `results/results.json`
- Handles 4-bit loading failures gracefully

**Run**: `python src/compare.py`

**Time**: 2-5 minutes on GPU, 5-15 minutes on CPU

### `src/chat.py`
### `src/chat.py`
- Loads GPT-2 and the fine-tuned LoRA adapter
- Provides interactive terminal loop
- User types a question → Model explains it in simple language → See the answer
- Type `quit`, `exit`, or `q` to stop
- Handles 4-bit loading failures gracefully

**Run**: `python src/chat.py`

**Time**: Loading takes 10-30 seconds; inference is real-time

---

## Common Issues & Fixes

### Issue 1: "Adapter Not Found" Error
**Error**: `ERROR: Adapter not found. Please run python src/train.py first.`

**Fix**: The `compare.py` and `chat.py` scripts need the adapter to be trained first.
```bash
python src/train.py
```
Then run:
```bash
python src/compare.py
python src/chat.py
```

---

### Issue 2: GPU Memory Issues
**Error**: `CUDA out of memory` or `RuntimeError: CUDA out of memory`

**Fixes**:
1. **Restart the Colab runtime**: `Runtime > Restart runtime`
2. **Use smaller batch size** (already set to 1 in config)
3. **Reduce gradient accumulation**: Edit `train.py`, change `gradient_accumulation_steps` from 8 to 4
4. **Run on CPU** (slower but works for GPT-2): The code auto-detects GPU; if you want CPU, just run without GPU enabled

---

### Issue 3: "`bitsandbytes` or `peft` Installation Fails"
**Error**: `RuntimeError: (libcuda.so.1...)` or other import errors

**Fix**: The code is designed to handle this! It will:
1. Try 4-bit loading if possible
2. Fall back to normal LoRA if 4-bit fails
3. Work on CPU if needed

Just try running the script again:
```bash
pip install --upgrade transformers peft
```

If still having issues, the code will gracefully use CPU or normal LoRA.

---

### Issue 4: "TRL Version Incompatibility"
**Error**: `SFTConfig not found`, `unexpected keyword argument 'max_seq_length'`, or similar

**Fix**: The code handles multiple TRL versions automatically. If issues persist:
```bash
pip install --upgrade trl transformers
```

---

### Issue 5: Very Slow Inference
**Symptom**: `compare.py` or `chat.py` runs on CPU and takes several minutes per prompt

**Cause**: GPU not enabled or not recognized

**Fix**: 
1. Verify GPU is enabled (in Colab: `Runtime > Change runtime type > GPU`)
2. In Colab, run `!nvidia-smi` to confirm GPU is available
3. Restart runtime and retry

---

## Training Parameters

The training script uses these settings for GPT-2:

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | GPT-2 (124M) | Small, fast, works on CPU/GPU |
| Epochs | 3 | Adjust based on dataset size |
| Batch Size | 1 | Fits easily on any GPU; GPU will handle it |
| Gradient Accumulation | 8 | Simulates batch size of 8 |
| Learning Rate | 2e-4 | Standard for LoRA |
| Max Sequence Length | 512 | Balances quality and speed |
| LoRA Rank (r) | 8 | Appropriate for 124M model |
| LoRA Alpha | 16 | Standard ratio (alpha = 2 * r) |
| LoRA Dropout | 0.05 | Prevents overfitting |
| Target Modules | `c_attn, c_proj, c_fc` | GPT-2 attention and MLP layers |
| Quantization | 4-bit (QLoRA) if CUDA | Else normal LoRA |

---

## Customization

### To adjust training:
Edit `src/train.py`:
- Change `num_train_epochs` (try 5 for more training, 1 for quick test)
- Change `learning_rate` (try 1e-4 for lighter tuning, 3e-4 for stronger)
- Change `gradient_accumulation_steps` (increase for larger effective batch size)

### To adjust generation:
Edit `src/compare.py` or `src/chat.py`:
- Change `max_new_tokens` (currently 120-150; increase for longer responses)
- Change `temperature` (currently 0.7; lower = more deterministic, higher = more random)
- Change `top_p` (currently 0.9; nucleus sampling threshold)

### To use different data:
1. Create `data/train.jsonl` with your training examples (JSONL format with `messages` objects)
2. Create `data/test_prompts.txt` with test prompts (one per line, plain text)
3. Run training as usual

---

## Notes

- This is a **proof-of-concept** project, not a production model
- **Why GPT-2?** It's small (124M), open-source, and runs on any hardware (including CPU)
- Model output quality depends heavily on training data quality and diversity
- Results improve significantly with:
  - More and higher-quality training examples
  - More training epochs (try 5-10 for small datasets)
  - Larger models (but they need more memory)
- LoRA is very parameter-efficient (~0.5% of model size)
- GPT-2 is not instruction-tuned like newer models, so output quality will vary
- Consider starting with GPT-2, then experimenting with larger models like Phi-2 or Llama

---

## Next Steps for Improvement

1. **Collect more training data**: 100+ examples improve generalization significantly
2. **Use a larger model**: Try Phi-2 (2.7B), Mistral (7B), or Llama-2 (7B+) for better quality
3. **Experiment with hyperparameters**: Try different learning rates (1e-4, 3e-4), LoRA ranks (r=16), etc.
4. **Evaluate systematically**: Create a test set and measure quality improvements
5. **Add domain-specific data**: Train on examples from your target domain
6. **Deploy locally**: Use `ollama` for local serving of the fine-tuned model

---

## References

- [GPT-2 Model Card](https://huggingface.co/openai-community/gpt2)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Ollama (Local Inference)](https://ollama.ai)

---

**Happy fine-tuning!** 🚀

