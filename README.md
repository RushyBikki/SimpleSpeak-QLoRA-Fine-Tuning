# SimpleSpeak Qwen3 1.7B QLoRA Fine-Tuning

## Project Goal

SimpleSpeak is a small command-line proof-of-concept project that fine-tunes `Qwen/Qwen3-1.7B` to explain Computer Science concepts in simple language for 12-13 year old students.

The goal is for the model to explain the actual concept, use simple wording, keep helpful analogies, use short sentences, explain technical words when needed, and sound clear without being too childish.

## Model Used

- Base model: `Qwen/Qwen3-1.7B`
- Fine-tuning method: QLoRA with a LoRA adapter
- Adapter output: `adapters/qwen3-1.7b-simple-speak`

## Prompt Format

The dataset uses chat-style `messages` records:

```json
{"messages":[{"role":"user","content":"Can you explain binary search?"},{"role":"assistant","content":"Sure! Binary search is like looking for a word in a dictionary. You do not start from the first page and check every word. You open near the middle, see if your word comes before or after that page, and then only search that half. It keeps cutting the search area in half until it finds the answer."}]}
```

The scripts convert these messages with Qwen's tokenizer chat template:

```python
tokenizer.apply_chat_template(messages, tokenize=False)
```

## Thinking Mode

Qwen3 supports thinking-style outputs. This project is meant to produce short, simple explanations, not long reasoning traces, so the scripts try to call the chat template with:

```python
enable_thinking=False
```

If the installed tokenizer or Transformers version does not support that argument, the scripts fall back to the normal chat template call.

## LoRA And QLoRA

LoRA trains a small adapter instead of changing the whole model. This makes fine-tuning smaller and easier to save.

QLoRA loads the base model in 4-bit precision to reduce GPU memory use while training the adapter. The base model mostly stays frozen, and the adapter learns the SimpleSpeak explanation style.

## Folder Structure

```text
simple-speak-qlora/
├── data/
│   ├── train.jsonl
│   └── test_prompts.txt
├── src/
│   ├── train.py
│   ├── compare.py
│   └── chat.py
├── adapters/
├── results/
├── requirements.txt
└── README.md
```

## Dataset Format

Each line in `data/train.jsonl` is one JSON object with a `messages` field:

```json
{"messages":[{"role":"user","content":"Can you explain an API?"},{"role":"assistant","content":"Sure! An API is a way for two programs to talk to each other. One program sends a request, and another program sends back a response."}]}
```

## How To Run In Colab

Enable a GPU first:

```text
Runtime > Change runtime type > GPU
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the adapter:

```bash
python src/train.py
```

Compare the base model and fine-tuned adapter:

```bash
python src/compare.py
```

Start the interactive chat loop:

```bash
python src/chat.py
```

## Outputs

Training saves the LoRA adapter to:

```text
adapters/qwen3-1.7b-simple-speak
```

Comparison saves results to:

```text
results/results.json
```

## Note

This is a proof-of-concept project, not a production model. Results depend heavily on dataset quality, training settings, and the GPU available in Colab.
