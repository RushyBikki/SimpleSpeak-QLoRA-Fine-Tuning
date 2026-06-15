# SimpleSpeak Qwen3 1.7B QLoRA Fine-Tuning

## Project Goal

SimpleSpeak is a small command-line proof-of-concept project that fine-tunes `Qwen/Qwen3-1.7B` to explain Computer Science concepts in simple language for 12-13 year old students.

The goal is for the model to explain the actual concept, use simple wording, keep helpful analogies, use short sentences, explain technical words when needed, and sound clear without being too childish.

## Model Used
- Base model: `Qwen/Qwen3-1.7B`
- Fine-tuning method: QLoRA with a LoRA adapter
- GPU Service: Google Colab
- Adapter output: `adapters/qwen3-1.7b-simple-speak`

## Prompt Format
The dataset uses chat-style `messages` records:

```json
{"messages":[{"role":"user","content":"Can you explain binary search?"},{"role":"assistant","content":"Sure! Binary search is like looking for a word in a dictionary. You do not start from the first page and check every word. You open near the middle, see if your word comes before or after that page, and then only search that half. It keeps cutting the search area in half until it finds the answer."}]}


## LoRA And QLoRA
LoRA trains a small adapter instead of changing the whole model. This makes fine-tuning smaller and easier to save.

QLoRA loads the base model in 4-bit precision to reduce GPU memory use while training the adapter. The base model mostly stays frozen, and the adapter learns the SimpleSpeak explanation style.

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