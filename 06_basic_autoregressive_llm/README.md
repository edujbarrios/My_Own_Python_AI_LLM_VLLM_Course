# Part 6 – Autoregressive Text Generation (GPT-Style Transformer)  
LLM–VLLM AI COURSE by edujbarrios.com

This module extends the mini-transformer to become a basic autoregressive language model, similar to GPT. It uses a causal self-attention mask to prevent tokens from attending to future positions and trains on simple text to generate coherent sequences.

---

## Features

- Causal (masked) self-attention
- Autoregressive training objective
- Token prediction and text generation from scratch
- Sampling from trained model with or without a prompt

---

## How It Works

- The model is trained to predict the next token in a sequence.
- During generation, the model takes an initial prompt (or empty input) and generates tokens one-by-one, feeding its own outputs back in.

---

## Project Structure

```
06_autoregressive_llm/
├── model/
│   ├── embeddings.py
│   ├── attention.py
│   └── transformer.py
├── config/
│   └── config.py
├── data/
│   └── tiny_corpus.txt
├── train.py
├── generate.py
├── generate_example.py
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install torch numpy
```

### 2. Train the model

```bash
python train.py
```

This will save:
- `model.pth` – trained weights
- `token_to_id.pt`, `id_to_token.pt` – vocabulary

### 3. Generate text from a prompt

```bash
python generate_example.py
```

You can modify `prompt` inside `generate_example.py`.

---

## Example Output

```
Prompt: hello
Generated: hello how are you doing today i am fine thanks
```

---

## Educational Purpose

This is a simplified, educational implementation to help you understand how GPT-like models work under the hood. It is not optimized for large-scale training.

---

## Next Steps (Part 7)

- Multimodal transformer (Vision + Language)
- CLIP-style contrastive training

