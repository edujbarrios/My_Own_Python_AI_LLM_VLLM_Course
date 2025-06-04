o# Part 5 – Mini Transformer LLM (Autoregressive)  
LLM–VLLM AI COURSE by edujbarrios.com

This module builds a simplified Transformer Encoder from scratch and uses it to generate text autoregressively, simulating how GPT-style LLMs work.

---

## Features

- Token embedding + positional encoding
- Multi-head self-attention
- LayerNorm + Feed Forward blocks
- Autoregressive token generation
- End-to-end training with PyTorch

---

## How to Use

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
- `token_to_id.pt`, `id_to_token.pt` – vocab mappings

### 3. Generate from prompt

```bash
python generate_example.py
```

### Example output:

```
Generated: hello how are you doing today i am fine
```

---

## Notes

This is a minimal, educational version of a Transformer for LLMs. For scaling to large datasets and training with GPUs, extend this foundation into Part 6.
