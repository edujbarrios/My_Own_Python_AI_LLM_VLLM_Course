# Python LLM–VLLM AI COURSE  
**by edujbarrios.com**  
_A project-based curriculum to understand and replicate the architecture of Large Language Models (LLMs) and Visual Language Models (VLLMs) from scratch._

---

## Overview

**Python LLM–VLLM AI COURSE** is a hands-on Python course designed to explore how modern AI models—particularly LLMs and VLLMs—work internally.

Instead of relying on pre-trained black-box models, this course walks you through implementing the **core structures step by step**, including tokenizers, text/image embeddings, multimodal systems, and transformers.

By the end of the course, you will understand the architecture and design logic behind systems like **GPT**, **BERT**, **ViT**, and **CLIP**.

---

## Why Python?

Python is the standard language for AI and machine learning development due to:

- Extensive scientific libraries (NumPy, PyTorch, PIL)
- Easy syntax for prototyping deep models
- Large community and ecosystem
- GPU and distributed computing integration

Its simplicity and flexibility make it the ideal choice to implement low-level AI architectures from scratch.

---

## Core Concepts Covered

- **Tokens**: Units of text (words or subwords) used by language models.
- **Embeddings**: Vector representations of tokens, images, and other modalities.
- **Patches**: Image segments (e.g., 16×16 pixels) used in visual models.
- **Multimodal Learning**: Combining text and image into a unified representation.
- **Transformers**: The fundamental architecture behind all modern LLMs and VLLMs.
- **Parametrization:** A parametrization based approach while programming, which allows to easily change settings while testing.

---

## What Has Been Built So Far

| Part | Title                                | Description                                                              |
|------|--------------------------------------|--------------------------------------------------------------------------|
| 1    | Basic Text Tokenizer                 | Tokenize raw text and build token-to-index vocab from scratch.           |
| 2    | Word Embedding Engine                | Simulate learned word vectors using NumPy.                               |
| 3    | Image Patch Embedding Generator      | Split images into patches and project them into embedding space.         |
| 4    | Multimodal Text–Image Matching       | Embed caption and image, then compute cosine similarity between them.    |
| 5    | Mini Transformer Encoder             | Core architecture behind LLMs, implemented using PyTorch.                |
| 6    | Basic Transformer for Language Generation (LLMs)   | Build a GPT-style transformer that generates text autoregressively.        |

---

## What’s Coming Next

| Part | Title                                              | Description                                                                 |
|------|----------------------------------------------------|-----------------------------------------------------------------------------|
| 7    | Transformer Applied to VLLMs                       | Extend the transformer to process image patches + positional encodings.    |
| 8    | Fine-tuning LLMs                                   | Apply training to pretrained LLM-like architecture on custom text data.    |
| 9    | Fine-tuning VLLMs                                  | Perform multimodal fine-tuning on image–text pairs using contrastive loss. |
| 10   | Building an LLM from Scratch with Vision Support   | Assemble a full language + vision model with encoder/decoder blocks.       |

---

## Learning Goals

- Understand what happens inside a transformer.
- Learn the difference between raw input, tokens, embeddings, and attention.
- Build working educational-scale versions of real architectures.
- Prepare for using or training real models like GPT, ViT, or CLIP.

---

## License

This course is developed and maintained by **edujbarrios.com**  

For questions or collaboration, feel free to contact the author.
