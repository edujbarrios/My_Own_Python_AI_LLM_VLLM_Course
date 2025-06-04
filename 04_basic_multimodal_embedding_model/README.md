# Part 4 – Multimodal Embedding Matching (Image + Text)  
MY OWN PYTHON AI LLM-VLLM COURSE BY https://edujbarrios.com

This project simulates a basic multimodal embedding system that evaluates whether a text caption matches a given image. It achieves this by embedding both the image and the text into a shared vector space, and then comparing them using cosine similarity.

---

## 🎯 Objective

Build a simple, interpretable pipeline that:

1. Extracts features from an image (as patch embeddings)
2. Embeds tokens from a descriptive caption
3. Combines both into a common vector representation
4. Measures their similarity
5. Decides whether the text accurately describes the image

---

## 🧱 Core Concepts

- **Token Embedding**: Each word in the caption is mapped to a learned vector.
- **Patch Embedding**: The image is divided into fixed-size patches, flattened and projected into vectors (mimicking ViT).
- **Cosine Similarity**: Measures similarity between image and text embeddings.
- **Threshold Decision**: If similarity ≥ threshold, the text is considered a valid description of the image.

---