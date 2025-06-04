import numpy as np

class PatchEmbeddingModel:
    def __init__(self, patch_size, embedding_dim, seed):
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.seed = seed