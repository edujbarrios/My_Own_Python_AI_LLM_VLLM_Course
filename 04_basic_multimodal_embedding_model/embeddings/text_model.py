import numpy as np

class TextEmbedding:
    def __init__(self, vocab, dim, seed):
        np.random.seed(seed)
        self.vocab = vocab
        self.dim = dim
        self.embeddings = {
            word: np.random.uniform(-1, 1, dim)
            for word in vocab
        }

    def embed(self, tokens):
        return np.array([self.embeddings[token] for token in tokens if token in self.embeddings])
