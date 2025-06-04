import numpy as np

class EmbeddingModel:
    def __init__(self, vocab, embedding_dim=50, seed=42):
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self):
        np.random.seed(self.seed)
        vocab_size = len(self.vocab)
        return {
            token: np.random.uniform(-0.5, 0.5, self.embedding_dim)
            for token in self.vocab
        }

    def get_vector(self, token):
        return self.embeddings.get(token)

    def get_embedding_matrix(self):
        return np.array([self.embeddings[token] for token in self.vocab])
