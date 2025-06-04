import numpy as np

class PatchEmbeddingModel:
    def __init__(self, patch_size, embedding_dim, seed):
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.seed = seed

    def extract_patches(self, image):
        h, w, c = image.shape
        ph = self.patch_size
        assert h % ph == 0 and w % ph == 0

        patches = []
        for i in range(0, h, ph):
            for j in range(0, w, ph):
                patch = image[i:i+ph, j:j+ph, :].reshape(-1)
                patches.append(patch)
        return np.array(patches)

    def project_patches(self, patches):
        np.random.seed(self.seed)
        projection_matrix = np.random.randn(patches.shape[1], self.embedding_dim)
        return patches @ projection_matrix
