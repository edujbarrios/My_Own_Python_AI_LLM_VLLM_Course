from config.config import IMAGE_PATH, PATCH_SIZE, EMBEDDING_DIM, SEED, RESIZE_TO
from embeddings.utils import load_image
from embeddings.patch_model import PatchEmbeddingModel

def main():
    image = load_image(IMAGE_PATH, RESIZE_TO)

    model = PatchEmbeddingModel(PATCH_SIZE, EMBEDDING_DIM, SEED)
    patches = model.extract_patches(image)
    embeddings = model.project_patches(patches)

    print(f"Image shape: {image.shape}")
    print(f"Number of patches: {len(patches)}")
    print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    main()
