from config.config import *
from embeddings.text_model import TextEmbedding
from embeddings.image_model import PatchEmbeddingModel
from embeddings.fusion import fuse_embeddings, compare_embeddings
from embeddings.utils import load_image
import string

def clean_tokens(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()

def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    raw_text = read_text(TEXT_PATH)
    tokens = clean_tokens(raw_text)
    vocab = list(set(tokens))

    text_model = TextEmbedding(vocab, TEXT_EMBED_DIM, SEED)
    text_embed = text_model.embed(tokens)

    image = load_image(IMAGE_PATH, RESIZE_TO)
    image_model = PatchEmbeddingModel(PATCH_SIZE, IMG_EMBED_DIM, SEED)
    patches = image_model.extract_patches(image)
    img_embed = image_model.project_patches(patches)

    text_vec, img_vec = fuse_embeddings(text_embed, img_embed)
    similarity = compare_embeddings(text_vec, img_vec)

    print(f"Text-image similarity: {similarity:.4f}")
    if similarity >= SIMILARITY_THRESHOLD:
        print("✅ The caption matches the image.")
    else:
        print("❌ The caption does NOT match the image.")

if __name__ == "__main__":
    main()
