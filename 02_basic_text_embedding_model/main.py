from config.config import TEXT_FILE_PATH, EMBEDDING_DIM, SEED, TO_LOWERCASE, REMOVE_PUNCTUATION, SHOW_SAMPLE_TOKENS
from embeddings.utils import read_text, clean_and_tokenize
from embeddings.embedding_model import EmbeddingModel

def main():
    raw_text = read_text(TEXT_FILE_PATH)
    tokens = clean_and_tokenize(raw_text, TO_LOWERCASE, REMOVE_PUNCTUATION)
    vocab = sorted(set(tokens))

    model = EmbeddingModel(vocab, EMBEDDING_DIM, SEED)

    print(f"\n--- Vocabulary (first {SHOW_SAMPLE_TOKENS}) ---")
    print(vocab[:SHOW_SAMPLE_TOKENS])

    sample_token = vocab[SHOW_SAMPLE_TOKENS - 1]
    vector = model.get_vector(sample_token)

    print(f"\n--- Embedding vector for token '{sample_token}' ---")
    print(vector)

if __name__ == "__main__":
    main()
