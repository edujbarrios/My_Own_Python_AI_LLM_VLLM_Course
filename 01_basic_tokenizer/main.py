from tokenizer.tokenizer import Tokenizer
from tokenizer.utils import read_text
from config.config import TEXT_FILE_PATH, PRINT_TOKEN_LIMIT, PRINT_VOCAB_LIMIT

def main():
    tokenizer = Tokenizer()
    text = read_text(TEXT_FILE_PATH)
    cleaned_text = tokenizer.clean_text(text)
    tokens = tokenizer.tokenize(cleaned_text)

    tokenizer.build_vocab(tokens)
    token_ids = tokenizer.encode(tokens)

    print("\n--- Cleaned Tokens (first {}) ---".format(PRINT_TOKEN_LIMIT))
    print(tokens[:PRINT_TOKEN_LIMIT])

    print("\n--- Vocabulary (first {}) ---".format(PRINT_VOCAB_LIMIT))
    for i, (word, idx) in enumerate(tokenizer.vocab.items()):
        if i >= PRINT_VOCAB_LIMIT:
            break
        print(f"{word}: {idx}")

    print("\n--- Token IDs (first {}) ---".format(PRINT_TOKEN_LIMIT))
    print(token_ids[:PRINT_TOKEN_LIMIT])

if __name__ == "__main__":
    main()
