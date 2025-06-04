import sys
import string

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def tokenize(self, text: str) -> list:
        return text.split()

    def build_vocab(self, tokens: list) -> None:
        unique_tokens = sorted(set(tokens))
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, tokens: list) -> list:
        return [self.vocab[token] for token in tokens]

    def decode(self, token_ids: list) -> list:
        return [self.inverse_vocab[token_id] for token_id in token_ids]

def main(file_path: str) -> None:
    tokenizer = SimpleTokenizer()

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    cleaned = tokenizer.clean_text(raw_text)
    tokens = tokenizer.tokenize(cleaned)
    tokenizer.build_vocab(tokens)
    token_ids = tokenizer.encode(tokens)

    print("\n--- Raw Tokens ---")
    print(tokens[:20])

    print("\n--- Vocabulary (first 10) ---")
    for i, (word, idx) in enumerate(tokenizer.vocab.items()):
        if i >= 10:
            break
        print(f"{word}: {idx}")

    print("\n--- Encoded Token IDs (first 20) ---")
    print(token_ids[:20])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tokenizer.py <file_path>")
        sys.exit(1)

    main(sys.argv[1])
