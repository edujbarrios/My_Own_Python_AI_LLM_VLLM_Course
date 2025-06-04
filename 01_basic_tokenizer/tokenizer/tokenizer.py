from config.config import TO_LOWERCASE, REMOVE_PUNCTUATION, VOCABULARY_SORTED
import string

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    def clean_text(self, text: str) -> str:
        if TO_LOWERCASE:
            text = text.lower()
        if REMOVE_PUNCTUATION:
            text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def tokenize(self, text: str) -> list:
        return text.split()

    def build_vocab(self, tokens: list) -> None:
        unique_tokens = set(tokens)
        if VOCABULARY_SORTED:
            unique_tokens = sorted(unique_tokens)
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, tokens: list) -> list:
        return [self.vocab[token] for token in tokens]

    def decode(self, token_ids: list) -> list:
        return [self.inverse_vocab[token_id] for token_id in token_ids]
