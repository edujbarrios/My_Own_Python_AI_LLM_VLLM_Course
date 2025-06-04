import torch.nn as nn
from model.attention import MultiHeadSelfAttention
from model.embeddings import TokenEmbedding, PositionalEncoding

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadSelfAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, num_layers, seq_len):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, emb_dim)
        self.pos_emb = PositionalEncoding(emb_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads) for _ in range(num_layers)
        ])
        self.output = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
