from config.config import *
from model.transformer import MiniTransformer
import torch
import torch.nn as nn
import torch.optim as optim

def dummy_tokenize(text):
    words = text.lower().split()
    vocab = list(set(words))
    token_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_token = {i: w for w, i in token_to_id.items()}
    encoded = [token_to_id[w] for w in words]
    return encoded, token_to_id, id_to_token

def get_batches(encoded, seq_len, batch_size):
    for i in range(0, len(encoded) - seq_len, seq_len * batch_size):
        x = torch.tensor([
            encoded[i + j*seq_len : i + (j+1)*seq_len]
            for j in range(batch_size)
        ])
        y = torch.tensor([
            encoded[i + j*seq_len + 1 : i + (j+1)*seq_len + 1]
            for j in range(batch_size)
        ])
        yield x, y

def main():
    with open(CORPUS_PATH, 'r') as f:
        text = f.read()

    encoded, token_to_id, id_to_token = dummy_tokenize(text)
    model = MiniTransformer(len(token_to_id), EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for x, y in get_batches(encoded, SEQ_LEN, BATCH_SIZE):
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pth")
    torch.save(token_to_id, "token_to_id.pt")
    torch.save(id_to_token, "id_to_token.pt")

if __name__ == "__main__":
    main()
