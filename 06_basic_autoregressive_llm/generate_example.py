import torch
from config.config import *
from model.transformer import MiniTransformer
from generate import generate_text

prompt = ['hello']
token_to_id = torch.load("token_to_id.pt")
id_to_token = torch.load("id_to_token.pt")
prompt_ids = [token_to_id[t] for t in prompt]

model = MiniTransformer(len(token_to_id), EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN)
model.load_state_dict(torch.load("model.pth"))

generated = generate_text(model, prompt_ids, max_new_tokens=10, vocab_size=len(token_to_id), id_to_token=id_to_token)
print("Generated:", ' '.join(generated))
