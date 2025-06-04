import torch

def generate_text(model, prompt_ids, max_new_tokens, vocab_size, id_to_token):
    model.eval()
    generated = prompt_ids.copy()

    for _ in range(max_new_tokens):
        x = torch.tensor([generated[-32:]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=0)
        next_token = torch.argmax(probs).item()
        generated.append(next_token)

    return [id_to_token[t] for t in generated]
