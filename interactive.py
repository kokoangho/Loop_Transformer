"""
Interactive chat with the trained MultiExpert memory model.

Loads saved checkpoint, lets you type prompts, see generated text.
Character-level LM trained on speak + math + code + reasoning.
"""

import json, math, os, sys, torch
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

# Same model definition as training
from multi_expert_training import (
    MultiExpertMemoryDNN, MixedDomainDataset, get_mixed_data,
    ExpConfig as _EC, make_model
)

# Build config matching training
class Cfg:
    dim=64; num_heads=4; num_loops=8; seq_len=64; mem_dim=32
    num_caps=4; moe_k=2; experts_per_cap=4; use_memory_pathway=True
    use_timestep_emb=True

device = "cpu"

# Load dataset to get vocab
train_ds, val_ds = get_mixed_data(seq_len=64)
vocab_size = train_ds.vocab_size
stoi = train_ds.stoi
itos = train_ds.itos
print(f"Vocab: {vocab_size} chars, device: {device}")

# Build model same arch as trained
model = make_model(vocab_size, Cfg()).to(device)

# Try loading saved state
ckpt_path = os.path.join(os.path.dirname(__file__), "multi_expert_checkpoint.pt")
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("Loaded checkpoint")
else:
    print("No checkpoint found — model is randomly initialized")

model.eval()

@torch.no_grad()
def generate(prompt: str, max_new: int = 200, temp: float = 0.8) -> str:
    """Generate text continuation from prompt."""
    chars = list(prompt)
    # Encode
    input_ids = torch.tensor([[stoi.get(c, 0) for c in chars]], dtype=torch.long, device=device)
    
    for _ in range(max_new):
        # Only use last seq_len tokens
        if input_ids.shape[1] > 64:
            ctx = input_ids[:, -64:]
        else:
            ctx = input_ids
        
        out = model(ctx)
        logits = out.logits[:, -1, :]  # [B, V]
        
        # Temperature sampling
        logits = logits / temp
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # [B, 1]
        
        input_ids = torch.cat([input_ids, next_id], dim=1)
        chars.append(itos[next_id.item()])
    
    return "".join(chars)


def main():
    print("\n" + "="*50)
    print("Loop Transformer — Interactive Chat")
    print("="*50)
    print("Type a prompt. The model continues it character-by-character.")
    print("Domains: speak, math, code, reasoning")
    print("Commands: /temp N (set temperature 0.1-2.0), /save, /quit")
    print("="*50 + "\n")
    
    temp = 0.8
    while True:
        prompt = input("You: ").strip()
        if prompt == "/quit":
            break
        if prompt.startswith("/temp"):
            try:
                temp = float(prompt.split()[1])
                print(f"  Temperature set to {temp}")
            except: print("  Usage: /temp 0.8")
            continue
        if prompt == "/save":
            ckpt_path = os.path.join(os.path.dirname(__file__), "multi_expert_checkpoint.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved to {ckpt_path}")
            continue
        if not prompt:
            continue
        
        print(f"  Generating (temp={temp})...")
        result = generate(prompt, max_new=300, temp=temp)
        print(f"\nModel: {result}\n")

if __name__ == "__main__":
    main()
