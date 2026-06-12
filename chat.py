"""
Interactive chat with trained MultiExpertMemoryDNN model.
Loads saved weights or trains fresh if none exist.
"""
import math, os, json, sys, time
from dataclasses import dataclass, asdict
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from looped_transformer import LoopedTransformer
from multi_expert_training import (
    MultiExpertMemoryDNN, get_mixed_data, make_model,
    TrainConfig, evaluate,
)
import torch.nn.functional as F

MODEL_PATH = os.path.join(os.path.dirname(__file__), "chat_model.pt")
CFG_PATH = os.path.join(os.path.dirname(__file__), "chat_config.json")


def train_and_save():
    """Quick 500-step training to get a usable model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg = TrainConfig(
        dim=64, num_heads=4, num_loops=8,
        mem_dim=32, num_caps=4, moe_k=2,
        experts_per_cap=4,
        seq_len=64, bs=16,
        lr=3e-4, wd=0.1, warmup=100,
        steps=1500, grad_clip=1.0,
        log_every=500, eval_every=500,
    )
    
    print("Building data...")
    train_ds, val_ds = get_mixed_data(seq_len=cfg.seq_len, stride=16)
    print(f"Vocab: {train_ds.vocab_size}, Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    model = make_model(train_ds.vocab_size, cfg).to(device)
    model.train()
    
    loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True)
    
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if "bias" in n or "norm" in n: no_decay.append(p)
        else: decay.append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.wd, "lr": cfg.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": cfg.lr},
    ], betas=(0.9, 0.95))
    
    def sched_fn(s):
        if s < cfg.warmup: return float(s) / max(1, cfg.warmup)
        p = float(s - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)
    
    step = 0
    t0 = time.time()
    while step < cfg.steps:
        for batch in loader:
            if step >= cfg.steps: break
            x, y, _ = batch if len(batch) == 3 else (batch[0], batch[1], None)
            out = model(x.to(device), labels=y.to(device))
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()
            if step % 500 == 0:
                print(f"  step {step:5d} loss {out.loss.item():.4f}")
            step += 1
    
    t = time.time() - t0
    print(f"Trained {cfg.steps} steps in {t:.0f}s. Saving...")
    
    # Save
    torch.save({
        'model_state': model.state_dict(),
        'vocab_size': train_ds.vocab_size,
        'stoi': train_ds.stoi,
        'itos': {i: c for c, i in train_ds.stoi.items()},
        'cfg': {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
    }, MODEL_PATH)
    with open(CFG_PATH, "w") as f:
        json.dump({
            'vocab_size': train_ds.vocab_size,
            'stoi': train_ds.stoi,
            'itos': {i: c for c, i in train_ds.stoi.items()},
        }, f, indent=2)
    
    print(f"Model saved to {MODEL_PATH}")
    return model, train_ds.stoi, {i: c for c, i in train_ds.stoi.items()}


def load_model():
    """Load saved model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    
    cfg = TrainConfig(**checkpoint['cfg'])
    model = make_model(checkpoint['vocab_size'], cfg).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, checkpoint['stoi'], checkpoint['itos'], cfg


def char_tokenize(text: str, stoi: dict, seq_len: int) -> torch.Tensor:
    """Tokenize string to tensor of indices, padding unknown chars."""
    ids = [stoi.get(c, 0) for c in text]
    if len(ids) > seq_len:
        ids = ids[-seq_len:]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def generate(model, stoi, itos, prompt: str, max_new: int = 200, temp: float = 0.8, top_k: int = 10):
    """Generate text. top-k filtering + temperature for character-level."""
    device = next(model.parameters()).device
    seq_len = 64
    
    input_ids = char_tokenize(prompt, stoi, seq_len)
    input_ids = input_ids.to(device)
    
    generated = list(input_ids[0].cpu().tolist())
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            ctx = torch.tensor([generated[-seq_len:]], dtype=torch.long, device=device)
            out = model(ctx)
            logits = out.logits[0, -1, :]
            
            # Top-k filtering
            if top_k > 0:
                top_vals, _ = logits.topk(top_k)
                logits[logits < top_vals[-1]] = float('-inf')
            
            if temp > 0:
                probs = F.softmax(logits / temp, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            else:
                next_id = logits.argmax().item()
            
            generated.append(next_id)
            
            # Stop at period + newline or end punctuation
            if next_id == stoi.get('.', 0) and len(generated) > len(prompt) + 10:
                break
    
    text = ''.join(itos.get(i, '?') for i in generated)
    return text[len(prompt):]


def interactive_loop(model, stoi, itos, cfg_dict):
    """Interactive chat loop."""
    print(f"\n{'='*60}")
    print(f"  MultiExpert Memory Chat")
    print(f"  {cfg_dict['num_loops']} loops, {cfg_dict['num_caps']} caps, {cfg_dict['experts_per_cap']} experts/cap")
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"{'='*60}")
    print(f"  Type prompts. The model generates character-by-character.")
    print(f"  'exit' to quit, 'gates' to see precision gates")
    print(f"{'='*60}\n")
    
    while True:
        prompt = input(">> ")
        if prompt.lower() in ('exit', 'quit', 'q'):
            break
        if prompt.lower() == 'gates':
            pg = model.capability_moe.precision_gates()
            names = ["memorize", "math", "speak", "reason"]
            print("  Precision gates:")
            for n, g in zip(names, pg):
                status = "HIGH" if g > 0.52 else ("LOW" if g < 0.48 else "MID")
                print(f"    {n:>10}: {g.item():.4f} ({status})")
            continue
        
        text = generate(model, stoi, itos, prompt, max_new=300, temp=0.1)
        print(f"  {text}")
        print()


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model, stoi, itos, cfg = load_model()
        interactive_loop(model, stoi, itos, cfg.__dict__ if hasattr(cfg, '__dict__') else cfg)
    else:
        print("No saved model found. Training one now...")
        model, stoi, itos = train_and_save()
        cfg = asdict(TrainConfig())
        interactive_loop(model, stoi, {i: c for c, i in stoi.items()}, cfg)
