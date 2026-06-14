"""
Train MultiExpert Memory DNN on FULL Shakespeare at word level.
~200K tokens, ~20K vocab, GPU-scale.
"""

from __future__ import annotations
import math, os, time, json, re
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def get_words(text: str, min_freq: int = 2):
    """Tokenize to words, build vocab with OOV."""
    words = re.findall(r"[A-Za-z']+|[.,!?;:\-()\"]|\\S", text)
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted(w for w, c in freq.items() if c >= min_freq)
    stoi = {w: i for i, w in enumerate(vocab)}
    return words, stoi, len(vocab), {i: w for w, i in stoi.items()}

class WordDataset(Dataset):
    def __init__(self, path: str, seq_len: int = 128, stride: int = 64, val_split: float = 0.1):
        with open(path) as f:
            text = f.read()
        split = int(len(text) * (1 - val_split))
        train_txt, val_txt = text[:split], text[split:]
        
        # Build vocab on train only
        train_words, stoi, vocab_size, itos = get_words(train_txt, min_freq=2)
        val_words = re.findall(r"[A-Za-z']+|[.,!?;:\-()\"]|\\S", val_txt)
        
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = vocab_size
        
        def to_tensor(words):
            ids = [stoi.get(w, stoi["<unk>"]) for w in words]
            return torch.tensor(ids, dtype=torch.long)
        
        train_ids = to_tensor(train_words)
        val_ids = to_tensor(val_words)
        
        self.train_ex = []
        for i in range(0, len(train_ids) - seq_len, stride):
            self.train_ex.append((train_ids[i:i+seq_len], train_ids[i+1:i+seq_len+1]))
        
        self.val_ex = []
        for i in range(0, len(val_ids) - seq_len, stride):
            self.val_ex.append((val_ids[i:i+seq_len], val_ids[i+1:i+seq_len+1]))
    
    def train(self): return type('DS', (), {'__len__': lambda s: len(self.train_ex), '__getitem__': lambda s, i: self.train_ex[i], 'vocab_size': self.vocab_size})()
    def val(self): return type('DS', (), {'__len__': lambda s: len(self.val_ex), '__getitem__': lambda s, i: self.val_ex[i], 'vocab_size': self.vocab_size})()


@dataclass
class WConfig:
    dim: int = 256; num_heads: int = 8; num_loops: int = 8
    mem_dim: int = 128; num_caps: int = 4; moe_k: int = 2; experts_per_cap: int = 4
    seq_len: int = 128; bs: int = 32; lr: float = 3e-4; wd: float = 0.1
    warmup: int = 1000; steps: int = 20000; grad_clip: float = 1.0


def main():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from looped_transformer import LoopedTransformer
    from multi_expert_training import MultiExpertMemoryDNN
    
    cfg = WConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Download full Shakespeare
    path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
    if not os.path.exists(path):
        import urllib.request
        print("Downloading Shakespeare...")
        urllib.request.urlretrieve(URL, path)
    
    # Word-level dataset
    print("Building word-level dataset...")
    ds = WordDataset(path, seq_len=cfg.seq_len, stride=64)
    train_ds = ds.train()
    val_ds = ds.val()
    vocab_size = ds.vocab_size
    print(f"Vocab: {vocab_size} words, Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    print(f"Building MultiExpert (dim={cfg.dim}, loops={cfg.num_loops}, caps={cfg.num_caps}, experts/cap={cfg.experts_per_cap})...")
    base = LoopedTransformer(
        vocab_size=vocab_size, dim=cfg.dim, num_heads=cfg.num_heads,
        num_loops=cfg.num_loops, max_seq_len=cfg.seq_len,
        use_memory_pathway=True, memory_dim=cfg.mem_dim, use_timestep_emb=True,
    )
    base.use_capability_moe = True
    base.capability_moe = MultiExpertMemoryDNN(
        hidden_dim=cfg.dim, memory_dim=cfg.mem_dim, mlp_dim=cfg.mem_dim * 4,
        moe_k=cfg.moe_k, num_caps=cfg.num_caps, max_loops=cfg.num_loops,
        experts_per_cap=cfg.experts_per_cap,
    )
    base.initial_memory = base.capability_moe.initial_memory
    model = base.to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Params: {nparams:,}")
    
    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.bs, drop_last=True)
    
    # Optimizer
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
    
    # Training
    @torch.no_grad()
    def evaluate(loader, n=20):
        model.eval()
        t, c = 0.0, 0
        for x, y, *_ in loader:
            t += model(x.to(device), labels=y.to(device)).loss.item()
            c += 1
            if c >= n: break
        return t / c
    
    best_val = float("inf")
    t0 = time.time()
    step = 0
    log_every = max(1, cfg.steps // 20)
    eval_every = max(1, cfg.steps // 10)
    
    print(f"\nTraining {cfg.steps} steps...")
    while step < cfg.steps:
        for batch in train_loader:
            if step >= cfg.steps: break
            x, y, *_ = batch
            out = model(x.to(device), labels=y.to(device))
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()
            
            if step % log_every == 0:
                pg = model.capability_moe.precision_gates()
                print(f"  step {step:6d}/{cfg.steps} loss {out.loss.item():.4f} gates {pg.detach().cpu().numpy().round(3)}", end="")
                if step > 0 and step % eval_every == 0:
                    vl = evaluate(val_loader)
                    print(f" val {vl:.4f}", end="")
                    if vl < best_val:
                        best_val = vl
                        ckpt = os.path.join(os.path.dirname(__file__), "word_level_best.pt")
                        torch.save(model.state_dict(), ckpt)
                print()
            step += 1
    
    final_val = evaluate(val_loader)
    t = time.time() - t0
    pg = model.capability_moe.precision_gates()
    print(f"\nDone. Best val: {best_val:.4f}, Final val: {final_val:.4f}, time: {t:.1f}s")
    print(f"Precision gates: {pg.detach().cpu().numpy().round(3)}")
    
    # Save final
    ckpt = os.path.join(os.path.dirname(__file__), "word_level_final.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"Final checkpoint saved to {ckpt}")


if __name__ == "__main__":
    main()
