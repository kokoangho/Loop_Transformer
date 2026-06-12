"""
Wrappers to train both architectures in same harness + comparison.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json, math, os, time, sys
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from looped_transformer import LoopedTransformer, LoopedTransformerBlock, RotaryEmbedding, SwiGLU
from moe_memory_pathway import CapabilityMoEMemory, MemoryBank

# -- Codex wrapper -----------------------------------------------------------

class CodexSolutionWrapper(nn.Module):
    """Wrap CodexSolution so it plugs into LoopedTransformer like CapabilityMoEMemory.
    
    Matches interface: (hidden_states, bank, memory_state) → (next_memory, injection, bank)
    """
    
    def __init__(self, hidden_dim: int, memory_dim: int, mlp_dim: int,
                 moe_k: int = 2, num_caps: int = 4, max_loops: int = 16):
        super().__init__()
        import importlib.util, sys
        path = os.path.join(os.path.dirname(__file__), ".codex", "codex_solution.py")
        spec = importlib.util.spec_from_file_location("codex_solution", path)
        codex_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codex_mod)
        CodexSolution = codex_mod.CodexSolution
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_caps = num_caps
        self.cap_dim = memory_dim // num_caps
        
        # Codex expects dim = memory_dim (it splits internally by C)
        self.codex = CodexSolution(
            dim=memory_dim,
            num_experts=moe_k * 2,  # more experts for richer capacity
            expert_top_k=moe_k,
            memory_top_k=moe_k,
            expert_hidden_dim=mlp_dim,
            auto_reset=False,
        )
        self.codex.dim = memory_dim
        self.codex.capability_dim = self.cap_dim
        
        # Projection + gate (same as CapabilityMoEMemory for consistent injection)
        self.project = nn.Linear(memory_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + memory_dim),
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Refiner on FULL bank (Codex already has its own refine)
        ref_layer = nn.TransformerEncoderLayer(
            d_model=memory_dim, nhead=min(4, memory_dim // 16 or 1),
            dim_feedforward=memory_dim * 2, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.refiner = nn.TransformerEncoder(ref_layer, num_layers=2)
        self.refiner_norm = nn.LayerNorm(memory_dim)
        
        self.initial_memory = nn.Parameter(torch.zeros(1, num_caps, self.cap_dim))
    
    def precision_gates(self) -> Tensor:
        return torch.stack([
            torch.sigmoid(self.codex.capability_moes[name].precision_logit)
            for name in self.codex.capabilities
        ])
    
    def forward(self, hidden_states: Tensor, bank: MemoryBank,
                memory_state: Tensor) -> Tuple[Tensor, Tensor, MemoryBank]:
        B, S, D = hidden_states.shape
        mem_flat = memory_state.reshape(B, -1)  # [B, mem_dim]
        
        # Codex step on full memory vector
        codex_out = self.codex.step(mem_flat, store=True)
        
        # Reshape back to [B, C, cap_dim] for bank
        next_memory = codex_out.reshape(B, self.num_caps, self.cap_dim)
        bank.append(next_memory)
        
        # Gated injection
        proj = self.project(codex_out).unsqueeze(1)
        expanded = codex_out.unsqueeze(1).expand(-1, S, -1)
        g = self.gate(torch.cat([hidden_states, expanded], dim=-1))
        injection = g * proj
        
        return next_memory, injection, bank
    
    def refine(self, bank: MemoryBank) -> Tensor:
        if bank.size == 0:
            return self.initial_memory.reshape(1, -1).expand(1, -1).squeeze(0)
        # Use Codex's built-in refine (which processes the full bank)
        return self.codex.refine()
    
    def init_bank(self, batch_size: int, device: torch.device) -> MemoryBank:
        bank = MemoryBank(batch_size, self.num_caps, self.cap_dim, 16)
        bank.to(device)
        return bank
    
    def init_memory(self, batch_size: int, device: torch.device) -> Tensor:
        return self.initial_memory.expand(batch_size, -1, -1).to(device)


# -- Data -------------------------------------------------------------------

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class ShakespeareDataset(Dataset):
    def __init__(self, path: str, seq_len: int = 128, stride: int = 64):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.examples = []
        for i in range(0, len(data) - seq_len, stride):
            self.examples.append((data[i:i+seq_len], data[i+1:i+seq_len+1]))
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


def get_data(seq_len=128, stride=64, val_split=0.1):
    path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path) as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    n = len(chars)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(len(data) * (1 - val_split))
    class DS(Dataset):
        def __init__(self, d):
            self.ex = [(d[i:i+seq_len], d[i+1:i+seq_len+1]) for i in range(0, len(d)-seq_len, stride)]
            self.vocab_size = n
        def __len__(self): return len(self.ex)
        def __getitem__(self, i): return self.ex[i]
    return DS(data[:split]), DS(data[split:])


def get_cosine_schedule(optimizer, warmup, total):
    def fn(step):
        if step < warmup: return float(step) / max(1, warmup)
        p = float(step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# -- Training ---------------------------------------------------------------

@dataclass
class TrainResult:
    variant: str; params: int; val_loss: float; time_s: float
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)


def make_model(variant, cfg, vocab_size):
    """Create model for variant. Variants:
      - looped: plain shared block
      - looped+moe (mine): CapabilityMoEMemory
      - looped+codex (codex): CodexSolutionWrapper
      - looped+merged: combined architecture
    """
    base = dict(vocab_size=vocab_size, dim=cfg.dim, num_heads=cfg.num_heads,
                num_loops=cfg.num_loops, max_seq_len=cfg.seq_len,
                use_memory_pathway=True, memory_dim=cfg.mem_dim,
                use_timestep_emb=True)
    
    if variant == "looped":
        return LoopedTransformer(**{**base, "use_memory_pathway": False})
    elif variant == "looped+moe":
        return LoopedTransformer(**base, use_capability_moe=True,
                                 num_capabilities=cfg.num_caps, moe_k=cfg.moe_k)
    elif variant == "looped+codex":
        # Integration trick: replace capability_moe with CodexSolutionWrapper
        m = LoopedTransformer(**base)
        # Don't use regular memory_pathway or capability_moe
        m.use_memory_pathway = False
        m.memory_cell = None
        m.use_capability_moe = True  # flag to trigger MoE path in forward
        m.capability_moe = CodexSolutionWrapper(
            hidden_dim=cfg.dim, memory_dim=cfg.mem_dim,
            mlp_dim=cfg.mem_dim * 4, moe_k=cfg.moe_k,
            num_caps=cfg.num_caps, max_loops=cfg.num_loops,
        )
        m.initial_memory = m.capability_moe.initial_memory
        return m
    else:
        raise ValueError(f"Unknown variant: {variant}")


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        loss = model(x.to(device), labels=y.to(device)).loss
        total += loss.item(); n += 1
        if n >= max_batches: break
    return total / n


def train_variant(variant, cfg, train_ds, val_ds, device="cpu"):
    print(f"\n{'='*50}\n{variant}\n{'='*50}")
    
    model = make_model(variant, cfg, train_ds.vocab_size).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Params: {nparams:,}")
    
    loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True)
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
    sched = get_cosine_schedule(opt, cfg.warmup, cfg.steps)
    
    res = TrainResult(variant=variant, params=nparams, val_loss=float("inf"), time_s=0)
    best_val = float("inf")
    t0 = time.time()
    step = 0
    
    while step < cfg.steps:
        for x, y in loader:
            if step >= cfg.steps: break
            out = model(x.to(device), labels=y.to(device))
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()
            
            if step % cfg.log_every == 0:
                res.train_losses.append(out.loss.item())
                print(f"  step {step:5d} loss {out.loss.item():.4f}", end="")
                if step > 0 and step % cfg.eval_every == 0:
                    vl = evaluate(model, val_loader, device)
                    res.val_losses.append(vl)
                    print(f"  val {vl:.4f}", end="")
                    if vl < best_val: best_val = vl
                print()
            step += 1
    
    res.time_s = time.time() - t0
    res.val_loss = evaluate(model, val_loader, device)
    print(f"Done. Val loss: {res.val_loss:.4f}, time: {res.time_s:.1f}s")
    return res, model


# -- Main -------------------------------------------------------------------

@dataclass
class ExpConfig:
    dim: int = 64; num_heads: int = 4; num_loops: int = 4
    mem_dim: int = 32; num_caps: int = 4; moe_k: int = 2
    seq_len: int = 64; bs: int = 16
    lr: float = 3e-4; wd: float = 0.1; warmup: int = 50
    steps: int = 2000; grad_clip: float = 1.0
    log_every: int = 200; eval_every: int = 500


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExpConfig()
    
    print("Loading data...")
    train_ds, val_ds = get_data(seq_len=cfg.seq_len, stride=32)
    print(f"Vocab: {train_ds.vocab_size}, Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    results = []
    models = {}
    
    for variant in ["looped", "looped+moe", "looped+codex"]:
        r, m = train_variant(variant, cfg, train_ds, val_ds, device)
        results.append(r)
        models[variant] = m
    
    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"{'Variant':<20} {'Params':>10} {'Val Loss':>10} {'Time':>8}")
    print("-"*50)
    for r in sorted(results, key=lambda x: x.val_loss):
        print(f"{r.variant:<20} {r.params:>10,} {r.val_loss:>10.4f} {r.time_s:>8.1f}s")
    
    # Architect diff
    print(f"\n=== Architecture Comparison ===")
    mine = models["looped+moe"]
    codex = models["looped+codex"]
    
    print(f"\nMine (CapabilityMoEMemory):")
    mm = mine.capability_moe
    if hasattr(mm, 'precision_gates'):
        print(f"  precision gates: {mm.precision_gates().detach().cpu().numpy().round(3)}")
    if hasattr(mm, 'refiner'):
        print(f"  refiner: {type(mm.refiner).__name__}")
    
    print(f"\nCodex (CodexSolution):")
    cm = codex.capability_moe
    if hasattr(cm, 'precision_gates'):
        print(f"  precision gates: {cm.precision_gates().detach().cpu().numpy().round(3)}")
    if hasattr(cm, 'refiner'):
        print(f"  refiner: {type(cm.refiner).__name__}")
    
    # Save
    out = {
        "config": asdict(cfg),
        "results": [{"variant": r.variant, "params": r.params,
                      "val_loss": r.val_loss, "time_s": r.time_s} for r in results],
    }
    with open(os.path.join(os.path.dirname(__file__), "comparison_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to comparison_results.json")


if __name__ == "__main__":
    main()
