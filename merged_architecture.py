"""
Merged architecture: CapabilityMoEMemory (mine) × CodexSolution (codex).

Takes best of both:

Level 1 — Memory MoE (mine):  top-k over prior same-capability memories
Level 2 — Expert MoE (codex): top-k over expert MLPs within each capability
Attention scoring (codex):  query·key smoother than hard score
Structured bank (mine):     [B, N, C, cap_dim] tensor, easy slicing
Precision gates (both):     sigmoid per-cap, multiply expert output
Refiner (both):             TransformerEncoder over full bank

Total: 2-level routing per capability channel.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json, math, os, time, sys
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from looped_transformer import LoopedTransformer
from moe_memory_pathway import MemoryBank  # reuse

CAPABILITIES: Tuple[str, str, str, str] = ("memorize", "math", "speak", "reason")


class MergedCapabilityMoE(nn.Module):
    """One capability channel: memory MoE → expert MoE.
    
    Input:  x_c [B, cap_dim], prior_candidates [B, N, cap_dim]
    Output: updated m_c [B, cap_dim]
    
    Steps:
      1. Memory MoE: score all prior memories, pick top-k, aggregate → context
      2. Expert MoE: route (x_c + context) to top-k of 4 expert MLPs
      3. Precision gate: sigmoid(logit) scales expert output
      4. Residual: x_c + gate * expert_output
    """
    
    def __init__(self, cap_dim: int, hidden_dim: int, num_experts: int = 4,
                 memory_k: int = 2, expert_k: int = 2):
        super().__init__()
        self.memory_k = memory_k
        self.expert_k = expert_k
        self.cap_dim = cap_dim
        
        # Level 1: Memory MoE (attention-style scoring from Codex)
        self.mem_norm_q = nn.LayerNorm(cap_dim)
        self.mem_norm_k = nn.LayerNorm(cap_dim)
        
        # Level 2: Expert MoE (router from Codex)
        self.expert_router = nn.Linear(cap_dim * 2, num_experts)  # concat(x, context)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(cap_dim * 2),
                nn.Linear(cap_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, cap_dim),
            )
            for _ in range(num_experts)
        ])
        
        # Precision gate (from mine & Codex)
        self.precision_logit = nn.Parameter(torch.zeros(()))
    
    def _read_memory(self, x: Tensor, candidates: Tensor) -> Tensor:
        """Top-k memory routing via attention scoring (Codex style)."""
        B, N, D = candidates.shape
        if N == 0:
            return torch.zeros_like(x)
        q = self.mem_norm_q(x).unsqueeze(1)      # [B, 1, D]
        k = self.mem_norm_k(candidates)           # [B, N, D]
        scores = (q * k).sum(dim=-1) / math.sqrt(D)  # [B, N]
        k_eff = min(self.memory_k, N)
        vals, idx = scores.topk(k_eff, dim=-1)
        w = F.softmax(vals, dim=-1).unsqueeze(-1)  # [B, k, 1]
        gathered = torch.gather(candidates, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        return (w * gathered).sum(dim=1)           # [B, D]
    
    def forward(self, x: Tensor, prior: Sequence[Tensor]) -> Tensor:
        """x: [B, cap_dim], prior: list of [B, cap_dim] prior states.
        Returns updated [B, cap_dim] with residual.
        """
        # Level 1: Memory MoE
        candidates = torch.stack(list(prior), dim=1) if prior else torch.zeros(x.shape[0], 0, self.cap_dim, device=x.device)
        context = self._read_memory(x, candidates)
        
        # Level 2: Expert MoE
        routed = torch.cat([x, context], dim=-1)  # [B, cap_dim*2]
        logits = self.expert_router(routed)        # [B, num_experts]
        k = min(self.expert_k, logits.shape[-1])
        top_l, top_i = logits.topk(k, dim=-1)
        top_w = F.softmax(top_l, dim=-1)           # [B, k]
        
        expert_outs = torch.stack([e(routed) for e in self.experts], dim=-2)  # [B, num_experts, D]
        selected = torch.gather(expert_outs, 1, top_i.unsqueeze(-1).expand(-1, -1, self.cap_dim))
        mixed = (selected * top_w.unsqueeze(-1)).sum(dim=1)  # [B, D]
        
        # Precision gate
        return x + torch.sigmoid(self.precision_logit) * mixed


class MergedMemoryDNN(nn.Module):
    """Full merged memory DNN. Drops into LoopedTransformer same slot as CapabilityMoEMemory."""
    
    def __init__(self, hidden_dim: int, memory_dim: int, mlp_dim: int,
                 moe_k: int = 2, num_caps: int = 4, max_loops: int = 16):
        super().__init__()
        assert memory_dim % num_caps == 0
        self.num_caps = num_caps
        self.cap_dim = memory_dim // num_caps
        self.moe_k = moe_k
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        
        # Per-capability merged MoEs (memory MoE + expert MoE)
        self.capability_moes = nn.ModuleDict({
            name: MergedCapabilityMoE(
                cap_dim=self.cap_dim,
                hidden_dim=mlp_dim,
                num_experts=moe_k * 2,  # richer expert set
                memory_k=moe_k,
                expert_k=moe_k,
            )
            for name in CAPABILITIES[:num_caps]
        })
        
        # Projection + gate for residual injection (from mine)
        self.project = nn.Linear(memory_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + memory_dim),
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Post-loop refiner (from both)
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
            torch.sigmoid(m.precision_logit) for m in self.capability_moes.values()
        ])
    
    def forward(self, hidden_states: Tensor, bank: MemoryBank,
                memory_state: Tensor) -> Tuple[Tensor, Tensor, MemoryBank]:
        B, S, D = hidden_states.shape
        pooled = hidden_states.mean(dim=1)  # [B, D]
        
        next_mem_caps = []
        for name in list(self.capability_moes.keys()):
            c_idx = list(self.capability_moes.keys()).index(name)
            # Query = previous memory + pooled hidden (from mine)
            prev = memory_state[:, c_idx]  # [B, cap_dim]
            # Inject hidden context into query
            query = prev + self.project.weight.new_zeros(B, self.cap_dim)  # placeholder
            
            # Get prior same-capability memories
            prior_bank = bank.get_capability(c_idx)  # [B, N, cap_dim]
            prior_list = [prior_bank[:, i] for i in range(prior_bank.shape[1])]
            
            # Merged MoE
            m_new = self.capability_moes[name](query, prior_list)
            next_mem_caps.append(m_new)
        
        next_memory = torch.stack(next_mem_caps, dim=1)
        bank.append(next_memory)
        
        # Gated injection (from mine)
        mem_flat = next_memory.reshape(B, -1)
        proj = self.project(mem_flat).unsqueeze(1)
        expanded = mem_flat.unsqueeze(1).expand(-1, S, -1)
        g = self.gate(torch.cat([hidden_states, expanded], dim=-1))
        injection = g * proj
        
        return next_memory, injection, bank
    
    def refine(self, bank: MemoryBank) -> Tensor:
        if bank.size == 0:
            return self.initial_memory.reshape(1, -1).expand(1, -1).squeeze(0)
        full = bank.get_full()
        refined = self.refiner(full)
        return self.refiner_norm(refined[:, -1])
    
    def init_bank(self, batch_size: int, device: torch.device) -> MemoryBank:
        bank = MemoryBank(batch_size, self.num_caps, self.cap_dim, 16)
        bank.to(device)
        return bank
    
    def init_memory(self, batch_size: int, device: torch.device) -> Tensor:
        return self.initial_memory.expand(batch_size, -1, -1).to(device)


# -- Smoke test -------------------------------------------------------------

def smoke_test():
    torch.manual_seed(42)
    B, S, D, M, C = 2, 8, 64, 32, 4
    mem = MergedMemoryDNN(hidden_dim=D, memory_dim=M, mlp_dim=64, moe_k=2, num_caps=C)
    bank = mem.init_bank(B, torch.device('cpu'))
    m = mem.init_memory(B, torch.device('cpu'))
    
    for step in range(3):
        hs = torch.randn(B, S, D)
        m, inj, bank = mem(hs, bank, m)
        assert m.shape == (B, C, M//C), f"step {step}"
        assert inj.shape == (B, S, D)
    
    refined = mem.refine(bank)
    assert refined.shape == (B, M)
    pg = mem.precision_gates()
    assert pg.shape == (C,)
    print(f"[OK] MergedMemoryDNN smoke: refined {tuple(refined.shape)}, gates {pg.detach().numpy().round(3)}")


# -- Training (reuse from compare_archs) ------------------------------------

def make_merged_model(vocab_size, cfg):
    """Wrap MergedMemoryDNN into LoopedTransformer."""
    m = LoopedTransformer(
        vocab_size=vocab_size, dim=cfg.dim, num_heads=cfg.num_heads,
        num_loops=cfg.num_loops, max_seq_len=cfg.seq_len,
        use_memory_pathway=True, memory_dim=cfg.mem_dim, use_timestep_emb=True,
    )
    # Replace with merged
    m.use_capability_moe = True
    m.capability_moe = MergedMemoryDNN(
        hidden_dim=cfg.dim, memory_dim=cfg.mem_dim,
        mlp_dim=cfg.mem_dim * 4, moe_k=cfg.moe_k,
        num_caps=cfg.num_caps, max_loops=cfg.num_loops,
    )
    m.initial_memory = m.capability_moe.initial_memory
    return m


def train_merged(cfg, train_ds, val_ds, device="cpu"):
    print(f"\n{'='*50}\nmerged (memory_moe + expert_moe)\n{'='*50}")
    model = make_merged_model(train_ds.vocab_size, cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Params: {nparams:,}")
    
    loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.bs, drop_last=True)
    
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
                print(f"  step {step:5d} loss {out.loss.item():.4f}", end="")
                if step > 0 and step % cfg.eval_every == 0:
                    vl = evaluate(model, val_loader, device)
                    print(f"  val {vl:.4f}", end="")
                    if vl < best_val: best_val = vl
                print()
            step += 1
    
    val_loss = evaluate(model, val_loader, device)
    t = time.time() - t0
    print(f"Done. Val loss: {val_loss:.4f}, time: {t:.1f}s")
    return val_loss, nparams, t


@torch.no_grad()
def evaluate(model, loader, device, n=20):
    model.eval()
    t, c = 0.0, 0
    for x, y in loader:
        t += model(x.to(device), labels=y.to(device)).loss.item()
        c += 1
        if c >= n: break
    return t / c


@dataclass
class ExpConfig:
    dim: int = 64; num_heads: int = 4; num_loops: int = 4
    mem_dim: int = 32; num_caps: int = 4; moe_k: int = 2
    seq_len: int = 64; bs: int = 16
    lr: float = 3e-4; wd: float = 0.1; warmup: int = 50
    steps: int = 200; grad_clip: float = 1.0
    log_every: int = 20; eval_every: int = 100


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExpConfig(steps=200)
    
    # Data
    path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
    with open(path) as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(len(data) * 0.9)
    class DS(Dataset):
        def __init__(self, d):
            self.ex = [(d[i:i+cfg.seq_len], d[i+1:i+cfg.seq_len+1]) for i in range(0, len(d)-cfg.seq_len, 32)]
            self.vocab_size = len(chars)
        def __len__(self): return len(self.ex)
        def __getitem__(self, i): return self.ex[i]
    train_ds, val_ds = DS(data[:split]), DS(data[split:])
    
    # Load previous results for comparison
    prev_path = os.path.join(os.path.dirname(__file__), "comparison_results.json")
    prev = json.load(open(prev_path)) if os.path.exists(prev_path) else {"results": []}
    
    # Train merged
    merged_loss, merged_params, merged_time = train_merged(cfg, train_ds, val_ds, device)
    
    # Full table
    print(f"\n{'='*50}")
    print(f"FINAL COMPARISON")
    print(f"{'='*50}")
    print(f"{'Variant':<25} {'Params':>10} {'Val Loss':>10} {'Time':>8}")
    print("-"*55)
    
    all_results = prev["results"] + [{"variant": "merged", "params": merged_params,
                                       "val_loss": merged_loss, "time_s": merged_time}]
    
    for r in sorted(all_results, key=lambda x: x["val_loss"]):
        print(f"{r['variant']:<25} {r['params']:>10,} {r['val_loss']:>10.4f} {r['time_s']:>8.1f}s")
    
    # Analysis
    print(f"\n=== Architecture Breakdown ===")
    print(f"Mine:   memory MoE (top-k prior) + 1 expert/cap + precision gate + refiner")
    print(f"Codex:  expert MoE (top-k experts/cap) + memory attention + precision gate + refiner")
    print(f"Merged: memory MoE → expert MoE (2-level routing) + precision gate + refiner")

    # Save merged result
    out = {"config": asdict(cfg),
           "results": all_results,
           "verdict": "merged adds expert MoE on top of memory MoE — 2-level routing per capability"}
    with open(os.path.join(os.path.dirname(__file__), "merged_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to merged_result.json")


if __name__ == "__main__":
    smoke_test()
    main()
