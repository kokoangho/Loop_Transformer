"""
Mixed-domain dataset: Shakespeare + math + code + reasoning.
Each encoded as char-level LM, but domains are tagged for capability routing.
"""
import math, os, random
import torch
from torch.utils.data import Dataset

SEED = 42
random.seed(SEED)

# -- Math problems ----------------------------------------------------------

MATH_PROBLEMS = [
    ("2 + 3 = ", "5"),
    ("7 - 4 = ", "3"),
    ("6 * 3 = ", "18"),
    ("15 / 3 = ", "5"),
    ("2 + 2 = ", "4"),
    ("10 - 7 = ", "3"),
    ("4 * 4 = ", "16"),
    ("20 / 4 = ", "5"),
    ("x + 3 = 7, x = ", "4"),
    ("2 * x = 12, x = ", "6"),
    ("x - 5 = 3, x = ", "8"),
    ("x / 2 = 6, x = ", "12"),
    ("3 + 4 * 2 = ", "11"),
    ("(2 + 3) * 4 = ", "20"),
    ("10 - 3 * 2 = ", "4"),
    ("sum of 1 to 5 = ", "15"),
    ("average of 2,4,6 = ", "4"),
    ("2^3 = ", "8"),
    ("sqrt of 9 = ", "3"),
    ("3! = ", "6"),
]

# -- Code snippets ----------------------------------------------------------

CODE_SNIPPETS = [
    ("def add(a, b): return a + ", "b"),
    ("x = 5; y = x + ", "3"),
    ("for i in range(3): print(", "i)"),
    ("if x > 0: print('", "positive')"),
    ("while n > 0: n = n - ", "1"),
    ("list.append(", "item)"),
    ("return x * ", "x"),
    ("class Dog: def bark(self): print('", "woof')"),
    ("try: x = 1/0\nexcept: print('", "error')"),
    ("import math; print(math.", "pi)"),
]

# -- Reasoning problems -----------------------------------------------------

REASONING_PROBLEMS = [
    ("if A > B and B > C then A > ", "C"),
    ("all men are mortal, socrates is a man, so socrates is ", "mortal"),
    ("if it rains then ground wet, ground not wet so it did not ", "rain"),
    ("every square has 4 sides, this shape has 3 sides so it is a ", "triangle"),
    ("if today is monday then tomorrow is ", "tuesday"),
    ("a > b and b > c implies a is greater than ", "c"),
    ("1, 2, 4, 8, 16, next is ", "32"),
    ("odd one out: cat dog table bird -> ", "table"),
    ("if x = 2 and x + y = 5 then y = ", "3"),
    ("all birds fly, penguin is a bird but cannot ", "fly"),
]

# -- Build mixed dataset ----------------------------------------------------

def make_mixed_texts():
    """Return list of (text_string, domain_label) tuples."""
    texts = []
    for prob, ans in MATH_PROBLEMS:
        texts.append((prob + ans, "math"))
    for code, ans in CODE_SNIPPETS:
        texts.append((code + ans, "code"))
    for prob, ans in REASONING_PROBLEMS:
        texts.append((prob + ans, "reason"))
    return texts


def get_mixed_chars(texts):
    """Get all unique characters across full dataset."""
    all_text = "".join(t for t, _ in texts)
    return sorted(list(set(all_text)))


def get_mixed_data(seq_len=64, stride=32, val_split=0.1):
    """Return train_ds, val_ds with char-level LM over mixed domains."""
    # Shakespeare (baseline language data)
    shak_path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
    with open(shak_path) as f:
        shak_text = f.read()[:50000]  # use 50K chars for speed
    
    # Domain texts
    domains_texts = make_mixed_texts()
    
    # Build vocab from ALL sources
    all_sources = [shak_text] + [t for t, _ in domains_texts]
    chars = sorted(list(set("".join(all_sources))))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    
    # Also assign domain labels to shakespeare examples
    class MixedDS(Dataset):
        def __init__(self, base_text, domain_texts, seq_len, stride):
            self.seq_len = seq_len
            self.examples = []
            # Shakespeare examples get "speak" domain tag
            base_data = torch.tensor([stoi[c] for c in base_text], dtype=torch.long)
            for i in range(0, len(base_data) - seq_len, stride):
                self.examples.append((base_data[i:i+seq_len], base_data[i+1:i+seq_len+1], "speak"))
            # Domain-specific examples
            for text, domain in domain_texts:
                if len(text) <= seq_len:
                    data = torch.tensor([stoi[c] for c in text.ljust(seq_len+1)], dtype=torch.long)
                else:
                    data = torch.tensor([stoi[c] for c in text[:seq_len+1]], dtype=torch.long)
                self.examples.append((data[:-1], data[1:], domain))
            self.vocab_size = vocab_size
            self.stoi = stoi
        def __len__(self): return len(self.examples)
        def __getitem__(self, i):
            x, y, d = self.examples[i]
            return x, y, d
    
    # Split
    split = int(len(shak_text) * (1 - val_split))
    train_text = shak_text[:split]
    val_text = shak_text[split:]
    
    train_ds = MixedDS(train_text, domains_texts, seq_len, stride)
    val_ds = MixedDS(val_text, [], seq_len, stride)  # val: only shakespeare
    return train_ds, val_ds


# -- Multi-expert MoE memory (mine + codex fusion) -------------------------

import sys
sys.path.insert(0, os.path.dirname(__file__))
from typing import List, Optional, Sequence, Tuple
import torch.nn.functional as F
from torch import Tensor, nn
from moe_memory_pathway import MemoryBank

CAPABILITIES: Tuple[str, str, str, str] = ("memorize", "math", "speak", "reason")


class MultiExpertCapabilityMoE(nn.Module):
    """One capability: memory MoE (mine) → expert MoE (codex) + precision gate.
    
    Memory MoE: top-k over prior same-capability memories (attention scoring)
    Expert MoE: top-k over 4 expert MLPs within the capability
    Precision gate: sigmoid(logit) controls how much expert output flows
    """
    
    def __init__(self, cap_dim: int, hidden_dim: int, num_experts: int = 4,
                 memory_k: int = 2, expert_k: int = 2, hidden_dim_mult: int = 4):
        super().__init__()
        self.memory_k = memory_k
        self.expert_k = expert_k
        self.cap_dim = cap_dim
        
        # Level 1: Memory MoE (attention scoring from Codex)
        self.mem_norm_q = nn.LayerNorm(cap_dim)
        self.mem_norm_k = nn.LayerNorm(cap_dim)
        
        # Level 2: Expert MoE (router from Codex)
        expert_hidden = cap_dim * 2 * hidden_dim_mult
        self.expert_router = nn.Linear(cap_dim * 2, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(cap_dim * 2),
                nn.Linear(cap_dim * 2, expert_hidden),
                nn.GELU(),
                nn.Linear(expert_hidden, cap_dim),
            )
            for _ in range(num_experts)
        ])
        
        # Precision gate
        self.precision_logit = nn.Parameter(torch.zeros(()))
    
    def _read_memory(self, x: Tensor, prior: Sequence[Tensor]) -> Tensor:
        """Top-k memory routing via attention scoring."""
        B = x.shape[0]
        if len(prior) == 0:
            return torch.zeros_like(x)
        candidates = torch.stack(list(prior), dim=1)  # [B, N, D]
        q = self.mem_norm_q(x).unsqueeze(1)
        k = self.mem_norm_k(candidates)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.cap_dim)
        k_eff = min(self.memory_k, candidates.shape[1])
        vals, idx = scores.topk(k_eff, dim=-1)
        w = F.softmax(vals, dim=-1).unsqueeze(-1)
        gathered = torch.gather(candidates, 1, idx.unsqueeze(-1).expand(-1, -1, self.cap_dim))
        return (w * gathered).sum(dim=1)
    
    def forward(self, x: Tensor, prior: Sequence[Tensor]) -> Tensor:
        """Level 1: memory MoE → Level 2: expert MoE → precision gate."""
        context = self._read_memory(x, prior)
        routed = torch.cat([x, context], dim=-1)
        
        # Expert MoE
        logits = self.expert_router(routed)
        k = min(self.expert_k, logits.shape[-1])
        top_l, top_i = logits.topk(k, dim=-1)
        top_w = F.softmax(top_l, dim=-1)
        
        expert_outs = torch.stack([e(routed) for e in self.experts], dim=-2)
        selected = torch.gather(expert_outs, 1, top_i.unsqueeze(-1).expand(-1, -1, self.cap_dim))
        mixed = (selected * top_w.unsqueeze(-1)).sum(dim=1)
        
        return x + torch.sigmoid(self.precision_logit) * mixed


class MultiExpertMemoryDNN(nn.Module):
    """Full multi-expert memory DNN. Drops into LoopedTransformer.
    
    Per loop step:
      1. Split hidden into C capabilities
      2. Each capability: MultiExpertCapabilityMoE (memory MoE → expert MoE)
      3. Concat → gated injection → residual
    Post-loop: BankRefiner over full memory bank.
    """
    
    def __init__(self, hidden_dim: int, memory_dim: int, mlp_dim: int,
                 moe_k: int = 2, num_caps: int = 4, max_loops: int = 16,
                 experts_per_cap: int = 4):
        super().__init__()
        assert memory_dim % num_caps == 0
        self.num_caps = num_caps
        self.cap_dim = memory_dim // num_caps
        self.moe_k = moe_k
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        
        self.capability_moes = nn.ModuleDict({
            name: MultiExpertCapabilityMoE(
                cap_dim=self.cap_dim,
                hidden_dim=mlp_dim,
                num_experts=experts_per_cap,
                memory_k=moe_k,
                expert_k=moe_k,
            )
            for name in CAPABILITIES[:num_caps]
        })
        
        self.project = nn.Linear(memory_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + memory_dim),
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Post-loop refiner
        ref_layer = nn.TransformerEncoderLayer(
            d_model=memory_dim, nhead=min(4, memory_dim // 16 or 1),
            dim_feedforward=memory_dim * 2, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.refiner = nn.TransformerEncoder(ref_layer, num_layers=2)
        self.refiner_norm = nn.LayerNorm(memory_dim)
        
        self.initial_memory = nn.Parameter(torch.zeros(1, num_caps, self.cap_dim))
    
    def precision_gates(self) -> Tensor:
        return torch.stack([torch.sigmoid(m.precision_logit) for m in self.capability_moes.values()])
    
    def forward(self, hidden_states: Tensor, bank: MemoryBank,
                memory_state: Tensor) -> Tuple[Tensor, Tensor, MemoryBank]:
        B, S, D = hidden_states.shape
        pooled = hidden_states.mean(dim=1)
        
        next_mem_caps = []
        for c_idx, name in enumerate(CAPABILITIES[:self.num_caps]):
            prev = memory_state[:, c_idx]
            query = prev + pooled[:, :self.cap_dim]  # blend prev mem + hidden
            
            prior_bank = bank.get_capability(c_idx)
            prior_list = [prior_bank[:, i] for i in range(prior_bank.shape[1])]
            
            m_new = self.capability_moes[name](query, prior_list)
            next_mem_caps.append(m_new)
        
        next_memory = torch.stack(next_mem_caps, dim=1)
        bank.append(next_memory)
        
        # Gated injection
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


# -- Training ---------------------------------------------------------------

import time
import json
from dataclasses import dataclass, field
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    dim: int = 64; num_heads: int = 4; num_loops: int = 8  # more loops for deeper reasoning
    mem_dim: int = 32; num_caps: int = 4; moe_k: int = 2
    experts_per_cap: int = 4
    seq_len: int = 64; bs: int = 16
    lr: float = 3e-4; wd: float = 0.1; warmup: int = 200
    steps: int = 3000; grad_clip: float = 1.0
    log_every: int = 300; eval_every: int = 500


def make_model(vocab_size, cfg):
    """Build LoopedTransformer with multi-expert MoE memory."""
    from looped_transformer import LoopedTransformer
    m = LoopedTransformer(
        vocab_size=vocab_size, dim=cfg.dim, num_heads=cfg.num_heads,
        num_loops=cfg.num_loops, max_seq_len=cfg.seq_len,
        use_memory_pathway=True, memory_dim=cfg.mem_dim, use_timestep_emb=True,
    )
    m.use_capability_moe = True
    m.capability_moe = MultiExpertMemoryDNN(
        hidden_dim=cfg.dim, memory_dim=cfg.mem_dim,
        mlp_dim=cfg.mem_dim * 4, moe_k=cfg.moe_k,
        num_caps=cfg.num_caps, max_loops=cfg.num_loops,
        experts_per_cap=cfg.experts_per_cap,
    )
    m.initial_memory = m.capability_moe.initial_memory
    return m


@torch.no_grad()
def evaluate(model, loader, device, n=10):
    model.eval()
    t, c = 0.0, 0
    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        t += model(x.to(device), labels=y.to(device)).loss.item()
        c += 1
        if c >= n: break
    return t / c


def train(cfg, train_ds, val_ds, device="cpu"):
    print(f"\n{'='*60}")
    print(f"Multi-Expert MoE Memory — {cfg.steps} steps, {cfg.num_loops} loops, {cfg.experts_per_cap} experts/cap")
    print(f"{'='*60}")
    
    model = make_model(train_ds.vocab_size, cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Params: {nparams:,}")
    
    loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.bs, drop_last=True)
    
    # Eval baseline before training
    # (model at init is random)
    
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
    train_losses, val_losses, steps_log = [], [], []
    t0 = time.time()
    step = 0
    
    while step < cfg.steps:
        for batch in loader:
            if step >= cfg.steps: break
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            
            out = model(x.to(device), labels=y.to(device))
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()
            
            if step % cfg.log_every == 0:
                vl = evaluate(model, val_loader, device)
                train_losses.append(out.loss.item())
                val_losses.append(vl)
                steps_log.append(step)
                if vl < best_val: best_val = vl
                pg = model.capability_moe.precision_gates()
                print(f"  step {step:5d} | train {out.loss.item():.4f} | val {vl:.4f} | gates {pg.detach().cpu().numpy().round(3)}")
            
            step += 1
    
    final_val = evaluate(model, val_loader, device)
    t = time.time() - t0
    print(f"\nDone. Best val: {best_val:.4f}, Final val: {final_val:.4f}, time: {t:.1f}s")
    print(f"Precision gates: {model.capability_moe.precision_gates().detach().cpu().numpy().round(3)}")
    
    return {
        "params": nparams,
        "best_val": best_val,
        "final_val": final_val,
        "time_s": round(t, 1),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "steps": steps_log,
        "precision_gates": model.capability_moe.precision_gates().detach().cpu().numpy().round(3).tolist(),
    }, model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    cfg = TrainConfig(steps=3000)
    
    print("Building mixed dataset (speak + math + code + reason)...")
    train_ds, val_ds = get_mixed_data(seq_len=cfg.seq_len, stride=32)
    print(f"Vocab: {train_ds.vocab_size}, Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    result, model = train(cfg, train_ds, val_ds, device)
    
    # Save
    out = {"config": {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
           "result": result}
    path = os.path.join(os.path.dirname(__file__), "multi_expert_result.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {path}")
