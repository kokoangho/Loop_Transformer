"""
Training experiment: compare 4 transformer architectures on tiny character LM.

Variants:
  1. Standard deep transformer (unique layers) — baseline
  2. Looped transformer (shared block) — parameter-efficient
  3. Looped + cross-layer memory pathway — our proposal
  4. Looped + pathway + INT4 quantized trunk — precision-efficiency extreme

All models matched in hidden dimension and total compute (FLOPs-matched).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from looped_transformer import (
    LoopedTransformer,
    LoopedTransformerBlock,
    RotaryEmbedding,
    SwiGLU,
)

import numpy as np

# ---------------------------------------------------------------------------
# Tiny character-level dataset (Shakespeare)
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharDataset(Dataset):
    """Character-level language modeling dataset."""

    def __init__(self, text: str, seq_len: int = 128, stride: int = 64):
        chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_len = seq_len
        self.stride = stride

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.examples = []
        for i in range(0, len(data) - seq_len, stride):
            x = data[i:i + seq_len]
            y = data[i + 1:i + seq_len + 1]
            self.examples.append((x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_shakespeare(seq_len: int = 128, stride: int = 64, val_split: float = 0.1):
    """Load Shakespeare dataset, split into train/val."""
    import urllib.request
    path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
    if not os.path.exists(path):
        print("Downloading Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Build vocab from FULL text to ensure train/val match
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)

    # Split
    split_idx = int(len(text) * (1 - val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    class FixedCharDataset(Dataset):
        def __init__(self, text_subset, seq_len, stride):
            data = torch.tensor([stoi[c] for c in text_subset], dtype=torch.long)
            self.examples = []
            for i in range(0, len(data) - seq_len, stride):
                self.examples.append((data[i:i + seq_len], data[i + 1:i + seq_len + 1]))
            self.vocab_size = vocab_size
            self.stoi = stoi
            self.itos = itos
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return self.examples[idx]

    train_ds = FixedCharDataset(train_text, seq_len, stride)
    val_ds = FixedCharDataset(val_text, seq_len, stride)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Standard Deep Transformer (baseline)
# ---------------------------------------------------------------------------

class DeepTransformerBlock(nn.Module):
    """Standard transformer block (not shared)."""

    def __init__(self, dim: int, num_heads: int, swiglu_mult: float = 8 / 3, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.ffn = SwiGLU(dim, swiglu_mult)
        self.rotary = RotaryEmbedding(dim // num_heads)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, causal_mask: Tensor) -> Tensor:
        B, S, D = x.shape
        residual = x
        xn = self.norm1(x)

        Q = self.q_proj(xn).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(xn).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(xn).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        from looped_transformer import apply_rotary
        Q, K = apply_rotary(Q, cos, sin), apply_rotary(K, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=causal_mask, dropout_p=0.0, is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.o_proj(attn_out)

        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


class DeepTransformer(nn.Module):
    """Standard deep transformer with N unique layers (baseline)."""

    def __init__(self, vocab_size: int, dim: int = 384, num_heads: int = 6,
                 num_layers: int = 4, max_seq_len: int = 512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.rotary = RotaryEmbedding(dim // num_heads, max_seq_len)
        self.blocks = nn.ModuleList([
            DeepTransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None):
        B, S = input_ids.shape
        x = self.token_embedding(input_ids)
        cos, sin = self.rotary(x)
        causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
        for block in self.blocks:
            x = block(x, cos, sin, causal_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100) if labels is not None else None
        return logits, loss


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Model
    dim: int = 128
    num_heads: int = 4
    vocab_size: int = 65  # shakespeare chars
    max_seq_len: int = 128
    # Loop/shared
    num_loops_or_layers: int = 4
    memory_dim: int = 32
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 2000
    warmup_steps: int = 200
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 100
    # Architecture variant
    variant: str = "looped+pathway"  # deep | looped | looped+pathway | looped+pathway+int4


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_model(variant: str, config: ExperimentConfig, vocab_size: int) -> nn.Module:
    """Create model for the given variant."""
    if variant == "deep":
        return DeepTransformer(
            vocab_size=vocab_size,
            dim=config.dim,
            num_heads=config.num_heads,
            num_layers=config.num_loops_or_layers,
            max_seq_len=config.max_seq_len,
        )
    elif variant == "looped":
        return LoopedTransformer(
            vocab_size=vocab_size,
            dim=config.dim,
            num_heads=config.num_heads,
            num_loops=config.num_loops_or_layers,
            max_seq_len=config.max_seq_len,
            use_memory_pathway=False,
            use_timestep_emb=True,
        )
    elif variant == "looped+pathway":
        return LoopedTransformer(
            vocab_size=vocab_size,
            dim=config.dim,
            num_heads=config.num_heads,
            num_loops=config.num_loops_or_layers,
            max_seq_len=config.max_seq_len,
            use_memory_pathway=True,
            memory_dim=config.memory_dim,
            use_timestep_emb=True,
        )
    elif variant == "looped+pathway+int4":
        model = LoopedTransformer(
            vocab_size=vocab_size,
            dim=config.dim,
            num_heads=config.num_heads,
            num_loops=config.num_loops_or_layers,
            max_seq_len=config.max_seq_len,
            use_memory_pathway=True,
            memory_dim=config.memory_dim,
            use_timestep_emb=True,
        )
        model.enable_quantization(bits=4)
        return model
    else:
        raise ValueError(f"Unknown variant: {variant}")


@dataclass
class ExperimentResult:
    config: dict
    variant: str
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    wall_time: float = 0.0
    param_count: int = 0
    final_val_loss: float = 0.0
    converged: bool = False


def run_experiment(config: ExperimentConfig, train_ds: Dataset, val_ds: Dataset,
                   device: str = "cpu") -> ExperimentResult:
    """Run a single training experiment."""
    print(f"\n{'='*60}")
    print(f"Variant: {config.variant}")
    print(f"{'='*60}")

    model = create_model(config.variant, config, train_ds.vocab_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=True)

    # Optimizer with weight decay groups
    if isinstance(model, LoopedTransformer):
        param_groups = model.get_param_groups(config.learning_rate, config.weight_decay)
    else:
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [
            {"params": decay, "weight_decay": config.weight_decay, "lr": config.learning_rate},
            {"params": no_decay, "weight_decay": 0.0, "lr": config.learning_rate},
        ]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.max_steps)

    result = ExperimentResult(config=asdict(config), variant=config.variant, param_count=param_count)
    best_val_loss = float("inf")
    start_time = time.time()
    step = 0
    epoch = 0

    model.train()
    while step < config.max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= config.max_steps:
                break

            x, y = [t.to(device) for t in batch]

            if isinstance(model, DeepTransformer):
                _, loss = model(x, labels=y)
            else:
                out = model(x, labels=y)
                loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % config.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                result.train_losses.append(loss.item())
                result.steps.append(step)
                print(f"  step {step:5d} | lr {current_lr:.2e} | train loss {loss.item():.4f}")

            if step % config.eval_every == 0 and step > 0:
                val_loss = evaluate(model, val_loader, device, config.variant)
                result.val_losses.append(val_loss)
                print(f"         | val loss   {val_loss:.4f} (best: {min(best_val_loss, val_loss):.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                model.train()

            step += 1

    result.wall_time = time.time() - start_time
    result.final_val_loss = best_val_loss
    result.converged = best_val_loss < 5.0  # reasonable threshold for char LM
    print(f"  Done. Best val loss: {best_val_loss:.4f}, time: {result.wall_time:.1f}s")
    return result


@torch.no_grad()
def evaluate(model, loader, device, variant):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x, y = [t.to(device) for t in batch]
        if variant == "deep":
            _, loss = model(x, labels=y)
        else:
            out = model(x, labels=y)
            loss = out.loss
        total_loss += loss.item()
        n += 1
        if n >= 20:  # limit eval batches for speed
            break
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Config — reduced for CPU speed
    config = ExperimentConfig(
        dim=64,
        num_heads=4,
        num_loops_or_layers=4,
        memory_dim=16,
        max_seq_len=64,
        batch_size=16,
        max_steps=500,
        learning_rate=3e-4,
        warmup_steps=100,
        grad_clip=1.0,
        log_every=100,
        eval_every=200,
    )

    # Data
    print("Loading Shakespeare...")
    train_ds, val_ds = load_shakespeare(seq_len=config.max_seq_len, stride=64)
    config.vocab_size = train_ds.vocab_size
    print(f"Vocab size: {config.vocab_size}, Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")

    # Run all 4 variants
    variants = ["deep", "looped", "looped+pathway", "looped+pathway+int4"]
    results = []

    for variant in variants:
        config.variant = variant
        result = run_experiment(config, train_ds, val_ds, device=device)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<25} {'Params':>10} {'Val Loss':>10} {'Time (s)':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.final_val_loss):
        print(f"{r.variant:<25} {r.param_count:>10,} {r.final_val_loss:>10.4f} {r.wall_time:>10.1f}")

    # Save results
    import json
    output = {
        "config": asdict(config),
        "results": [
            {
                "variant": r.variant,
                "params": r.param_count,
                "final_val_loss": r.final_val_loss,
                "wall_time_sec": r.wall_time,
                "train_losses": r.train_losses,
                "val_losses": r.val_losses,
                "steps": r.steps,
            }
            for r in results
        ],
    }
    out_path = os.path.join(os.path.dirname(__file__), "experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Quick analysis print
    print("\nKey finding:")
    base = [r for r in results if r.variant == "deep"][0]
    for r in results:
        if r.variant == "deep":
            continue
        delta = r.final_val_loss - base.final_val_loss
        pct_savings = (1 - r.param_count / base.param_count) * 100
        print(f"  {r.variant}: {delta:+.4f} loss vs deep, {pct_savings:.0f}% fewer params")


if __name__ == "__main__":
    main()
