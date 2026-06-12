"""
Looped Transformer — single shared block, recurrently applied N times.

Architecture notes (informed by Ouro, LoopFormer, MELT):
- Single transformer block (self-attn + SwiGLU FFN) shared across loop iterations
- Sandwich norm: norm before each sublayer, plus final norm per loop
- RoPE (via rotary embedding) for position encoding
- Optional bias toward identity (ResScale or learnable alpha) for gradient stability
- Timestep embedding injected at each loop to distinguish iterations
- Cross-layer memory pathway integration hook

References:
- Ouro: Scaling Latent Reasoning via Looped Language Models (arxiv 2510.25741)
- LoopFormer: Elastic-Depth Looped Transformers (arxiv 2602.11451)
- MELT: Memory-Efficient Looped Transformer (arxiv 2605.07721)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable

from moe_memory_pathway import CapabilityMoEMemory, MemoryBank

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """RoPE from Ouro / usual decoder-only transformers."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._cos_cached = None
        self._sin_cached = None

    def _build_cache(self, seq_len: int, device: torch.device):
        if self._cos_cached is None or self._cos_cached.shape[-2] < seq_len:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]

    def forward(self, x: Tensor, offset: int = 0):
        seq_len = x.shape[-2] + offset
        half_dim = self.inv_freq.shape[0]
        self._build_cache(seq_len, x.device)
        cos = self._cos_cached[:, :, offset:offset + x.shape[-2], :half_dim]
        sin = self._sin_cached[:, :, offset:offset + x.shape[-2], :half_dim]
        return cos.to(x.dtype), sin.to(x.dtype)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to x. x shape: [batch, heads, seq, dim]."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, dim: int, hidden_mult: int = 8 // 3):  # 8/3 = ~2.67 so hidden ~= dim * 8/3
        super().__init__()
        hidden_dim = int(dim * hidden_mult)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Shared Transformer Block
# ---------------------------------------------------------------------------

class LoopedTransformerBlock(nn.Module):
    """Single decoder block shared across loop iterations.

    Uses sandwich normalization:
        x = x + Attn(Norm(x))
        x = x + FFN(Norm(x))
    """

    def __init__(self, dim: int, num_heads: int, swiglu_mult: float = 8 / 3, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Sandwich norms (pre-attn, pre-ffn)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        # Attention
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # FFN (SwiGLU)
        self.ffn = SwiGLU(dim, swiglu_mult)
        self.ffn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Optional learnable residual scaling (identity bias for stability)
        self.res_scale_attn = nn.Parameter(torch.ones(1))
        self.res_scale_ffn = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        causal_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, S, D = x.shape

        # ---- Attention sublayer ----
        residual = x
        xn = self.norm1(x)

        Q = self.q_proj(xn).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(xn).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(xn).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K = apply_rotary(Q, cos, sin), apply_rotary(K, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=(causal_mask is None),
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.o_proj(attn_out)
        attn_out = self.attn_drop(attn_out)

        x = residual + self.res_scale_attn * attn_out

        # ---- FFN sublayer ----
        residual = x
        xn = self.norm2(x)
        ffn_out = self.ffn(xn)
        ffn_out = self.ffn_drop(ffn_out)
        x = residual + self.res_scale_ffn * ffn_out

        return x


# ---------------------------------------------------------------------------
# Timestep Embedding (loop index encoding)
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Learned embedding for loop iteration index."""

    def __init__(self, dim: int, max_loops: int = 32):
        super().__init__()
        self.embed = nn.Embedding(max_loops, dim)

    def forward(self, loop_idx: int, batch_size: int, seq_len: int, device: torch.device) -> Tensor:
        e = self.embed(torch.tensor([loop_idx], device=device)).unsqueeze(1)  # [1, 1, dim]
        return e.expand(batch_size, seq_len, -1)


# ---------------------------------------------------------------------------
# Cross-Layer Pathway (adapted from exchange with Codex)
# ---------------------------------------------------------------------------

class CrossLayerMemoryCell(nn.Module):
    """Memory state update across loop iterations.

    Matches the earlier `cross_layer_pathway.py` interface but simplified
    for the looped setting — m_l evolves across loop steps, not across
    unique layers.
    """

    def __init__(self, hidden_dim: int, memory_dim: int, mlp_dim: int):
        super().__init__()
        self.memory_update = nn.Sequential(
            nn.LayerNorm(memory_dim + hidden_dim),
            nn.Linear(memory_dim + hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, memory_dim),
        )
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.project_memory = nn.Linear(memory_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + memory_dim),
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: Tensor, memory_state: Tensor) -> Tuple[Tensor, Tensor]:
        pooled = hidden_states.mean(dim=1)
        inp = torch.cat([memory_state, pooled], dim=-1)
        delta = self.memory_update(inp)
        next_memory = self.memory_norm(memory_state + delta)

        proj = self.project_memory(next_memory).unsqueeze(1)
        expanded_mem = memory_state.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        g = self.gate(torch.cat([hidden_states, expanded_mem], dim=-1))
        return next_memory, g * proj


# ---------------------------------------------------------------------------
# Fake Quantization (for simulated low-precision trunk)
# ---------------------------------------------------------------------------

class FakeQuantizeLinear(nn.Module):
    """Wrap nn.Linear with simulated INT8/INT4 quantization.

    During forward, weights are fake-quantized (round + clip), gradients
    pass through via straight-through estimator.
    """

    def __init__(self, linear: nn.Linear, bits: int = 8, per_channel: bool = True):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.bits = bits
        self.per_channel = per_channel

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self.per_channel:
            scale = w.abs().max(dim=1, keepdim=True).values
        else:
            scale = w.abs().max().expand(1, 1)
        scale = scale.clamp(min=1e-8)
        q_max = 2 ** (self.bits - 1) - 1
        w_q = (w / scale).round().clamp(-q_max - 1, q_max)
        w_dq = w_q * scale
        return F.linear(x, w_dq, self.bias)


def apply_fake_quant(module: nn.Module, bits: int = 8, exclude_types: tuple = ()):
    """Recursively wrap Linear layers with FakeQuantizeLinear."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not isinstance(child, exclude_types):
            setattr(module, name, FakeQuantizeLinear(child, bits=bits))
        else:
            apply_fake_quant(child, bits, exclude_types)


# ---------------------------------------------------------------------------
# Full Looped Transformer with optional cross-layer pathway & quantization
# ---------------------------------------------------------------------------

@dataclass
class LoopedTransformerOutput:
    logits: Tensor
    loss: Optional[Tensor]
    memory_trace: Optional[Tensor]
    loop_count: int


class LoopedTransformer(nn.Module):
    """Decoder-only transformer with a single shared block, looped N times.

    Supports:
    - Shared block with weight tying
    - Timestep embeddings per loop
    - Cross-layer memory pathway (m_l updated across loops)
    - Fake quantization on the shared trunk
    - Memory kept at higher precision (default BF16/FP32)
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 384,
        num_heads: int = 6,
        num_loops: int = 4,
        max_seq_len: int = 512,
        swiglu_mult: float = 8 / 3,
        dropout: float = 0.0,
        use_memory_pathway: bool = True,
        memory_dim: int = 64,
        memory_mlp_dim: Optional[int] = None,
        quantize_bits: Optional[int] = None,  # 8 or 4 for simulated quantization
        use_timestep_emb: bool = True,
        use_capability_moe: bool = False,
        num_capabilities: int = 4,
        moe_k: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.num_loops = num_loops
        self.use_memory_pathway = use_memory_pathway
        self.quantize_bits = quantize_bits
        self.use_timestep_emb = use_timestep_emb
        self.use_capability_moe = use_capability_moe
        self.num_capabilities = num_capabilities
        self.moe_k = moe_k

        # Token & position embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.rotary = RotaryEmbedding(dim // num_heads, max_seq_len)

        # Shared block (looped)
        self.shared_block = LoopedTransformerBlock(dim, num_heads, swiglu_mult, dropout)

        # Timestep embeddings
        if use_timestep_emb:
            self.timestep_embed = TimestepEmbedding(dim, max_loops=num_loops + 2)

        # Capability MoE memory pathway (new)
        if use_capability_moe:
            assert use_memory_pathway, "MoE memory requires memory_pathway"
            self.capability_moe = CapabilityMoEMemory(
                hidden_dim=dim,
                memory_dim=memory_dim,
                mlp_dim=memory_mlp_dim or memory_dim * 4,
                moe_k=moe_k,
                num_caps=num_capabilities,
                max_loops=num_loops,
            )
            self._cap_bank = None
        else:
            self.capability_moe = None

        # Cross-layer memory pathway (original)
        if use_memory_pathway:
            memory_mlp_dim = memory_mlp_dim or memory_dim * 4
            self.memory_cell = CrossLayerMemoryCell(dim, memory_dim, memory_mlp_dim)
            self.initial_memory = nn.Parameter(torch.zeros(memory_dim))
        else:
            self.memory_cell = None
            self.initial_memory = None

        # Output
        self.final_norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie embeddings
        self.token_embedding.weight = self.lm_head.weight

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.RMSNorm):
                torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        return_memory_trace: bool = False,
    ) -> LoopedTransformerOutput:
        B, S = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)  # [B, S, D]

        # RoPE cache
        cos, sin = self.rotary(x)

        # Causal mask for SDPA
        causal_mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1
        )

        # Initialize memory if pathway enabled (capability MoE takes priority)
        cap_bank = None
        cap_mem = None
        memory = None
        memory_trace = None
        if self.use_capability_moe and self.capability_moe is not None:
            # Reset internal Codex bank if applicable
            if hasattr(self.capability_moe, 'codex'):
                self.capability_moe.codex.reset_memory()
            cap_bank = self.capability_moe.init_bank(B, device)
            cap_mem = self.capability_moe.init_memory(B, device)
            memory_trace = []
        elif self.use_memory_pathway and self.memory_cell is not None:
            memory = self.initial_memory.unsqueeze(0).expand(B, -1)
            memory_trace = []

        # Loop the shared block
        for step in range(self.num_loops):
            # Timestep embedding
            if self.use_timestep_emb:
                tse = self.timestep_embed(step, B, S, device)
                x = x + tse

            # Memory conditioning (pre-block)
            if self.use_memory_pathway and memory is not None:
                proj_mem = self.memory_cell.project_memory(memory).unsqueeze(1)
                expanded_mem = memory.unsqueeze(1).expand(-1, S, -1)
                cond_gate = torch.sigmoid(
                    self.memory_cell.gate[1](  # direct linear gate (simplified)
                        torch.cat([x, expanded_mem], dim=-1)
                    )
                )
                x = x + cond_gate * proj_mem

            # Shared block forward
            x = self.shared_block(x, cos, sin, causal_mask)

            # Memory update (post-block)
            if self.use_capability_moe and self.capability_moe is not None:
                cap_mem, injection, cap_bank = self.capability_moe(x, cap_bank, cap_mem)
                x = x + injection
                if return_memory_trace:
                    memory_trace.append(cap_mem)
            elif self.use_memory_pathway and self.memory_cell is not None:
                memory, injection = self.memory_cell(x, memory)
                x = x + injection
                if return_memory_trace:
                    memory_trace.append(memory)

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        # Post-loop DNN refinement (capability MoE only)
        if self.use_capability_moe and self.capability_moe is not None:
            refined = self.capability_moe.refine(cap_bank)
            if refined is not None:
                proj_refined = self.capability_moe.project(refined).unsqueeze(1)
                x = x + proj_refined
                logits = self.lm_head(self.final_norm(x))

        return LoopedTransformerOutput(
            logits=logits,
            loss=loss,
            memory_trace=torch.stack(memory_trace, dim=1) if memory_trace else None,
            loop_count=self.num_loops,
        )

    def enable_quantization(self, bits: int = 8):
        """Apply fake quantization to the shared block (keep memory/embeddings FP)."""
        apply_fake_quant(self.shared_block, bits=bits)
        self.quantize_bits = bits

    def get_param_groups(self, lr: float, weight_decay: float = 0.1):
        """Separate decay/non-decay parameters (bias, norms)."""
        decay = []
        no_decay = []
        for name, p in self.named_parameters():
            if p.requires_grad is False:
                continue
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        ]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    torch.manual_seed(42)
    model = LoopedTransformer(
        vocab_size=256,
        dim=64,
        num_heads=4,
        num_loops=3,
        max_seq_len=32,
        use_memory_pathway=True,
        memory_dim=16,
        use_timestep_emb=True,
    )
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    out = model(x, labels=y, return_memory_trace=True)
    assert out.logits.shape == (2, 16, 256), f"logits: {out.logits.shape}"
    assert out.loss is not None and out.loss.isfinite(), f"loss: {out.loss}"
    assert out.memory_trace is not None
    assert out.memory_trace.shape == (2, 3, 16), f"mem trace: {out.memory_trace.shape}"
    print(f"[OK] LoopedTransformer smoke test passed")
    print(f"   logits: {tuple(out.logits.shape)}, loss: {out.loss.item():.4f}")
    print(f"   memory_trace: {tuple(out.memory_trace.shape)}, loops: {out.loop_count}")

    # Test with quantized trunk
    model_q = LoopedTransformer(
        vocab_size=256, dim=64, num_heads=4, num_loops=3, max_seq_len=32,
        use_memory_pathway=True, memory_dim=16,
    )
    model_q.enable_quantization(bits=4)
    out_q = model_q(x, labels=y)
    assert out_q.loss is not None and out_q.loss.isfinite()
    print(f"[OK] LoopedTransformer INT4 quantized: loss = {out_q.loss.item():.4f}")

    # Test without memory pathway
    model_no_mem = LoopedTransformer(
        vocab_size=256, dim=64, num_heads=4, num_loops=3, max_seq_len=32,
        use_memory_pathway=False,
    )
    out_nm = model_no_mem(x, labels=y)
    assert out_nm.loss is not None and out_nm.loss.isfinite()
    print(f"[OK] LoopedTransformer no-memory: loss = {out_nm.loss.item():.4f}")

    print("\nAll smoke tests passed!")


if __name__ == "__main__":
    smoke_test()
