"""
Capability-split MoE memory pathway for looped transformers.

Architecture (ultra):
  m_t split into C channels: [memorize | math | speak | reason]
  Each channel: own MoE router (top-k prior) + own expert MLP
  Precision gate per channel: sigmoid(learned logit) ∈ (0,1)
  Post-loop BankRefiner: small transformer over complete bank
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math

CAPABILITIES: Tuple[str, str, str, str] = ("memorize", "math", "speak", "reason")


class MemoryBank(nn.Module):
    """Tensor bank for per-step per-capability memory states."""
    
    def __init__(self, batch_size: int, n_caps: int, cap_dim: int, max_steps: int):
        super().__init__()
        device = torch.device('cpu')
        self.register_buffer('_data', torch.zeros(batch_size, max_steps, n_caps, cap_dim))
        self.size = 0
        self.max_steps = max_steps
        self.n_caps = n_caps
    
    def append(self, m: Tensor) -> None:
        if self.size >= self.max_steps:
            return
        B, C, D = m.shape
        self._data[:, self.size, :, :] = m
        self.size += 1
    
    def get_capability(self, cap_idx: int) -> Tensor:
        """Return all prior steps for one capability: [B, N, cap_dim]"""
        return self._data[:, :self.size, cap_idx, :]
    
    def get_full(self) -> Tensor:
        """Full bank: [B, N, C*cap_dim]"""
        B, _, C, D = self._data.shape
        return self._data[:, :self.size].reshape(B, -1, C * D)
    
    def to(self, device):
        self._data = self._data.to(device)
        return self


class MoERouter(nn.Module):
    """Top-k routing over prior memories for ONE capability channel."""
    
    def __init__(self, cap_dim: int, hidden_dim: int, k: int = 2):
        super().__init__()
        self.k = k
        # Score function: query(x) · key(candidate)
        self.query_proj = nn.Linear(cap_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(cap_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(cap_dim)
    
    def forward(self, x: Tensor, candidates: Tensor) -> Tensor:
        """Route query x (from hidden) to top-k candidates.
        
        x: [B, cap_dim]
        candidates: [B, N, cap_dim]
        returns: weighted context [B, cap_dim]
        """
        B, N, D = candidates.shape
        if N == 0:
            return torch.zeros_like(x)
        
        q = self.query_proj(self.norm(x)).unsqueeze(1)  # [B, 1, H]
        k = self.key_proj(self.norm(candidates))          # [B, N, H]
        
        scores = (q * k).sum(dim=-1) / math.sqrt(D)      # [B, N]
        k_eff = min(self.k, N)
        vals, idx = scores.topk(k_eff, dim=-1)
        w = F.softmax(vals, dim=-1)                       # [B, k]
        
        gathered = torch.gather(
            candidates, 1,
            idx.unsqueeze(-1).expand(-1, -1, D)
        )                                                # [B, k, D]
        return (w.unsqueeze(-1) * gathered).sum(dim=1)   # [B, D]


class CapabilityMoEMemory(nn.Module):
    """Full capability-split MoE memory for looped transformer.
    
    Forward per loop step:
      1. Split x into C channels
      2. Each channel: MoERouter(x_c, prior_c_bank) → context_c
      3. Each channel: Expert(concat(x_c, context_c)) → m_c
      4. m_c *= sigmoid(precision_logit_c)  ← learnable precision
      5. Restore: concat(m_0..m_C) → injection into residual
      
    Post-loop:
      refine(): TransformerEncoder over full bank → corrected output
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        mlp_dim: int,
        moe_k: int = 2,
        num_caps: int = 4,
        max_loops: int = 16,
    ):
        super().__init__()
        assert memory_dim % num_caps == 0, f"memory_dim ({memory_dim}) must divide by {num_caps}"
        
        self.num_caps = num_caps
        self.cap_dim = memory_dim // num_caps
        self.moe_k = moe_k
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        
        # Per-capability routers + experts
        self.routers = nn.ModuleList([
            MoERouter(self.cap_dim, mlp_dim, k=moe_k)
            for _ in range(num_caps)
        ])
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.cap_dim * 2 + hidden_dim),
                nn.Linear(self.cap_dim * 2 + hidden_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, self.cap_dim),
            )
            for _ in range(num_caps)
        ])
        
        # Learnable precision gates (one per capability)
        self.precision_logits = nn.Parameter(torch.zeros(num_caps))
        
        # Projection: memory_dim → hidden_dim for residual injection
        self.project = nn.Linear(memory_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + memory_dim),
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Post-loop refiner: small transformer over bank
        ref_layer = nn.TransformerEncoderLayer(
            d_model=memory_dim, nhead=min(4, memory_dim // 16 or 1),
            dim_feedforward=memory_dim * 2, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.refiner = nn.TransformerEncoder(ref_layer, num_layers=2)
        self.refiner_norm = nn.LayerNorm(memory_dim)
        
        # Initial memory (learned per-capability)
        self.initial_memory = nn.Parameter(torch.zeros(1, num_caps, self.cap_dim))
    
    def precision_gates(self) -> Tensor:
        return torch.sigmoid(self.precision_logits)  # [C]
    
    def forward(
        self,
        hidden_states: Tensor,
        bank: MemoryBank,
        memory_state: Tensor,
    ) -> Tuple[Tensor, Tensor, MemoryBank]:
        """One loop step.
        
        hidden_states: [B, S, D] — token residual
        bank: MemoryBank with prior m_0..m_{t-1}
        memory_state: [B, C, cap_dim] — previous m_{t-1}
        
        returns: (next_memory [B, C, cap_dim], injection [B, S, D], bank)
        """
        B, S, D = hidden_states.shape
        C = self.num_caps
        cd = self.cap_dim
        
        # Pool hidden to global vector
        pooled = hidden_states.mean(dim=1)  # [B, D]
        
        # Process each capability channel
        next_mem_caps = []
        for c in range(C):
            # Query from previous memory + pooled hidden
            query = torch.cat([memory_state[:, c], pooled], dim=-1)  # [B, cap_dim + D]
            
            # Route over prior same-capability memory
            candidates = bank.get_capability(c)  # [B, N, cap_dim]
            # Use previous memory (not query) as routing key for consistency
            context = self.routers[c](memory_state[:, c], candidates)
            
            # Expert update
            expert_in = torch.cat([memory_state[:, c], context, pooled], dim=-1)
            m_new = self.experts[c](expert_in)  # [B, cap_dim]
            
            # Precision scaling
            m_new = m_new * torch.sigmoid(self.precision_logits[c])
            
            next_mem_caps.append(m_new)
        
        # Stack: [B, C, cap_dim]
        next_memory = torch.stack(next_mem_caps, dim=1)
        
        # Append to bank
        bank.append(next_memory)
        
        # Gated injection into residual
        mem_flat = next_memory.reshape(B, -1)  # [B, mem_dim]
        proj = self.project(mem_flat).unsqueeze(1)  # [B, 1, D]
        expanded_mem = mem_flat.unsqueeze(1).expand(-1, S, -1)
        g = self.gate(torch.cat([hidden_states, expanded_mem], dim=-1))
        injection = g * proj
        
        return next_memory, injection, bank
    
    def refine(self, bank: MemoryBank) -> Tensor:
        """Post-loop DNN refinement over complete bank.
        
        Returns: refined memory [B, memory_dim] (final state after correction)
        """
        if bank.size == 0:
            return self.initial_memory.reshape(1, -1).expand(1, -1).squeeze(0)
        
        full = bank.get_full()  # [B, N, mem_dim]
        refined = self.refiner(full)  # [B, N, mem_dim]
        last = refined[:, -1]  # [B, mem_dim]
        return self.refiner_norm(last)
    
    def init_bank(self, batch_size: int, device: torch.device) -> MemoryBank:
        bank = MemoryBank(batch_size, self.num_caps, self.cap_dim, 16)
        bank.to(device)
        return bank
    
    def init_memory(self, batch_size: int, device: torch.device) -> Tensor:
        return self.initial_memory.expand(batch_size, -1, -1).to(device)


def smoke_test():
    torch.manual_seed(42)
    B, S, D, M, C = 2, 8, 64, 32, 4
    
    mem = CapabilityMoEMemory(hidden_dim=D, memory_dim=M, mlp_dim=64, moe_k=2, num_caps=C)
    
    bank = mem.init_bank(B, torch.device('cpu'))
    m = mem.init_memory(B, torch.device('cpu'))
    
    assert m.shape == (B, C, M // C), f"init_memory: {m.shape}"
    
    for step in range(3):
        hs = torch.randn(B, S, D)
        m, inj, bank = mem(hs, bank, m)
        assert m.shape == (B, C, M // C), f"m step {step}: {m.shape}"
        assert inj.shape == (B, S, D), f"inj step {step}: {inj.shape}"
        assert bank.size == step + 1
    
    # Post-loop refinement
    refined = mem.refine(bank)
    assert refined.shape == (B, M), f"refined: {refined.shape}"
    
    # Precision gates
    pg = mem.precision_gates()
    assert pg.shape == (C,), f"precision gates: {pg.shape}"
    assert (pg > 0).all() and (pg < 1).all(), f"gates out of (0,1): {pg}"
    
    # Bank capability access
    for c in range(C):
        cap_bank = bank.get_capability(c)
        assert cap_bank.shape == (B, 3, M // C)
    
    print("[OK] CapabilityMoEMemory smoke test")
    print(f"  precision gates: {pg.detach().numpy().round(3)}")
    print(f"  memory_trace: {bank.size} steps, refined: {tuple(refined.shape)}")


if __name__ == "__main__":
    smoke_test()
