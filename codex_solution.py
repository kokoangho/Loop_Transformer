"""
Capability-separated precision-routed memory for looped transformers.

Design
------
Standard looped transformers reuse one block N times, but the recurrent
memory often degenerates into a single fragile state m_t <- f(m_{t-1}).  This
file implements a different memory path:

1. Capability-separated memory
   The memory vector is projected into C independent capability channels
   (default names: math, language, reasoning, memorization).  A channel has its
   own router, precision gate, expert router, and expert MLPs.  Channels share
   no router weights, so one channel can learn long-range symbolic retrieval
   while another learns short-range lexical carry-over.

2. Cross-iteration routing
   At loop step t, each channel routes over all prior memories of the same
   channel: [m_0^c, ..., m_{t-1}^c].  The top-k selected prior states are
   softly mixed, then passed through a channel-local MoE update.  This lets
   early loop iterations influence late iterations without forcing all
   information through m_{t-1}.

3. Learnable precision gates
   Each channel owns a scalar p_c in [0, 1].  The channel state is stored as
   quantized + p_c * (full_precision - quantized).  Low p_c spends INT4-like
   precision; high p_c keeps the full precision residual.  Training can add the
   provided precision_regularizer() to encourage a budget.

4. Drift correction
   After the loop, the complete bank [m_0, ..., m_N] can pass through a small
   residual DNN refiner.  The default refiner is self-attention across loop
   time with zero-initialized correction, so the initial behavior is stable and
   the network can learn to repair quantization drift.

5. LoopedTransformer integration
   LoopedTransformerWithCapabilityMemory accepts any shared block that maps
   hidden states [B, S, D] -> [B, S, D].  It bootstraps m_0 from the input,
   loops the same block N times, injects the latest memory into the hidden
   state with a learned residual gate, appends m_t to the bank, and optionally
   refines the bank once at the end.

Run this file directly for a smoke test:

    python codex_solution.py
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_CAPABILITIES = ("math", "language", "reasoning", "memorization")


def _as_capability_names(
    num_capabilities: int,
    names: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    if names is not None:
        if len(names) != num_capabilities:
            raise ValueError("capability_names length must equal num_capabilities")
        return tuple(names)

    out: List[str] = []
    for i in range(num_capabilities):
        out.append(DEFAULT_CAPABILITIES[i] if i < len(DEFAULT_CAPABILITIES) else f"capability_{i}")
    return tuple(out)


def fake_int_quantize_ste(x: torch.Tensor, bits: int = 4, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric per-vector fake quantization with a straight-through gradient."""
    if bits < 2:
        raise ValueError("bits must be >= 2")

    qmax = float(2 ** (bits - 1) - 1)
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
    dequantized = (x / scale).round().clamp(-qmax, qmax) * scale
    return x + (dequantized - x).detach()


class PrecisionGate(nn.Module):
    """
    Scalar precision allocator for one capability channel.

    output = quantized + p * (full_precision - quantized)

    p close to 0 behaves like low precision storage.  p close to 1 preserves
    full precision.  The gate is scalar by design: the channel must earn a
    higher precision budget as a capability, not per hidden feature.
    """

    def __init__(self, init_p: float = 0.55, bits: int = 4):
        super().__init__()
        init_p = min(max(float(init_p), 1e-4), 1.0 - 1e-4)
        self.logit = nn.Parameter(torch.tensor(math.log(init_p / (1.0 - init_p))))
        self.bits = bits

    @property
    def p(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q = fake_int_quantize_ste(x, bits=self.bits)
        p = self.p.to(dtype=x.dtype, device=x.device)
        mixed = q + p * (x - q)
        quant_error = (x.detach() - q.detach()).pow(2).mean()
        return mixed, {"precision": p.detach(), "quant_error": quant_error}


class ExpertMLP(nn.Module):
    """Small residual expert used inside one capability channel."""

    def __init__(self, input_dim: int, output_dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden_dim = max(output_dim, hidden_mult * output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CapabilityChannelMoE(nn.Module):
    """
    One capability channel.

    It routes over prior memories of the same capability, then runs a
    channel-private expert MoE update.  There is no parameter sharing between
    channels at this level.
    """

    def __init__(
        self,
        channel_dim: int,
        num_experts: int = 4,
        memory_top_k: int = 4,
        expert_top_k: int = 2,
        precision_init: float = 0.55,
        quant_bits: int = 4,
        expert_hidden_mult: int = 4,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if memory_top_k < 1:
            raise ValueError("memory_top_k must be >= 1")
        if expert_top_k < 1:
            raise ValueError("expert_top_k must be >= 1")

        self.channel_dim = channel_dim
        self.num_experts = num_experts
        self.memory_top_k = memory_top_k
        self.expert_top_k = min(expert_top_k, num_experts)

        self.memory_q = nn.Linear(channel_dim, channel_dim, bias=False)
        self.memory_k = nn.Linear(channel_dim, channel_dim, bias=False)
        self.memory_v = nn.Linear(channel_dim, channel_dim, bias=False)
        self.memory_out = nn.Linear(channel_dim, channel_dim, bias=False)
        self.empty_context = nn.Parameter(torch.zeros(channel_dim))

        self.router_log_temperature = nn.Parameter(torch.tensor(0.0))
        self.recency_strength = nn.Parameter(torch.tensor(-4.0))

        expert_input_dim = channel_dim * 3
        self.expert_norm = nn.LayerNorm(expert_input_dim)
        self.expert_router = nn.Linear(expert_input_dim, num_experts)
        self.experts = nn.ModuleList(
            [ExpertMLP(expert_input_dim, channel_dim, expert_hidden_mult) for _ in range(num_experts)]
        )

        self.output_norm = nn.LayerNorm(channel_dim)
        self.precision = PrecisionGate(init_p=precision_init, bits=quant_bits)

    def _route_memory(
        self,
        proposal: torch.Tensor,
        prior: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = proposal.shape[0]
        if prior is None or prior.shape[1] == 0:
            context = self.empty_context.to(dtype=proposal.dtype, device=proposal.device).expand(batch, -1)
            empty_idx = torch.empty(batch, 0, dtype=torch.long, device=proposal.device)
            empty_w = torch.empty(batch, 0, dtype=proposal.dtype, device=proposal.device)
            return context, {"memory_indices": empty_idx, "memory_weights": empty_w}

        q = self.memory_q(proposal)
        k = self.memory_k(prior)
        v = self.memory_v(prior)

        temperature = self.router_log_temperature.exp().clamp(0.05, 20.0)
        scores = (k * q.unsqueeze(1)).sum(dim=-1) / (math.sqrt(self.channel_dim) * temperature)

        steps = prior.shape[1]
        if steps > 1:
            age = torch.arange(steps - 1, -1, -1, device=prior.device, dtype=proposal.dtype)
            age = age / float(max(steps - 1, 1))
            scores = scores - F.softplus(self.recency_strength) * age.unsqueeze(0)

        k_mem = min(self.memory_top_k, steps)
        top_scores, top_idx = scores.topk(k_mem, dim=-1)
        weights = top_scores.softmax(dim=-1)
        selected_v = v.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, self.channel_dim))
        context = (weights.unsqueeze(-1) * selected_v).sum(dim=1)
        context = self.memory_out(context)
        context, precision_aux = self.precision(context)

        aux = {
            "memory_indices": top_idx.detach(),
            "memory_weights": weights.detach(),
            "context_precision": precision_aux["precision"],
            "context_quant_error": precision_aux["quant_error"],
        }
        return context, aux

    def _expert_update(
        self,
        proposal: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        expert_input = torch.cat([proposal, context, proposal - context], dim=-1)
        expert_input = self.expert_norm(expert_input)

        logits = self.expert_router(expert_input)
        top_logits, top_idx = logits.topk(self.expert_top_k, dim=-1)
        weights = top_logits.softmax(dim=-1)

        all_outputs = torch.stack([expert(expert_input) for expert in self.experts], dim=1)
        selected = all_outputs.gather(
            1,
            top_idx.unsqueeze(-1).expand(-1, -1, self.channel_dim),
        )
        update = (weights.unsqueeze(-1) * selected).sum(dim=1)

        return update, {
            "expert_indices": top_idx.detach(),
            "expert_weights": weights.detach(),
        }

    def forward(
        self,
        proposal: torch.Tensor,
        prior: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        context, route_aux = self._route_memory(proposal, prior)
        update, expert_aux = self._expert_update(proposal, context)
        candidate = self.output_norm(proposal + context + update)
        state, precision_aux = self.precision(candidate)

        aux: Dict[str, torch.Tensor] = {}
        aux.update(route_aux)
        aux.update(expert_aux)
        aux["state_precision"] = precision_aux["precision"]
        aux["state_quant_error"] = precision_aux["quant_error"]
        return state, aux


class CapabilitySeparatedMemoryCell(nn.Module):
    """
    Splits model state into capability channels and updates each channel with
    an independent memory router plus expert MoE.
    """

    def __init__(
        self,
        d_model: int,
        num_capabilities: int = 4,
        channel_dim: Optional[int] = None,
        capability_names: Optional[Sequence[str]] = None,
        num_memory_experts: int = 4,
        memory_top_k: int = 4,
        expert_top_k: int = 2,
        precision_init: float = 0.55,
        quant_bits: int = 4,
        expert_hidden_mult: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_capabilities = num_capabilities
        self.channel_dim = channel_dim or math.ceil(d_model / num_capabilities)
        self.flat_memory_dim = self.num_capabilities * self.channel_dim
        self.capability_names = _as_capability_names(num_capabilities, capability_names)

        self.summary_norm = nn.LayerNorm(d_model)
        self.input_to_channels = nn.Linear(d_model, self.flat_memory_dim)
        self.capability_embedding = nn.Parameter(torch.zeros(num_capabilities, self.channel_dim))
        self.bootstrap_norms = nn.ModuleList([nn.LayerNorm(self.channel_dim) for _ in range(num_capabilities)])

        self.channels = nn.ModuleList(
            [
                CapabilityChannelMoE(
                    channel_dim=self.channel_dim,
                    num_experts=num_memory_experts,
                    memory_top_k=memory_top_k,
                    expert_top_k=expert_top_k,
                    precision_init=precision_init,
                    quant_bits=quant_bits,
                    expert_hidden_mult=expert_hidden_mult,
                )
                for _ in range(num_capabilities)
            ]
        )

        self.flat_norm = nn.LayerNorm(self.flat_memory_dim)
        self.channels_to_model_proj = nn.Linear(self.flat_memory_dim, d_model)

    def pool_tokens(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        if attention_mask is None:
            return hidden.mean(dim=1)

        mask = attention_mask.to(dtype=hidden.dtype, device=hidden.device)
        if mask.dim() > 2:
            mask = mask.reshape(mask.shape[0], -1)
        while mask.dim() < hidden.dim():
            mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def proposals_from_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.summary_norm(self.pool_tokens(hidden, attention_mask))
        proposals = self.input_to_channels(pooled).view(
            hidden.shape[0],
            self.num_capabilities,
            self.channel_dim,
        )
        capability_bias = self.capability_embedding.unsqueeze(0).to(dtype=hidden.dtype, device=hidden.device)
        return proposals + capability_bias

    def bootstrap(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        proposals = self.proposals_from_hidden(hidden, attention_mask)
        states = []
        precisions = []
        quant_errors = []

        for c, channel in enumerate(self.channels):
            candidate = self.bootstrap_norms[c](proposals[:, c, :])
            state, aux = channel.precision(candidate)
            states.append(state)
            precisions.append(aux["precision"])
            quant_errors.append(aux["quant_error"])

        stacked = torch.stack(states, dim=1)
        aux = {
            "bootstrap_precision": torch.stack(precisions),
            "bootstrap_quant_error": torch.stack(quant_errors).mean(),
        }
        return stacked, aux

    def update(
        self,
        hidden: torch.Tensor,
        prior_bank: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if prior_bank.dim() != 4:
            raise ValueError("prior_bank must have shape [batch, time, capabilities, channel_dim]")
        proposals = self.proposals_from_hidden(hidden, attention_mask)

        states = []
        per_channel: Dict[str, Dict[str, torch.Tensor]] = {}
        for c, channel in enumerate(self.channels):
            prior_c = prior_bank[:, :, c, :]
            state, aux = channel(proposals[:, c, :], prior_c)
            states.append(state)
            per_channel[self.capability_names[c]] = aux

        return torch.stack(states, dim=1), {"channels": per_channel}

    def channels_to_model(self, channels: torch.Tensor) -> torch.Tensor:
        flat = channels.reshape(channels.shape[0], self.flat_memory_dim)
        return self.channels_to_model_proj(self.flat_norm(flat))

    def bank_to_model(self, bank: torch.Tensor) -> torch.Tensor:
        if bank.dim() != 4:
            raise ValueError("bank must have shape [batch, time, capabilities, channel_dim]")
        bsz, steps, _, _ = bank.shape
        flat = bank.reshape(bsz * steps, self.flat_memory_dim)
        full = self.channels_to_model_proj(self.flat_norm(flat))
        return full.view(bsz, steps, self.d_model)

    def precision_values(self) -> torch.Tensor:
        return torch.stack([channel.precision.p for channel in self.channels])

    def precision_regularizer(self, target_mean: Optional[float] = None) -> torch.Tensor:
        values = self.precision_values()
        if target_mean is None:
            return values.mean()
        return (values.mean() - float(target_mean)).pow(2)


def sinusoidal_time_encoding(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    half = max(dim // 2, 1)
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=dtype) * (-math.log(10000.0) / max(half - 1, 1))
    ).unsqueeze(0)
    emb = torch.cat([torch.sin(pos * freqs), torch.cos(pos * freqs)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]


class AttentionMemoryBankRefiner(nn.Module):
    """Residual self-attention over the complete loop memory bank."""

    def __init__(
        self,
        flat_memory_dim: int,
        layers: int = 1,
        heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        heads = max(1, min(heads, flat_memory_dim))
        while flat_memory_dim % heads != 0 and heads > 1:
            heads -= 1

        layer = nn.TransformerEncoderLayer(
            d_model=flat_memory_dim,
            nhead=heads,
            dim_feedforward=max(flat_memory_dim, ff_mult * flat_memory_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.correction = nn.Linear(flat_memory_dim, flat_memory_dim)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, flat_bank: torch.Tensor) -> torch.Tensor:
        pos = sinusoidal_time_encoding(
            flat_bank.shape[1],
            flat_bank.shape[2],
            flat_bank.device,
            flat_bank.dtype,
        )
        encoded = self.encoder(flat_bank + pos.unsqueeze(0))
        return flat_bank + self.correction(encoded)


class MixerMemoryBankRefiner(nn.Module):
    """Variable-length MLP-mixer style residual refiner for memory banks."""

    def __init__(self, flat_memory_dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden_dim = max(flat_memory_dim, hidden_mult * flat_memory_dim)
        self.feature_norm = nn.LayerNorm(flat_memory_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(flat_memory_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flat_memory_dim),
        )
        self.time_norm = nn.LayerNorm(flat_memory_dim)
        self.time_mixer = nn.Conv1d(
            flat_memory_dim,
            flat_memory_dim,
            kernel_size=3,
            padding=1,
            groups=flat_memory_dim,
        )
        nn.init.zeros_(self.feature_mlp[-1].weight)
        nn.init.zeros_(self.feature_mlp[-1].bias)
        nn.init.zeros_(self.time_mixer.weight)
        nn.init.zeros_(self.time_mixer.bias)

    def forward(self, flat_bank: torch.Tensor) -> torch.Tensor:
        x = flat_bank + self.feature_mlp(self.feature_norm(flat_bank))
        mixed = self.time_mixer(self.time_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + mixed


class MemoryBankRefiner(nn.Module):
    """Refines [batch, time, capabilities, channel_dim] once after looping."""

    def __init__(
        self,
        num_capabilities: int,
        channel_dim: int,
        kind: str = "attention",
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_capabilities = num_capabilities
        self.channel_dim = channel_dim
        self.flat_memory_dim = num_capabilities * channel_dim

        if kind == "attention":
            self.refiner = AttentionMemoryBankRefiner(
                self.flat_memory_dim,
                layers=layers,
                heads=heads,
                dropout=dropout,
            )
        elif kind in {"mixer", "mlp-mixer", "mlp_mixer"}:
            self.refiner = MixerMemoryBankRefiner(self.flat_memory_dim)
        else:
            raise ValueError("kind must be 'attention' or 'mixer'")

    def forward(self, bank: torch.Tensor) -> torch.Tensor:
        if bank.dim() != 4:
            raise ValueError("bank must have shape [batch, time, capabilities, channel_dim]")
        bsz, steps, caps, dim = bank.shape
        if caps != self.num_capabilities or dim != self.channel_dim:
            raise ValueError("bank capability shape does not match refiner configuration")
        flat = bank.reshape(bsz, steps, self.flat_memory_dim)
        refined = self.refiner(flat)
        return refined.view(bsz, steps, caps, dim)


@dataclass
class CapabilityMemoryOutput:
    hidden: torch.Tensor
    memory_bank: torch.Tensor
    memory_channels: torch.Tensor
    refined_memory_bank: Optional[torch.Tensor]
    aux: Dict[str, Any]


class LoopedTransformerWithCapabilityMemory(nn.Module):
    """
    Generic integration wrapper for a shared looped transformer block.

    shared_block must accept hidden states shaped [batch, sequence, d_model] and
    return either a tensor with the same shape or a tuple whose first item is
    that tensor.  This matches the core "same block looped N times" contract
    without assuming a particular tokenizer, embedding, or LM head.
    """

    def __init__(
        self,
        shared_block: nn.Module,
        d_model: int,
        num_loops: int,
        num_capabilities: int = 4,
        channel_dim: Optional[int] = None,
        capability_names: Optional[Sequence[str]] = None,
        num_memory_experts: int = 4,
        memory_top_k: int = 4,
        expert_top_k: int = 2,
        precision_init: float = 0.55,
        quant_bits: int = 4,
        refine: bool = True,
        refiner_kind: str = "attention",
        refiner_layers: int = 1,
        refiner_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_loops < 1:
            raise ValueError("num_loops must be >= 1")

        self.shared_block = shared_block
        self.d_model = d_model
        self.num_loops = num_loops
        self.memory_cell = CapabilitySeparatedMemoryCell(
            d_model=d_model,
            num_capabilities=num_capabilities,
            channel_dim=channel_dim,
            capability_names=capability_names,
            num_memory_experts=num_memory_experts,
            memory_top_k=memory_top_k,
            expert_top_k=expert_top_k,
            precision_init=precision_init,
            quant_bits=quant_bits,
        )

        self.memory_to_hidden = nn.Linear(d_model, d_model, bias=False)
        self.refined_memory_to_hidden = nn.Linear(d_model, d_model, bias=False)
        self.feedback_logit = nn.Parameter(torch.tensor(-2.0))
        self.refined_feedback_logit = nn.Parameter(torch.tensor(-2.0))

        nn.init.normal_(self.memory_to_hidden.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.refined_memory_to_hidden.weight)

        self.refiner = (
            MemoryBankRefiner(
                num_capabilities=num_capabilities,
                channel_dim=self.memory_cell.channel_dim,
                kind=refiner_kind,
                layers=refiner_layers,
                heads=refiner_heads,
                dropout=dropout,
            )
            if refine
            else None
        )
        self.output_norm = nn.LayerNorm(d_model)

    def _call_shared_block(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        block_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        kwargs = dict(block_kwargs)

        try:
            sig = inspect.signature(self.shared_block.forward)
            params = sig.parameters
        except (TypeError, ValueError):
            params = {}

        if attention_mask is not None:
            if "attention_mask" in params:
                kwargs["attention_mask"] = attention_mask
            elif "mask" in params:
                kwargs["mask"] = attention_mask

        out = self.shared_block(hidden, **kwargs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.shape != hidden.shape:
            raise ValueError("shared_block must preserve hidden shape [batch, sequence, d_model]")
        return out

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **block_kwargs: Any,
    ) -> Union[CapabilityMemoryOutput, torch.Tensor]:
        m0, bootstrap_aux = self.memory_cell.bootstrap(hidden, attention_mask)
        bank_states = [m0]
        loop_aux: List[Dict[str, Any]] = [{"bootstrap": bootstrap_aux}]

        for _ in range(self.num_loops):
            bank = torch.stack(bank_states, dim=1)
            previous_full = self.memory_cell.channels_to_model(bank_states[-1])
            feedback = torch.sigmoid(self.feedback_logit).to(dtype=hidden.dtype, device=hidden.device)
            hidden = hidden + feedback * self.memory_to_hidden(previous_full).unsqueeze(1)
            hidden = self._call_shared_block(hidden, attention_mask, block_kwargs)
            next_state, aux = self.memory_cell.update(hidden, bank, attention_mask)
            bank_states.append(next_state)
            loop_aux.append(aux)

        memory_channels = torch.stack(bank_states, dim=1)
        refined_channels = self.refiner(memory_channels) if self.refiner is not None else None
        active_channels = refined_channels if refined_channels is not None else memory_channels

        final_memory = self.memory_cell.channels_to_model(active_channels[:, -1, :, :])
        refined_feedback = torch.sigmoid(self.refined_feedback_logit).to(dtype=hidden.dtype, device=hidden.device)
        hidden = self.output_norm(hidden + refined_feedback * self.refined_memory_to_hidden(final_memory).unsqueeze(1))

        memory_bank = self.memory_cell.bank_to_model(memory_channels)
        refined_memory_bank = self.memory_cell.bank_to_model(refined_channels) if refined_channels is not None else None
        aux = {
            "loop": loop_aux,
            "precision": self.memory_cell.precision_values().detach(),
            "precision_regularizer": self.memory_cell.precision_regularizer(),
        }

        if not return_dict:
            return hidden
        return CapabilityMemoryOutput(
            hidden=hidden,
            memory_bank=memory_bank,
            memory_channels=memory_channels,
            refined_memory_bank=refined_memory_bank,
            aux=aux,
        )

    def precision_regularizer(self, target_mean: Optional[float] = None) -> torch.Tensor:
        return self.memory_cell.precision_regularizer(target_mean=target_mean)


def find_shared_loop_block(model: nn.Module) -> nn.Module:
    """
    Best-effort helper for existing LoopedTransformer implementations.

    Prefer passing the shared block explicitly.  This helper only searches
    common attribute names used by looped-transformer prototypes.
    """
    candidates = (
        "shared_block",
        "loop_block",
        "block",
        "transformer_block",
        "layer",
        "shared_layer",
    )
    for name in candidates:
        value = getattr(model, name, None)
        if isinstance(value, nn.Module):
            return value
    raise AttributeError("could not find a shared loop block on the provided model")


def wrap_existing_looped_transformer(
    model: nn.Module,
    d_model: Optional[int] = None,
    num_loops: Optional[int] = None,
    **memory_kwargs: Any,
) -> LoopedTransformerWithCapabilityMemory:
    """
    Wrap an existing model's shared block with the capability memory loop.

    This intentionally wraps the loop block, not tokenizer/embedding/head
    logic.  For a production model, call this around the internal shared block
    and keep the original input/output heads unchanged.
    """
    block = find_shared_loop_block(model)
    d_model = d_model or getattr(model, "d_model", None) or getattr(model, "n_embd", None)
    num_loops = num_loops or getattr(model, "num_loops", None) or getattr(model, "n_loops", None)
    if d_model is None:
        raise ValueError("d_model was not provided and could not be inferred")
    if num_loops is None:
        raise ValueError("num_loops was not provided and could not be inferred")
    return LoopedTransformerWithCapabilityMemory(block, d_model=d_model, num_loops=num_loops, **memory_kwargs)


class _TinySharedBlock(nn.Module):
    """Smoke-test block: a residual MLP with the same shape contract as a loop block."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.ff(self.norm(hidden))


def smoke_test() -> bool:
    torch.manual_seed(7)
    batch, seq, d_model = 2, 6, 64
    num_loops = 4

    model = LoopedTransformerWithCapabilityMemory(
        shared_block=_TinySharedBlock(d_model),
        d_model=d_model,
        num_loops=num_loops,
        num_capabilities=4,
        channel_dim=16,
        num_memory_experts=3,
        memory_top_k=3,
        expert_top_k=2,
        refine=True,
        refiner_kind="attention",
        refiner_layers=1,
        refiner_heads=4,
    )

    hidden = torch.randn(batch, seq, d_model)
    out = model(hidden)

    assert isinstance(out, CapabilityMemoryOutput)
    assert out.hidden.shape == (batch, seq, d_model)
    assert out.memory_channels.shape == (batch, num_loops + 1, 4, 16)
    assert out.memory_bank.shape == (batch, num_loops + 1, d_model)
    assert out.refined_memory_bank is not None
    assert out.refined_memory_bank.shape == out.memory_bank.shape
    assert torch.isfinite(out.hidden).all()
    assert torch.isfinite(out.memory_bank).all()

    precision = out.aux["precision"]
    assert precision.shape == (4,)
    assert ((precision >= 0.0) & (precision <= 1.0)).all()

    loss = out.hidden.pow(2).mean() + out.memory_bank.pow(2).mean() + 0.01 * model.precision_regularizer()
    loss.backward()

    for channel in model.memory_cell.channels:
        assert channel.precision.logit.grad is not None
        assert torch.isfinite(channel.precision.logit.grad).all()

    return True


if __name__ == "__main__":
    ok = smoke_test()
    print(f"smoke_test={ok}")
