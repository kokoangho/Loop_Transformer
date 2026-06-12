"""Cross-layer deep pathway prototype for decoder-only transformers.

This file implements the conceptual recurrence:

    x_{l+1} = Block_l(x_l, condition=m_l)
    m_{l+1} = DNN_l(m_l, summary(x_l))
    x_{l+1} += gate_l(x_l, m_l) * project(m_{l+1})

The goal is clarity, not production completeness.  The module wraps the
per-layer stack of a GPT-2-style transformer and adds a learned memory state
that is updated across depth.  That memory is then projected back into the
token hidden dimension and injected into the residual stream under a gate.

If Hugging Face `transformers` is installed, `CrossLayerPathwayGPT2` uses
`GPT2Model` and its real GPT-2 blocks.  If it is not installed, the smoke test
uses a small PyTorch-only transformer fallback so the architecture can still be
run without downloading anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

try:
    from transformers import GPT2Config, GPT2Model
except ImportError:  # pragma: no cover - optional research dependency.
    GPT2Config = None
    GPT2Model = None


@dataclass
class CrossLayerOutput:
    """Return type for the prototype forward pass.

    Attributes:
        last_hidden_state: Final token representations, shape [batch, seq, dim].
        memory_state: Final cross-layer memory state, shape [batch, memory_dim].
        memory_trace: Memory after each depth step, shape [batch, layers, memory_dim].
    """

    last_hidden_state: Tensor
    memory_state: Tensor
    memory_trace: Tensor


class CrossLayerMemoryCell(nn.Module):
    """One depth step of the cross-layer DNN pathway.

    The cell reads the previous memory state and a pooled summary of the current
    residual stream.  It emits:

    - the next memory state m_{l+1}
    - a gated residual injection derived from m_{l+1}

    The gate is token-wise: every token receives the same projected global
    memory vector, but the amount of injection is decided from that token's
    hidden state plus the previous global memory.
    """

    def __init__(self, hidden_dim: int, memory_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.memory_update = nn.Sequential(
            nn.LayerNorm(memory_dim + hidden_dim),
            nn.Linear(memory_dim + hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, memory_dim),
        )
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.project_memory = nn.Linear(memory_dim, hidden_dim)
        self.condition_memory = nn.Linear(memory_dim, hidden_dim)
        self.condition_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + memory_dim),
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + memory_dim),
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def condition_block_input(self, hidden_states: Tensor, memory_state: Tensor) -> Tensor:
        """Condition the next transformer block on the current memory m_l."""

        projected_memory = self.condition_memory(memory_state).unsqueeze(1)
        expanded_memory = memory_state.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        gate_input = torch.cat([hidden_states, expanded_memory], dim=-1)
        return hidden_states + self.condition_gate(gate_input) * projected_memory

    def forward(self, hidden_states: Tensor, memory_state: Tensor) -> Tuple[Tensor, Tensor]:
        """Update memory and produce the gated residual injection.

        Args:
            hidden_states: Token residual stream x_l, shape [batch, seq, hidden].
            memory_state: Previous memory m_l, shape [batch, memory_dim].

        Returns:
            A tuple `(next_memory, residual_injection)`.
        """

        pooled_hidden = hidden_states.mean(dim=1)
        memory_input = torch.cat([memory_state, pooled_hidden], dim=-1)

        memory_delta = self.memory_update(memory_input)
        next_memory = self.memory_norm(memory_state + memory_delta)

        projected_memory = self.project_memory(next_memory).unsqueeze(1)
        expanded_memory = memory_state.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        gate_input = torch.cat([hidden_states, expanded_memory], dim=-1)
        gate = self.gate(gate_input)

        return next_memory, gate * projected_memory


class TinyDecoderBlock(nn.Module):
    """Small causal transformer block used only as a no-dependency fallback."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, hidden_states: Tensor, causal_mask: Optional[Tensor]) -> Tensor:
        attn_input = self.attn_norm(hidden_states)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + attn_output
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class CrossLayerPathwayTransformer(nn.Module):
    """PyTorch-only decoder transformer with the cross-layer memory pathway.

    This class is intentionally small and transparent.  It demonstrates the
    architecture without relying on Hugging Face internals.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        max_position_embeddings: int = 128,
        memory_dim: int = 64,
        transformer_mlp_dim: Optional[int] = None,
        memory_mlp_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        transformer_mlp_dim = transformer_mlp_dim or hidden_dim * 4
        memory_mlp_dim = memory_mlp_dim or memory_dim * 4

        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TinyDecoderBlock(hidden_dim, num_heads, transformer_mlp_dim)
                for _ in range(num_layers)
            ]
        )
        self.memory_cells = nn.ModuleList(
            [
                CrossLayerMemoryCell(hidden_dim, memory_dim, memory_mlp_dim)
                for _ in range(num_layers)
            ]
        )
        self.initial_memory = nn.Parameter(torch.zeros(memory_dim))
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: Tensor, memory_state: Optional[Tensor] = None) -> CrossLayerOutput:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)

        if memory_state is None:
            memory_state = self.initial_memory.unsqueeze(0).expand(batch_size, -1)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

        memory_trace = []
        for block, memory_cell in zip(self.blocks, self.memory_cells):
            residual_input = hidden_states
            hidden_states = memory_cell.condition_block_input(hidden_states, memory_state)
            hidden_states = block(hidden_states, causal_mask)
            memory_state, injection = memory_cell(residual_input, memory_state)
            hidden_states = hidden_states + injection
            memory_trace.append(memory_state)

        hidden_states = self.final_norm(hidden_states)
        return CrossLayerOutput(
            last_hidden_state=hidden_states,
            memory_state=memory_state,
            memory_trace=torch.stack(memory_trace, dim=1),
        )


class CrossLayerPathwayGPT2(nn.Module):
    """GPT-2 wrapper with an added cross-layer deep pathway.

    The standard GPT-2 block stack is preserved.  After each block, a separate
    memory MLP updates a global state and injects a gated projection of that
    state into the residual stream.

    This is a conceptual prototype:
    - it supports `input_ids` only, not every GPT-2 generation feature;
    - it assumes decoder-style causal self-attention;
    - it returns hidden states and memory traces rather than logits.
    """

    def __init__(
        self,
        config: Optional["GPT2Config"] = None,
        memory_dim: int = 128,
        memory_mlp_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        if GPT2Config is None or GPT2Model is None:
            raise ImportError(
                "CrossLayerPathwayGPT2 requires `transformers`. "
                "Use CrossLayerPathwayTransformer for a PyTorch-only prototype."
            )

        config = config or GPT2Config(
            n_layer=num_layers or 12,
            n_embd=768,
            n_head=12,
            n_positions=1024,
            n_ctx=1024,
            vocab_size=50257,
        )
        if num_layers is not None:
            config.n_layer = num_layers

        self.transformer = GPT2Model(config)
        self.hidden_dim = config.n_embd
        self.memory_dim = memory_dim
        self.num_layers = config.n_layer
        memory_mlp_dim = memory_mlp_dim or memory_dim * 4

        self.memory_cells = nn.ModuleList(
            [
                CrossLayerMemoryCell(config.n_embd, memory_dim, memory_mlp_dim)
                for _ in range(config.n_layer)
            ]
        )
        self.initial_memory = nn.Parameter(torch.zeros(memory_dim))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        memory_state: Optional[Tensor] = None,
    ) -> CrossLayerOutput:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if memory_state is None:
            memory_state = self.initial_memory.unsqueeze(0).expand(batch_size, -1)

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = (
            self.transformer.wte(input_ids)
            + self.transformer.wpe(position_ids)
        )
        hidden_states = self.transformer.drop(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        memory_trace = []
        for block, memory_cell in zip(self.transformer.h, self.memory_cells):
            residual_input = hidden_states
            hidden_states = memory_cell.condition_block_input(hidden_states, memory_state)
            block_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = block_outputs[0]
            memory_state, injection = memory_cell(residual_input, memory_state)
            hidden_states = hidden_states + injection
            memory_trace.append(memory_state)

        hidden_states = self.transformer.ln_f(hidden_states)
        return CrossLayerOutput(
            last_hidden_state=hidden_states,
            memory_state=memory_state,
            memory_trace=torch.stack(memory_trace, dim=1),
        )


def smoke_test() -> None:
    """Run a simple forward pass and assert the important tensor shapes."""

    torch.manual_seed(0)

    model = CrossLayerPathwayTransformer(
        vocab_size=128,
        hidden_dim=32,
        num_layers=3,
        num_heads=4,
        max_position_embeddings=16,
        memory_dim=12,
    )
    input_ids = torch.randint(0, 128, (2, 8))
    output = model(input_ids)

    assert output.last_hidden_state.shape == (2, 8, 32)
    assert output.memory_state.shape == (2, 12)
    assert output.memory_trace.shape == (2, 3, 12)
    assert torch.isfinite(output.last_hidden_state).all()
    assert torch.isfinite(output.memory_state).all()

    print("smoke_test passed")
    print(f"last_hidden_state: {tuple(output.last_hidden_state.shape)}")
    print(f"memory_state: {tuple(output.memory_state.shape)}")
    print(f"memory_trace: {tuple(output.memory_trace.shape)}")


if __name__ == "__main__":
    smoke_test()
