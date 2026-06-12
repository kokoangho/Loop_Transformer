import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CAPABILITIES: Tuple[str, str, str, str] = ("memorize", "math", "speak", "reason")


class CapabilityMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_experts: int = 4,
        expert_top_k: int = 2,
        memory_top_k: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.expert_top_k = min(expert_top_k, num_experts)
        self.memory_top_k = memory_top_k
        hidden_dim = hidden_dim or dim * 4

        self.router = nn.Linear(dim * 2, num_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_experts)
        )
        self.precision_logit = nn.Parameter(torch.zeros(()))
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)

    def _read_prior(self, x: Tensor, prior: Sequence[Tensor]) -> Tensor:
        if not prior:
            return torch.zeros_like(x)

        bank = torch.stack(list(prior), dim=1)
        query = self.norm(x).unsqueeze(1)
        keys = self.context_norm(bank)
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.dim)

        k = min(self.memory_top_k, bank.shape[1])
        values, indices = scores.topk(k, dim=1)
        weights = F.softmax(values, dim=1).unsqueeze(-1)
        gathered = torch.gather(bank, 1, indices.unsqueeze(-1).expand(*indices.shape, self.dim))
        return (gathered * weights).sum(dim=1)

    def forward(self, x: Tensor, prior: Sequence[Tensor]) -> Tensor:
        context = self._read_prior(x, prior)
        routed = torch.cat([x, context], dim=-1)

        logits = self.router(routed)
        k = min(self.expert_top_k, logits.shape[-1])
        top_logits, top_idx = logits.topk(k, dim=-1)
        top_weights = F.softmax(top_logits, dim=-1)

        expert_outputs = torch.stack([expert(routed) for expert in self.experts], dim=-2)
        selected = torch.gather(
            expert_outputs,
            -2,
            top_idx.unsqueeze(-1).expand(*top_idx.shape, self.dim),
        )
        mixed = (selected * top_weights.unsqueeze(-1)).sum(dim=-2)
        return x + torch.sigmoid(self.precision_logit) * mixed


class BankRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: int = 2,
        heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        while dim % heads != 0 and heads > 1:
            heads -= 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, bank: Sequence[Tensor]) -> Tensor:
        if not bank:
            raise RuntimeError("Cannot refine an empty memory bank.")

        stacked = torch.stack(list(bank), dim=1)
        if stacked.dim() == 3:
            refined = self.encoder(stacked)
            return self.norm(refined[:, -1])

        batch = stacked.shape[0]
        steps = stacked.shape[1]
        middle = stacked.shape[2:-1]
        dim = stacked.shape[-1]
        seq = int(torch.tensor(middle).prod().item()) if middle else 1
        tokens = stacked.reshape(batch, steps * seq, dim)
        refined = self.encoder(tokens).reshape(batch, steps, *middle, dim)
        return self.norm(refined[:, -1])


class CodexSolution(nn.Module):
    """Capability-split loop hook for LoopedTransformer-style recurrent passes."""

    def __init__(
        self,
        dim: Optional[int] = None,
        d_model: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_experts: int = 4,
        expert_top_k: int = 2,
        memory_top_k: int = 2,
        expert_hidden_dim: Optional[int] = None,
        refinement_layers: int = 2,
        refinement_heads: int = 4,
        dropout: float = 0.0,
        auto_reset: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = dim or d_model or hidden_size
        if self.dim is None:
            raise ValueError("Pass dim, d_model, or hidden_size.")
        if self.dim % len(CAPABILITIES) != 0:
            raise ValueError(f"dim must be divisible by {len(CAPABILITIES)} capabilities.")

        self.capability_dim = self.dim // len(CAPABILITIES)
        self.capabilities = CAPABILITIES
        self.capability_moes = nn.ModuleDict(
            {
                name: CapabilityMoE(
                    self.capability_dim,
                    hidden_dim=expert_hidden_dim,
                    num_experts=num_experts,
                    expert_top_k=expert_top_k,
                    memory_top_k=memory_top_k,
                    dropout=dropout,
                )
                for name in self.capabilities
            }
        )
        self.refiner = BankRefiner(
            self.dim,
            layers=refinement_layers,
            heads=refinement_heads,
            dropout=dropout,
        )
        self.memory_bank: List[Tensor] = []
        self.capability_banks = {name: [] for name in self.capabilities}
        self.auto_reset = auto_reset

    @property
    def precision_gates(self) -> Tensor:
        return torch.stack(
            [torch.sigmoid(self.capability_moes[name].precision_logit) for name in self.capabilities]
        )

    def reset_memory(self) -> None:
        self.memory_bank.clear()
        for bank in self.capability_banks.values():
            bank.clear()

    reset = reset_memory

    def step(self, x: Tensor, step: Optional[int] = None, store: bool = True) -> Tensor:
        del step
        chunks = x.split(self.capability_dim, dim=-1)
        outputs = []
        for name, chunk in zip(self.capabilities, chunks):
            outputs.append(self.capability_moes[name](chunk, self.capability_banks[name]))

        out = torch.cat(outputs, dim=-1)
        if store:
            self.memory_bank.append(out)
            for name, chunk in zip(self.capabilities, outputs):
                self.capability_banks[name].append(chunk)
        return out

    def refine(self, bank: Optional[Sequence[Tensor]] = None) -> Tensor:
        return self.refiner(self.memory_bank if bank is None else bank)

    finalize = refine

    def hook(self, x: Tensor, step: Optional[int] = None, is_last: bool = False, **_: object) -> Tensor:
        y = self.step(x, step=step)
        return self.refine() if is_last else y

    def loop_hook(self, *args: object, **kwargs: object) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        x = kwargs.pop("x", None)
        for name in ("hidden", "hidden_state", "state", "memory", "m", "h", "output"):
            if x is None:
                x = kwargs.pop(name, None)
        step = kwargs.pop("step", kwargs.pop("iteration", kwargs.pop("t", None)))
        n_steps = kwargs.pop("n_steps", kwargs.pop("num_steps", kwargs.pop("n_iterations", None)))
        final = bool(
            kwargs.pop("final", False)
            or kwargs.pop("is_final", False)
            or kwargs.pop("is_last", False)
            or kwargs.pop("last", False)
            or kwargs.pop("done", False)
            or kwargs.pop("complete", False)
        )
        reset = bool(kwargs.pop("reset", False))
        store = bool(kwargs.pop("store", True))

        if reset:
            self.reset_memory()

        for arg in args:
            if isinstance(arg, Tensor) and x is None:
                x = arg
            elif isinstance(arg, int) and step is None:
                step = arg

        if x is None:
            return self.refine()

        if isinstance(step, int) and isinstance(n_steps, int):
            final = final or step == n_steps - 1
        if self.auto_reset and isinstance(step, int) and step == 0 and self.memory_bank:
            self.reset_memory()

        y = self.step(x, step=step if isinstance(step, int) else None, store=store)
        return self.refine() if final else y

    def as_forward_hook(self):
        def _hook(_module: nn.Module, _inputs: Tuple[object, ...], output: Tensor) -> Tensor:
            return self.step(output)

        return _hook


Solution = CodexSolution
CodexSolutionHook = CodexSolution
LoopedTransformerHook = CodexSolution


def build_hook(**kwargs: object) -> CodexSolution:
    return CodexSolution(**kwargs)


def build_model(**kwargs: object) -> CodexSolution:
    return CodexSolution(**kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)
    hook = CodexSolution(dim=64, num_experts=4, expert_top_k=2, memory_top_k=2)
    x = torch.randn(2, 5, 64)
    hook.reset_memory()
    for i in range(4):
        x = hook(x, step=i)
        assert x.shape == (2, 5, 64)
    y = hook(final=True)
    assert y.shape == (2, 5, 64)
    assert len(hook.memory_bank) == 4
    assert hook.precision_gates.shape == (4,)
    print("smoke ok", tuple(y.shape), "bank", len(hook.memory_bank))
