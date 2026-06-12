# MoE-Routed Inter-Layer DNN Communication Pathway — Implementation Plan

> **For Hermes:** Use subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace the current single-vector sequential memory pathway with a full DNN memory bank where loop iterations communicate via an MoE (Mixture-of-Experts) routing mechanism. Each loop step selects top-k previous memory states to read from, enabling flexible cross-iteration information flow.

**Architecture:**
- **Memory Bank** — Store all loop-step memory states as a tensor M = [m_0, m_1, ..., m_{N-1}], one per loop iteration
- **MoE Router** — At loop step `t`, a gated network with top-k sparsity selects which prior memories are relevant and aggregates them
- **Blended Update** — The selected memories are combined with the current hidden state to produce m_t

**Tech Stack:** PyTorch, existing `LoopedTransformer` in `C:\Loop-transformer\`

**Status Quo (current):** `CrossLayerMemoryCell` updates a single `memory` vector sequentially with no cross-iteration visibility. Each step reads only the immediately previous `memory` + pooled hidden state — no MoE, no inter-layer communication.

---

## Implementation Plan

### Task 1: Create `MoEMemoryBank` — Memory bank storing all loop states

**Objective:** A module that accumulates memory state vectors across loop iterations into a tensor bank.

**Files:**
- Create: `C:\Loop-transformer\moe_memory_pathway.py`
- Test: built-in smoke test in same file

**Step 1: Write MemoryBank class**

```python
class MemoryBank(nn.Module):
    """Stores a growing tensor of per-loop memory states.
    
    Maintains M = [m_0, m_1, ..., m_{t-1}] across loop steps.
    Provides read access to any prior state by index.
    """
    def __init__(self, batch_size: int, memory_dim: int, max_loops: int, device: torch.device):
        super().__init__()
        self.bank = torch.zeros(batch_size, max_loops, memory_dim, device=device)
        self.size = 0
    
    def append(self, m: Tensor):
        """Store m at position self.size, increment counter."""
        self.bank[:, self.size] = m
        self.size += 1
    
    def read(self, indices: Tensor) -> Tensor:
        """Read memory states at given indices. indices: [batch, k]"""
        # Gather from bank using batch indices
        return ...  # shape [batch, k, memory_dim]
    
    def all_prior(self) -> Tensor:
        """Return all stored states: [batch, size, memory_dim]"""
        return self.bank[:, :self.size]
    
    def reset(self):
        self.size = 0
```

**Step 2: Smoke test**

```python
bank = MemoryBank(2, 16, 8, device)
m0 = torch.randn(2, 16)
bank.append(m0)
m1 = torch.randn(2, 16)
bank.append(m1)
assert bank.size == 2
all_m = bank.all_prior()  # [2, 2, 16]
```

---

### Task 2: Create `MoERouter` — Top-k sparse routing over memory candidates

**Objective:** An MoE gating network that takes a query vector and a set of candidate memory vectors, then returns top-k selected indices and weights.

**Files:**
- Modify: `C:\Loop-transformer\moe_memory_pathway.py`
- Test: built-in smoke test

**Step 1: Write MoERouter class**

```python
class MoERouter(nn.Module):
    """Top-k MoE gating over memory bank candidates.
    
    For each query (the current hidden state or previous memory),
    scores all candidate memories and selects top-k via softmax.
    
    Architecture:
        score = W2 * GELU(W1 * concat(query, candidate))
        noise = softplus(W_noise * concat(query, candidate)) * N(0,1)  # load balancing
        top_k = topk(score + noise, k)
        weights = softmax(top_k_scores)
    """

    def __init__(self, query_dim: int, candidate_dim: int, hidden_dim: int, k: int = 2):
        super().__init__()
        self.k = k
        self.w_proj = nn.Linear(query_dim + candidate_dim, hidden_dim, bias=False)
        self.w_score = nn.Linear(hidden_dim, 1, bias=False)
        self.w_noise = nn.Linear(query_dim + candidate_dim, 1, bias=False) if k < candidate_dim else None
    
    def forward(self, query: Tensor, candidates: Tensor, num_candidates: int) -> Tuple[Tensor, Tensor]:
        """Route query to top-k candidates.
        
        Args:
            query: [batch, query_dim]
            candidates: [batch, N, candidate_dim]
            num_candidates: int (may be less than candidates.size(1) if bank not full)
        
        Returns:
            weights: [batch, k]
            indices: [batch, k]  (indices into candidates)
        """
        B, N, D = candidates.shape
        
        # Expand query: [B, 1, Q] -> [B, N, Q]
        q_expanded = query.unsqueeze(1).expand(-1, N, -1)
        
        # Score each candidate
        inp = torch.cat([q_expanded, candidates], dim=-1)  # [B, N, Q+C]
        h = F.gelu(self.w_proj(inp))
        scores = self.w_score(h).squeeze(-1)  # [B, N]
        
        # Mask out non-existent candidates (beyond num_candidates)
        mask = torch.arange(N, device=query.device) >= num_candidates
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Top-k selection
        weights, indices = torch.topk(scores, min(self.k, num_candidates), dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        return weights, indices
```

**Step 2: Handle edge cases**
- `num_candidates < k`: reduce effective k to num_candidates
- All candidates masked: return uniform weights over available

**Step 3: Smoke test**

```python
router = MoERouter(query_dim=16, candidate_dim=16, hidden_dim=32, k=2)
q = torch.randn(2, 16)
cands = torch.randn(2, 5, 16)
w, idx = router(q, cands, num_candidates=5)
assert w.shape == (2, 2)
assert idx.shape == (2, 2)
assert torch.allclose(w.sum(dim=-1), torch.ones(2))
```

---

### Task 3: Create `MoEMemoryDNN` — Full DNN pathway with MoE-routed inter-layer communication

**Objective:** The main module that replaces `CrossLayerMemoryCell`. Maintains a memory bank across loop iterations, uses MoE routing at each step to decide which prior memories to read from, and produces the gated injection.

**Files:**
- Modify: `C:\Loop-transformer\moe_memory_pathway.py`
- Test: built-in smoke test

**Step 1: Write MoEMemoryDNN**

```python
class MoEMemoryDNN(nn.Module):
    """DNN memory pathway with MoE-routed inter-layer communication.
    
    At each loop step t:
      1. Query = pooled hidden state (mean over tokens) + previous memory m_{t-1}
      2. MoE router selects top-k prior memories from bank
      3. Context = weighted sum of selected memories
      4. New memory: m_t = MLP(concat(m_{t-1}, context, pooled_hidden))
      5. Injection: gated projection of m_t into token residual
    
    The memory bank M = [m_0, ..., m_{N-1}] is the DNN figure
    spanning all layers — every layer can now talk to every other
    layer via the MoE router.
    """

    def __init__(self, hidden_dim: int, memory_dim: int, mlp_dim: int,
                 moe_k: int = 2, max_loops: int = 16):
        super().__init__()
        self.memory_dim = memory_dim
        self.moe_k = moe_k
        
        # Memory update MLP
        self.memory_update = nn.Sequential(
            nn.LayerNorm(memory_dim * 2 + hidden_dim),  # m_{t-1}, context, pooled
            nn.Linear(memory_dim * 2 + hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, memory_dim),
        )
        self.memory_norm = nn.LayerNorm(memory_dim)
        
        # MoE router
        self.router = MoERouter(
            query_dim=memory_dim + hidden_dim,  # m_{t-1} + pooled_hidden
            candidate_dim=memory_dim,
            hidden_dim=mlp_dim,
            k=moe_k,
        )
        
        # Gated residual injection (same as before)
        self.project_memory = nn.Linear(memory_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + memory_dim),
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, hidden_states: Tensor, memory_bank: MemoryBank,
                memory_state: Tensor) -> Tuple[Tensor, Tensor, MemoryBank]:
        """One step of the MoE memory DNN.
        
        Args:
            hidden_states: Token residual at current step [B, S, D]
            memory_bank: MemoryBank with prior states
            memory_state: Previous memory m_{t-1} [B, mem_dim]
        
        Returns:
            next_memory: [B, mem_dim]
            injection: [B, S, D]  (gated projection to add to residual)
            memory_bank: updated with new memory
        """
        B, S, D = hidden_states.shape
        pooled = hidden_states.mean(dim=1)  # [B, D]
        
        # Build query: concat previous memory + pooled hidden
        query = torch.cat([memory_state, pooled], dim=-1)  # [B, mem_dim + D]
        
        # Get all prior candidates from bank
        num_candidates = memory_bank.size
        if num_candidates > 0:
            candidates = memory_bank.all_prior()  # [B, N, mem_dim]
            weights, indices = self.router(query, candidates, num_candidates)
            
            # Gather weighted context
            # indices: [B, k], candidates: [B, N, D]
            gathered = torch.gather(
                candidates, 1,
                indices.unsqueeze(-1).expand(-1, -1, self.memory_dim)
            )  # [B, k, mem_dim]
            context = (weights.unsqueeze(-1) * gathered).sum(dim=1)  # [B, mem_dim]
        else:
            context = torch.zeros_like(memory_state)
        
        # Memory update
        update_in = torch.cat([memory_state, context, pooled], dim=-1)
        delta = self.memory_update(update_in)
        next_memory = self.memory_norm(memory_state + delta)
        
        # Append to bank
        memory_bank.append(next_memory)
        
        # Gated injection (same as existing CrossLayerMemoryCell)
        proj = self.project_memory(next_memory).unsqueeze(1)
        expanded_mem = memory_state.unsqueeze(1).expand(-1, S, -1)
        g = self.gate(torch.cat([hidden_states, expanded_mem], dim=-1))
        
        return next_memory, g * proj, memory_bank
    
    def init_bank(self, batch_size: int, device: torch.device) -> MemoryBank:
        return MemoryBank(batch_size, self.memory_dim, 16, device)
```

**Step 2: Smoke test**

```python
dnn = MoEMemoryDNN(hidden_dim=64, memory_dim=16, mlp_dim=64, moe_k=2)
hs = torch.randn(2, 8, 64)
bank = dnn.init_bank(2, hs.device)
m0 = torch.zeros(2, 16)
m1, inj, bank = dnn(hs, bank, m0)
assert m1.shape == (2, 16)
assert inj.shape == (2, 8, 64)
assert bank.size == 1

m2, inj, bank = dnn(hs, bank, m1)  # 2nd step, reads m0 via MoE
assert bank.size == 2

m3, inj, bank = dnn(hs, bank, m2)  # 3rd step, reads m0,m1 via MoE
assert bank.size == 3
```

---

### Task 4: Integrate `MoEMemoryDNN` into `LoopedTransformer`

**Objective:** Replace the single `CrossLayerMemoryCell` with the new `MoEMemoryDNN`. Add `use_moe_memory` flag to select between old and new pathway.

**Files:**
- Modify: `C:\Loop-transformer\looped_transformer.py`

**Changes needed:**

1. Add import:
```python
from moe_memory_pathway import MoEMemoryDNN, MemoryBank
```

2. Add constructor parameter:
```python
use_moe_memory: bool = False,
moe_k: int = 2,
```

3. In `__init__`, conditionally create either `CrossLayerMemoryCell` or `MoEMemoryDNN`:
```python
if use_moe_memory:
    assert use_memory_pathway, "MoE memory requires memory pathway enabled"
    self.moe_memory = MoEMemoryDNN(
        hidden_dim=dim,
        memory_dim=memory_dim,
        mlp_dim=memory_mlp_dim,
        moe_k=moe_k,
        max_loops=num_loops,
    )
else:
    self.moe_memory = None
```

4. In `forward`, replace the memory loop body:
```python
# Initialize memory bank for MoE
if self.use_moe_memory and self.moe_memory is not None:
    memory_bank = self.moe_memory.init_bank(B, device)
    memory = self.initial_memory.unsqueeze(0).expand(B, -1)
    memory_trace = []
else:
    memory_bank = None
```

5. Replace the memory update logic inside the loop:
```python
if self.use_moe_memory and self.moe_memory is not None:
    memory, injection, memory_bank = self.moe_memory(x, memory_bank, memory)
    x = x + injection
    if return_memory_trace:
        memory_trace.append(memory)
else:
    # original CrossLayerMemoryCell logic...
```

**Step 5: Smoke test in existing smoke_test()**

Add a new block to the existing `smoke_test()`:
```python
# Test with MoE memory
model_moe = LoopedTransformer(
    vocab_size=256, dim=64, num_heads=4, num_loops=3, max_seq_len=32,
    use_memory_pathway=True, use_moe_memory=True, memory_dim=16, moe_k=2,
)
out_moe = model_moe(x, labels=y, return_memory_trace=True)
assert out_moe.loss is not None and out_moe.loss.isfinite()
print(f"[OK] LoopedTransformer MoE memory: loss = {out_moe.loss.item():.4f}")
assert out_moe.memory_trace.shape == (2, 3, 16)
```

---

### Task 5: Memory trace visualisation (inspect MoE routing decisions)

**Objective:** Add a method to inspect which prior memories were selected at each step.

**Files:**
- Modify: `C:\Loop-transformer\moe_memory_pathway.py`

**Step 1: Add routing trace to MoEMemoryDNN**

```python
def forward_with_trace(self, hidden_states, memory_bank, memory_state):
    """Same as forward but returns routing decisions for analysis."""
    ...
    return next_memory, injection, memory_bank, {
        'step': memory_bank.size,  # 0-indexed
        'num_candidates': num_candidates,
        'routing_weights': weights.detach(),
        'routing_indices': indices.detach(),
        'context_norm': context.norm(dim=-1).mean().item(),
    }
```

---

### Task 6: Add MoE memory variant to training experiment

**Objective:** Add `looped+moe` as a new variant in `train_experiment.py` and compare against the existing variants.

**Files:**
- Modify: `C:\Loop-transformer\train_experiment.py`

**Step 1: Add variant to `create_model()`**

```python
elif variant == "looped+moe":
    return LoopedTransformer(
        vocab_size=vocab_size,
        dim=config.dim,
        num_heads=config.num_heads,
        num_loops=config.num_loops_or_layers,
        max_seq_len=config.max_seq_len,
        use_memory_pathway=True,
        use_moe_memory=True,
        memory_dim=config.memory_dim,
        moe_k=2,
        use_timestep_emb=True,
    )
```

**Step 2: Add to variant list**

```python
variants = ["looped", "looped+pathway", "looped+pathway+int4", "looped+moe"]
```

**Step 3: Validation** — After training, check:
- MoE variant doesn't regress vs plain looped
- MoE routing diversity (all prior memories used, not just the most recent)
- Wall time overhead from routing

---

### Task 7: Write architectural README section

**Objective:** Document the MoE-routed inter-layer DNN architecture.

**Files:**
- Modify: `C:\Loop-transformer\README.md`

**Content to add:**

```markdown
## MoE-Routed Inter-Layer DNN Pathway

**What it is:** Instead of updating a single memory vector sequentially,
the model maintains a **memory bank** M = [m_0, m_1, ..., m_{N-1}]
where each m_i is the memory state after loop iteration i.

At each loop step t, an **MoE router** selects top-k prior memories
and aggregates them into the new memory:

```
q_t = concat(m_{t-1}, pool(h_t))                  # query
c_t = sum(w_i * M[idx_i]) for top-k(w_i, idx_i)    # MoE routing
m_t = MLP(concat(m_{t-1}, c_t, pool(h_t)))         # update
```

This enables:
- **Sparse inter-layer communication** — each step reads only the most
  relevant prior memories via learned top-k routing
- **Flexible information flow** — early layers can directly influence
  late layers, not just via the chain m_0→m_1→...→m_N
- **Parameter efficiency** — routing adds negligible params vs full
  pairwise attention over all memories

**MoE Router:**
- Query: previous memory + pooled hidden (captures both state and input)
- Candidates: all prior memory states in bank
- Scoring: MLP → top-k softmax
- Sparsity: only top-k memories are read (k=2 default)

**Hypothesis:** MoE routing enables the DNN pathway to learn
task-dependent communication patterns — routing to early layers
for factual recall vs. routing to recent layers for local context.
```

---

### Task 8: Verify everything passes

**Step 1:** Run smoke tests:
```bash
cd /c/Loop-transformer && source .venv/Scripts/activate && python looped_transformer.py
```
Expected: All variants pass including MoE memory.

**Step 2:** Run training experiment (quick, 200 steps):
```bash
cd /c/Loop-transformer && source .venv/Scripts/activate && python -c "
from train_experiment import *
config = ExperimentConfig(dim=64, num_heads=4, num_loops_or_layers=4, memory_dim=16,
                          max_seq_len=64, batch_size=16, max_steps=200, warmup=50,
                          variant='looped+moe')
# ... run and print loss
"
```

**Step 3:** Check routing diversity — all prior memories should be selected across steps.

**Step 4:** Commit:
```bash
cd /c/Loop-transformer && git add -A && git commit -m "feat: add MoE-routed inter-layer DNN pathway"
```

---

## Files Changed Summary

| File | Action | Change |
|---|---|---|
| `moe_memory_pathway.py` | **Create** | `MemoryBank`, `MoERouter`, `MoEMemoryDNN` with routing trace |
| `looped_transformer.py` | **Modify** | Add `use_moe_memory` flag, integrate MoE memory pathway |
| `train_experiment.py` | **Modify** | Add `looped+moe` variant, update comparison |
| `README.md` | **Modify** | Document MoE-routed inter-layer DNN architecture |

## Open Questions / Risks

1. **Routing collapse** — MoE may converge to always selecting the same memories (e.g., always the most recent). Mitigation: add auxiliary load-balancing loss (importance-weighted entropy, or variance penalty on routing weights).

2. **Cold start** — At step t=0, the bank is empty (no candidates). Memory updates use zero context. This bootstraps OK but the first 1-2 steps have no inter-layer communication. Acceptable tradeoff.

3. **Compute overhead** — MoE routing adds O(N) scoring per step (where N is bank size). For N <= 16 loops this is negligible vs. transformer self-attention. For larger N, consider hierarchical routing.

4. **Gradient flow** — The routing is discrete (top-k indices), so gradients flow through the selected memories' aggregated values but not through the routing decisions. For true end-to-end training, consider soft top-k (e.g., Gumbel-SoftMax relaxation). For now, the weighted softmax + straight-through routing is sufficient.

5. **Memory bank on GPU** — `MemoryBank` maintains a persistent buffer. Ensure `.to(device)` is called correctly or bank is re-created on each forward to avoid stale device issues.
