# IDEA.md — Looped Transformer with Capability-Split MoE Memory

## Core Idea

Replace the single-vector sequential memory in looped transformers with a **DNN memory bank** split into **capability channels**, each with its own **MoE routing** and **precision gate**.

```
Standard forward:   m_t = MLP(m_{t-1}, h_t)
Our forward:        m_t = [memorize | math | speak | reason]_t
                          ↑ each channel = MoE_router(top-k prior) → ExpertMLP × precision_gate
Post-loop:          M = DNN_refiner(stack(m_0..m_N))  ← corrects drift at FP32
```

## Key Insights

### 1. Looped Transformers trade params for compute
Sharing one block across N loops gives parameter efficiency at the cost of recurrent training instability. Ouro (ByteDance) and LoopFormer (2025-2026) show this works for reasoning tasks.

### 2. Quantization error accumulates across loops
Experiment: INT4 on looped trunk → 3.10 loss vs 0.98 FP32 (3× degradation). Each loop re-quantizes, error compounds.

**Solution:** Keep memory/refinement path at FP32 (precision anchor), quantize only non-recurrent weights. DNN refinement step corrects quantization drift.

### 3. Capability separation prevents routing competition
Rather than one MoE routing all tokens through the same expert pool, split the memory space into C channels. Each channel routes independently over its own prior states.

| Capability | Precision need | Typical task |
|-----------|----------------|-------------|
| memorize | Low (INT4) | Factual recall, lookup |
| math | High (FP32) | Multi-step computation |
| speak | Medium (INT8) | Language generation |
| reason | Medium-High (INT8/FP32) | Multi-hop reasoning |

### 4. Two-level routing per capability (merged architecture)

```
Level 1 — Memory MoE (top-k over prior same-cap memories)
  query = m_{t-1}^c + pool(h_t)
  candidates = [m_0^c, m_1^c, ..., m_{t-1}^c]
  context = sum(w_i × m_i^c)  for top-k(w_i)

Level 2 — Expert MoE (top-k over expert MLPs within capability)
  routed = concat(m_t^c, context)
  expert_output = sum(w_j × expert_j(routed))  for top-k(w_j)

Output = m_t^c + sigmoid(precision_gate_c) × expert_output
```

### 5. DNN refinement as error correction
Post-loop: feed complete bank M = [m_0, m_1, ..., m_N] through a small TransformerEncoder (2 layers). Self-attention lets each step's memory read all others, correcting quantization drift and overthinking.

### 6. Learnable precision gates
Each capability has `sigmoid(logit_c)` gate that controls how much expert output flows through. With an L0 sparsity penalty (or actual quantization during training), gates converge to different values per capability.

## Architecture Zoo

```
C:\Loop-transformer\
├── looped_transformer.py           # Shared block + fake quant + integration hooks
├── moe_memory_pathway.py           # My impl: CapabilityMoEMemory, MemoryBank, MoERouter
├── .codex/codex_solution.py        # Codex impl: CapabilityMoE, CodexSolution, BankRefiner
├── merged_architecture.py          # Merged: 2-level routing (memory MoE → expert MoE)
├── multi_expert_training.py        # Production: MultiExpertMemoryDNN, mixed domain data, training
├── compare_archs.py                # Training harness comparing all variants
├── interactive.py / chat.py        # Chat interfaces
```

## Experimental Results (Shakespeare char-LM, 200-3000 steps)

| Variant | Params | Val Loss (200s) | Val Loss (3000s) | Key feature |
|---|---|---|---|---|
| Plain looped | 54K | 2.8738 | — | Shared block, no memory |
| Mine (memory only) | 159K | 2.8408 | — | Per-cap memory MoE only |
| Codex (experts only) | 148K | 2.9765 | — | Per-cap expert MoE only |
| Merged (2-level) | 157K | 2.9531 | — | Memory MoE + expert MoE |
| MultiExpert DNN | 132K | — | 0.0415 | 8 loops, 4 caps, 4 exp/cap |

At small training budget, simpler converges faster. At 3000 steps, MultiExpert model reaches near-zero loss but on tiny dataset (1.4K examples, 92K chars).

**Precision gates remained at ~0.5** because no actual quantization was active during training. Need quantization + more data for gate divergence.

## Next Steps

1. **Big data**: Full Shakespeare (1M chars), token-level (BPE), all 38 plays
2. **Real quantization**: Fake quant on trunk during training, measure gate divergence
3. **Precision loss**: L0 sparsity penalty on gates to force low-precision channels
4. **Domain probing**: Test math/code/reason subsets to verify capability routing
5. **Ouro-scale**: Port to HuggingFace Trainer, train 1B+ param version

## References

| Paper | Year | Key idea |
|---|---|---|
| Ouro (arxiv 2510.25741) | 2025 | Looped LM with sandwich norm |
| LoopFormer (arxiv 2602.11451) | 2026 | Elastic-depth looped, shortcut modulation |
| MELT (arxiv 2605.07721) | 2026 | Shared KV cache, constant-memory loops |
| On Expressive Power of Looped Transformers (ICML 2025) | 2025 | Universal approximation with loops |
| Loop, Think, & Generalize (arxiv 2604.07822) | 2026 | Overthinking problem in recurrent-depth |
| LLM-QAT (arxiv 2305.17888) | 2023 | QAT for LLMs |
| SpinQuant (arxiv 2405.16406) | 2024 | Quantization-aware training |
