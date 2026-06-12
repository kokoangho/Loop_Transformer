# Cross-Layer Deep Pathway Prototype

`cross_layer_pathway.py` implements a GPT-2-style transformer with an added
cross-layer memory pathway:

```text
x_{l+1} = Block_l(x_l, condition=m_l)
m_{l+1} = DNN_l(m_l, summary(x_l))
x_{l+1} += gate_l(x_l, m_l) * project(m_{l+1})
```

Unlike a standard transformer, each layer receives more than the token residual
stream.  A separate global memory state is updated across depth by a small MLP
and then injected back into the residual stream through a learned gate.

Unlike a looped transformer, the pathway is not only repeated application of
the same block state.  It carries an explicit cross-layer state that can evolve
monotonically through depth, even if the visible token pathway is looped,
shared, or quantized.

Hypothesis:

> A cross-layer deep pathway can stabilize quantized looped transformers by
> carrying a higher-precision global state across depth.

Suggested experiment setup:

1. Train a baseline small GPT-2-style decoder.
2. Train a looped/shared-layer version with the same parameter budget.
3. Train the same looped model with the cross-layer pathway enabled.
4. Quantize the transformer block weights more aggressively than the memory
   pathway weights.
5. Compare loss curves, activation norms, gate values, memory trajectories, and
   downstream perplexity under equal compute.
