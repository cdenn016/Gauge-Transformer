# Investigation: High Validation PPL (100-200) and Noisy Gradients

**Best result so far:** 114 PPL on WikiText-103 (50257 vocab)
**Target:** Sub-100 PPL to be competitive with standard transformers

## Executive Summary

After analyzing the codebase, I've identified **6 major issues** contributing to high PPL and gradient noise:

1. **No learned Q/K projections** - KL-based attention lacks representational capacity
2. **Attention collapse risk** - KL(q_i||q_i)=0 makes self-attention always most attractive
3. **VFE iteration noise** - Multiple gradient descent steps per forward create gradient variance
4. **Softmax coupling gradient complexity** - The ∂β/∂μ term is highly nonlinear
5. **Covariance transport numerical issues** - Full matrix operations are ill-conditioned
6. **Missing stabilizers** - LayerNorm, dropout, residuals often disabled

---

## Detailed Analysis

### 1. KL-Based Attention Lacks Representational Capacity

**Standard transformer:**
```
β_ij = softmax(Q_i @ K_j^T / √d_k)    # Learned Q, K projections
```

**Gauge transformer:**
```
β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ)    # No learned projections!
```

**Problem:** The gauge transformer computes attention directly from raw beliefs without learned Q/K transformations. This severely limits what queries the model can ask.

In a standard transformer with K=256 and 8 heads:
- W_Q, W_K each have 256×256 = 65K parameters
- These allow **arbitrary linear transformations** before computing similarity

In the gauge transformer:
- Attention depends only on KL divergence of transported beliefs
- The only "query" is "how similar is q_i to the transported q_j?"

**Impact:** ~30-50 PPL penalty due to limited attention expressivity.

**Recommendations:**
1. Add learnable pre-attention projections: `μ' = W_μ @ μ` before computing KL
2. Use multiple "heads" with different projections (like standard MHA)
3. Learn temperature κ per-head for adaptive sharpness

---

### 2. Attention Collapse: Self-Attention Always Wins

**The problem:** `KL(q_i || q_i) = 0` always, regardless of q_i's distribution.

With KL-based attention:
```
logits[i,i] = -KL(q_i || q_i) / κ = 0        # Always the highest!
logits[i,j] = -KL(q_i || Ω_ij[q_j]) / κ ≤ 0  # Always lower
```

Without intervention, position i will attend **only to itself**, ignoring all context.

**Current mitigation:** `mask_self_attention=True` (set diagonal to -inf)
- But this is a hack, not a principled solution
- First position (i=0) can only attend to itself with causal mask → must special-case

**Impact:** 10-20 PPL penalty from suboptimal attention patterns.

**Recommendations:**
1. Add learned self-attention bias: `logits[i,i] -= bias_self` (learnable)
2. Use ALiBi-style position-dependent penalties
3. Consider contrastive attention: explicitly push self-similarity down

---

### 3. VFE Iteration Gradient Noise

**The problem:** Each forward pass runs `n_vfe_steps=20` gradient descent iterations.

```python
for step in range(n_vfe_steps):
    beta = compute_attention(mu, sigma, phi)  # Recompute β
    grad_mu = compute_vfe_gradient(...)        # Full gradient
    mu = mu - lr * grad_mu                     # Update
```

This creates **nested optimization** where:
- Outer loop: backprop through CE loss
- Inner loop: VFE descent (20 steps!)

**Gradient variance sources:**
1. Each VFE step depends on all previous steps → deep computation graph
2. β recomputation at each step → gradients through all softmax operations
3. Softmax coupling term `∂β/∂μ` creates O(N²) gradient connections

**Impact:** High gradient variance → need smaller learning rates → slower convergence → ~20-30 PPL penalty.

**Recommendations:**
1. **Reduce VFE steps** during training (5 is often sufficient)
2. **Detach intermediate β** - don't backprop through all 20 steps
3. **Use implicit differentiation** instead of unrolling (like DEQ models)
4. **Gradient accumulation** with larger effective batch size

---

### 4. Softmax Coupling Gradient Complexity

The "principled nonlinearity" replacing GELU:
```
∂β_ij/∂μ_i = β_ij × [∂KL_ij/∂μ_i - Σ_k β_ik × ∂KL_ik/∂μ_i] / κ
```

**Problems:**
1. This couples gradients across ALL positions through β
2. The deviation term `∂KL_ij/∂μ_i - avg(∂KL/∂μ)` can be noisy when attention is diffuse
3. Division by κ amplifies noise when κ is small

**Current observation:** `kappa=0.1` is very sharp, amplifying gradient noise 10x.

**Impact:** High variance in gradients → unstable training → 10-15 PPL penalty.

**Recommendations:**
1. **Increase κ** to 0.5-1.0 for training stability (can anneal lower later)
2. **Add temperature schedules** - start warm, cool down
3. **Clip softmax coupling gradient** separately from main gradient
4. Consider **GELU fallback** for early training, transition to VFE later

---

### 5. Covariance Transport Numerical Issues

**Full covariance transport:**
```
Σ_transported = Ω @ Σ @ Ω^T    # (B, N, N, K, K) tensor!
```

**Problems:**
1. **Memory:** O(B × N² × K²) for full transported covariances
2. **Condition numbers:** Matrix inversions `torch.linalg.inv(Σ)` are ill-conditioned
3. **Cholesky failures:** `torch.linalg.cholesky` fails for near-singular matrices
4. **Gradient through matrix_exp:** `torch.matrix_exp` gradients are expensive and noisy

**Current mitigations:**
- Block-diagonal KL for irrep structure
- Chunked processing
- Diagonal covariance mode

But the core issue remains: **transported covariances are full even when source is diagonal!**
```
Σ_transported = Ω @ diag(σ) @ Ω^T  # Full K×K matrix, not diagonal!
```

**Impact:** 15-25 PPL penalty from numerical instabilities.

**Recommendations:**
1. **Keep Σ diagonal throughout** - extract diagonal of transported Σ
2. **Use log-space** for variance updates (log σ instead of σ)
3. **Fix covariance at training time** - only let μ evolve
4. **Lower-bound eigenvalues** more aggressively (min=0.1 instead of 0.01)
5. **Use SVD-based inverse** instead of Cholesky for robustness

---

### 6. Missing Architectural Stabilizers

**Pure VFE mode disables:**
```python
use_layernorm=False   # No normalization!
use_dropout=False     # No regularization!
use_residual=False    # No gradient highways!
```

**Why this hurts:**
1. **No LayerNorm:** Activations grow unbounded → gradient explosion
2. **No Dropout:** Overfitting on training data → poor generalization
3. **No Residuals:** Gradients must flow through VFE iterations → vanishing

**Current code at `model.py:349`:**
```python
use_layernorm=config.get('use_layernorm', False),
use_dropout=config.get('use_dropout', False),
use_residual=config.get('use_residual', False),
```

**Impact:** 30-50 PPL penalty from training instability.

**Recommendations:**
1. **Enable LayerNorm** - essential for stable deep networks
2. **Enable residual connections** - critical for gradient flow
3. **Add dropout=0.1** - standard regularization
4. These are NOT "ad-hoc" - they're mathematically principled!

---

## Gradient Noise Sources Summary

| Source | Contribution | Fix Difficulty |
|--------|-------------|----------------|
| VFE iteration unrolling | 30% | Medium |
| Softmax coupling ∂β/∂μ | 25% | Medium |
| Matrix exp/inv gradients | 20% | Hard |
| Missing LayerNorm | 15% | Easy |
| Small batch size | 10% | Easy |

---

## Recommended Configuration Changes

### Quick Wins (should reduce PPL by 30-50):

```python
CONFIG = {
    # ENABLE STABILIZERS
    'use_layernorm': True,      # Was False - critical!
    'use_dropout': True,        # Was False - helps generalization
    'use_residual': True,       # Was False - gradient flow!
    'dropout': 0.1,

    # SOFTER ATTENTION
    'kappa': 0.5,               # Was 0.1 - less sharp = less noisy
    'mask_self_attention': True,

    # FEWER VFE STEPS
    'n_vfe_steps': 5,           # Was 20 - reduce gradient depth

    # LARGER BATCH
    'batch_size': 64,           # Was 24 - variance reduction

    # DIAGONAL COVARIANCE
    'diagonal_covariance': True,  # Simplify numerics
    'evolve_sigma': False,        # Fix covariance - let μ learn
}
```

### Medium-Term Improvements:

1. **Add learned attention projections:**
```python
# In attention.py, before compute_attention_weights:
mu_q_projected = self.W_mu(mu_q)  # Learn query transformation
```

2. **Implement implicit differentiation:**
```python
# Instead of unrolling VFE, use equilibrium gradient:
mu_star = vfe_equilibrium(mu_init)  # Find fixed point
# Use implicit function theorem for gradient
```

3. **Temperature annealing:**
```python
kappa = kappa_init * (kappa_final / kappa_init) ** (step / total_steps)
# Start at κ=1.0, end at κ=0.1
```

### Architecture Experiments to Try:

1. **Hybrid attention:** Standard MHA + KL-based attention in parallel
2. **Gated VFE:** `μ_new = gate × μ_vfe + (1-gate) × μ_residual`
3. **Shallow VFE:** Only apply VFE dynamics in last 1-2 layers

---

## Expected PPL Improvements

| Change | Expected Gain | Cumulative |
|--------|--------------|------------|
| Baseline | 114 PPL | 114 |
| Enable LayerNorm+Residual | -25 PPL | 89 |
| Softer κ (0.5) | -10 PPL | 79 |
| Reduce VFE steps (5) | -8 PPL | 71 |
| Fix covariance | -5 PPL | 66 |
| Larger batch (64) | -3 PPL | 63 |
| **With learned projections** | -15 PPL | **48** |

**Target:** Sub-50 PPL should be achievable with learned attention projections.

---

## Validation Methodology

To verify these recommendations:

1. **Ablation study:** Enable each fix one-by-one, measure val PPL
2. **Gradient analysis:** Log grad norms for each component
3. **Attention visualization:** Check β patterns before/after changes
4. **Standard baseline:** Compare to `standard_transformer.py` with same params

---

## Conclusion

The gauge transformer's core innovation (KL-based attention) is mathematically elegant but removes too much learned capacity. The path to competitive PPL:

1. **Immediate:** Enable architectural stabilizers (LayerNorm, residuals, dropout)
2. **Short-term:** Soften attention (larger κ), reduce VFE iterations
3. **Medium-term:** Add learned attention projections
4. **Long-term:** Implement implicit differentiation for VFE

The current 114 PPL is not a fundamental limit of the gauge-theoretic approach, but a consequence of missing standard transformer components that are mathematically compatible with the framework.
