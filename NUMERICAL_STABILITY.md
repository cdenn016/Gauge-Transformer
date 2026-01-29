# Numerical Stability for Large Gauge Groups

## The Problem

Training with SO(100) / K=100 exhibits non-monotonic loss (PPL oscillates instead
of decreasing). This document explains the root causes and the fixes applied,
distinguishing between **bug fixes** (mathematical correctness), **justified
engineering** (sound approximations), and **pragmatic choices** (hyperparameter
convenience at the cost of strict theoretical form).

---

## 1. Transport Operator Re-Orthogonalization

**Category: Bug fix**

### Theory

The gauge-theoretic framework requires transport operators to live in SO(K):

```
Ω_ij = exp(φ_i · G) · exp(-φ_j · G) ∈ SO(K)
```

This means Ω^T Ω = I and det(Ω) = +1. Transported covariances must be
isospectral (same eigenvalues as the original), and the transitivity property
Ω_ij Ω_jk = Ω_ik must hold.

### What went wrong

The numpy path (`transport.py`) computes in float64 and projects to SO(K) via
SVD (`_project_to_orthogonal`). The PyTorch path (`attention.py`) used float32
`torch.matrix_exp` with **no re-orthogonalization**. For K=100, the Padé
scaling-squaring algorithm accumulates enough rounding error that Ω drifts from
SO(K). Consequences:

- `Ω Σ Ω^T` is no longer isospectral → spurious KL contributions
- Gauge symmetry is broken by numerical artifacts, not by physics
- The diagonal approximation `diag(Ω diag(σ) Ω^T)_k = Σ_l Ω_{kl}² σ_l` relies
  on `Σ_l Ω_{kl}² = 1` (row normalization of orthogonal matrix); drift breaks this

### Fix

1. Compute `matrix_exp` in float64 for K ≥ 16, cast back to working precision
2. Apply one Newton-Schulz iteration: `Q ← Q(3I - Q^TQ)/2`

Newton-Schulz converges quadratically for near-orthogonal matrices and is cheaper
than SVD. It preserves the autograd graph (unlike `.data` assignment or SVD).

### Justification

This is unambiguously required. The framework defines Ω ∈ SO(K); failing to
enforce this is a numerical bug, not a modeling choice.

---

## 2. Attention Temperature Scaling: `logits = -KL / (κ · √K)`

**Category: Justified engineering**

### Theory

Attention weights are a Boltzmann distribution over information distances:

```
β_ij = softmax_j( -KL(q_i || Ω_ij[q_j]) / κ )
```

The temperature κ is a free parameter in the theory—the active inference
framework does not specify its value.

### What went wrong

KL divergence between K-dimensional Gaussians sums K terms:

```
KL = 0.5 · [ Σ_k (σ_q[k]/σ_p[k])  +  Σ_k (Δμ[k])²/σ_p[k]  -  K  +  Σ_k log(σ_p[k]/σ_q[k]) ]
```

At initialization with σ_q ≈ σ_p, the trace term ≈ K and the −K cancels exactly,
giving KL ≈ 0. During training, per-dimension deviations accumulate. The
statistical question is how KL scales with K:

| Regime | KL scaling | Explanation |
|--------|-----------|-------------|
| Independent per-dim deviations | O(√K) variance | Central Limit Theorem |
| Correlated deviations (e.g., transport mixing) | O(K) worst case | All terms shift together |

With K=100 and κ=1.0, logits reach −50 to −100, making softmax essentially
one-hot. This produces:

- Zero gradients for all but one token pair (softmax saturation)
- Abrupt attention switching when the dominant target changes
- Non-monotonic loss as the model oscillates between attention patterns

### Fix

```python
dim_scale = math.sqrt(max(K, 1))
logits = -kl_matrix / (kappa * dim_scale)
```

This is equivalent to `κ_eff = κ · √K`.

### Justification

This is directly analogous to `1/√d_k` scaling in standard transformer attention
(Vaswani et al., 2017). There, the dot product Q·K sums d_k terms and is
normalized by √d_k to keep softmax non-degenerate. The KL divergence plays the
role of the dot product; K plays the role of d_k.

From a statistical mechanics perspective: the temperature of a Boltzmann
distribution should be set relative to the energy scale of the system. If typical
KL ∼ O(√K) (independent per-dimension deviations), then κ ∼ √K keeps the
distribution non-degenerate.

**The theory permits any κ.** The √K scaling chooses κ such that the same
numerical value of the hyperparameter produces similar attention entropy across
different K. This is a normalization convention, not a theoretical claim.

**Caveat:** Previously tuned κ values implicitly absorbed the K-dependence. For
K=3, the scaling factor is √3 ≈ 1.7 (mild). For K=100, it is 10 (significant).
If restoring a checkpoint trained without this scaling, adjust κ accordingly.

---

## 3. KL Clamp Ceiling: `max(100, 5K)`

**Category: Bug fix**

### Theory

The KL clamp is a numerical guard, not a theoretical quantity. Its purpose is to
prevent `inf` or overflow in downstream computation. The theory says nothing about
clamping—it's pure implementation.

### What went wrong

The ceiling was hardcoded at 100. With K=100, a per-dimension KL of just 1.0 nat
gives total KL = 100, hitting the ceiling for moderate divergence. When the clamp
activates:

- Gradient is exactly zero (flat region in the loss landscape)
- The model cannot improve beliefs that are moderately far from the prior
- These "dead zones" contribute to non-monotonic training

### Fix

```python
kl_ceil = max(100.0, 5.0 * K)
```

### Justification

Straightforward: a dimension-dependent quantity needs a dimension-dependent guard.
The factor 5 is generous (allows per-dimension KL up to 5 nats before clamping).

---

## 4. Per-Parameter-Group Gradient Clipping

**Category: Standard optimization practice**

### Theory

The gauge-theoretic framework is agnostic to optimization algorithm. Gradient
clipping is an implementation concern.

### What went wrong

With SO(100), parameter dimensions are:

| Parameter | Dimension per token | Fraction of total |
|-----------|--------------------:|------------------:|
| phi_embed | 4950 | ~96% |
| mu_embed  | 100  | ~2%  |
| sigma     | 100  | ~2%  |

Global `clip_grad_norm_` at 1.0 means phi consumes ~96% of the gradient budget.
The mu and sigma gradients are scaled down by ~50x, effectively frozen.

### Fix

When `use_param_groups=True`, clip each parameter group independently:

```python
for group in optimizer.param_groups:
    torch.nn.utils.clip_grad_norm_(group['params'], grad_clip)
```

### Justification

Standard practice in multi-scale optimization (used in DALL-E, Stable Diffusion,
etc.). Each parameter type gets its own gradient budget regardless of
dimensionality.

---

## 5. Free Energy Loss Normalization by √K

**Category: Pragmatic choice (changes the functional)**

### Theory

The variational free energy is:

```
F = α · Σ_i KL(q_i || p_i)                         [self-consistency]
  + λ_β · Σ_{i,j} β_ij · KL(q_i || Ω_ij q_j)      [belief alignment]
  + CE(logits, targets)                              [observation likelihood]
```

The theory says **minimize F**. It does not prescribe normalization of the KL
terms.

### What went wrong

The CE loss is O(1) in K—it operates over the vocabulary, not the latent
dimension. The KL terms scale as O(√K) to O(K) during training. With K=100 and
the default hyperparameters (α=0.1, λ_β=1.0), the KL terms dominate CE by 10–30x.
The model minimizes belief divergence at the expense of token prediction accuracy.

### Fix

```python
dim_scale = math.sqrt(max(K, 1))
belief_align_loss = lambda_beta * weighted_kl.sum(...).mean() / dim_scale
self_consistency_loss = alpha * kl_per_agent.mean() / dim_scale
```

### Justification

**Arguments for:**

1. **β-VAE analogy.** In β-VAE (Higgins et al., 2017), the KL term is explicitly
   scaled relative to reconstruction loss to balance complexity vs. accuracy. Our
   situation is structurally identical: KL (complexity) vs. CE (accuracy).

2. **Per-dimension information rate.** Dividing total KL by √K can be interpreted
   as measuring per-dimension divergence intensity rather than total divergence.
   Both are valid information-geometric quantities.

3. **Hyperparameter transferability.** The same α and λ_β values produce similar
   loss landscapes across different K, making experimental results comparable.

**Arguments against:**

1. **Changes the mathematical functional.** The theory defines F as the sum of
   specific information-theoretic quantities. Normalizing changes what is being
   minimized. Strictly, the fix is equivalent to redefining:
   ```
   α_eff = α / √K
   λ_β_eff = λ_β / √K
   ```

2. **User should tune hyperparameters.** A purist position: when changing K, the
   user should rescale α and λ_β by ~1/√K to maintain the same balance with CE.
   The normalization does this automatically but silently.

3. **Not derivable from the variational principle.** The free energy functional
   emerges from variational inference. The normalization is not part of that
   derivation.

### Recommendation

This is the least theoretically clean fix. If strict adherence to the theoretical
functional is preferred, **revert this change** and instead:

- Set `α = 0.1 / sqrt(K)` and `λ_β = 1.0 / sqrt(K)` in the training config
- Document that KL-based hyperparameters must be scaled with K

The current implementation chooses convenience (same hyperparameters work across
K) over theoretical purity (exact variational free energy). Both approaches
produce the same optimization landscape.

---

## Summary Table

| Fix | Category | Theoretical status | Reversible? |
|-----|----------|-------------------|-------------|
| Re-orthogonalization | Bug fix | **Required** by Ω ∈ SO(K) | No—always needed |
| Attention √K scaling | Engineering | Justified (≡ standard 1/√d_k scaling) | Adjust κ if reverting |
| KL clamp ceiling | Bug fix | Dimension-dependent guard | No—always needed |
| Per-group grad clip | Optimization | Standard practice, theory-agnostic | Adjust grad_clip if reverting |
| Loss √K normalization | Pragmatic | Changes effective α, λ_β | Scale hyperparameters by 1/√K instead |

---

## When These Fixes Matter

The fixes are most impactful when **all three conditions hold**:

1. **Large gauge group**: SO(N) with N ≥ 16 (phi_dim ≥ 120)
2. **Large latent dimension**: K ≥ 16
3. **Non-trivial transport**: `use_identity_transport=False` (gauge transport active)

For SO(3) with K ≤ 45, the effects are mild:
- √K scaling at K=3 is a factor of 1.7 (absorbed by κ tuning)
- Float32 matrix_exp on 3×3 matrices is exact to machine precision
- KL ceiling of 100 is rarely hit
- Phi has only 3 dims, no gradient budget imbalance
