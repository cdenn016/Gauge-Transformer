# Pure FEP Transformer: Ground-Up Implementation Plan

## Executive Summary

This document provides a detailed plan to implement a **Pure Free Energy Principle (FEP) Transformer** from first principles, avoiding ad hoc neural network components and deriving all mechanisms from the variational free energy functional.

**Key Design Decisions:**
1. **Flat Gauge Limit**: We operate in the trivial gauge limit (Ω_ij = I) unless empirically proven otherwise - this eliminates gauge frame complexity while preserving the core FEP structure
2. **Position-Dependent Priors**: Position information emerges through learned position-specific priors, NOT through gauge frames or sinusoidal encodings
3. **Token Priors**: A unified prior bank encodes/decodes via KL divergence to token distributions
4. **Two Timescales**: Q-flow (fast belief updates) and P-flow (slow prior updates)
5. **No Neural Networks**: Zero MLPs, zero learned projection matrices, zero activation functions

---

## I. Theoretical Foundation

### 1.1 The Variational Free Energy Functional

The complete VFE functional from the papers is:

```
F[{q_i}, {p_i}] =
    Σ_i α·KL(q_i || p_i)                           [Self-coupling]
  + Σ_i,j β_ij·KL(q_i || q_j)                      [Belief alignment - FLAT GAUGE]
  - Σ_i E_{q_i}[log p(y_i | z_i)]                  [Observation likelihood]
```

**Critical simplification for transformers**: In the flat gauge limit where Ω_ij = I (identity), the transport operator disappears and belief alignment becomes direct comparison KL(q_i || q_j).

### 1.2 Why Flat Gauge?

From the LLM manuscript (Section 2.5, "The Zero-Dimensional Limit"):
> "Standard transformer attention emerges as the degenerate limit when gauge frames trivialize... Standard attention (softmax(QK^T/√d)) emerges when... gauge frames become trivial (Ω_ij → I)"

**Benefits of flat gauge:**
- Eliminates O(K³) matrix exponential computations
- Removes O(N²×K²) covariance transport
- Simplifies implementation without losing core VFE structure
- Standard transformers ARE the flat gauge limit - we start there and can add gauge structure later if needed

**When to add gauge frames back:**
- Only if flat gauge performance is insufficient
- Only if we need equivariance properties
- Only for specific domains requiring frame transformations

### 1.3 Attention as Belief Alignment

Attention weights emerge from KL divergence minimization:

```
β_ij = softmax_j(-KL(q_i || q_j) / κ)
```

where κ is the temperature parameter controlling attention sharpness.

For Gaussian beliefs q_i = N(μ_i, σ_i²) with diagonal covariance:

```
KL(q_i || q_j) = ½ Σ_k [log(σ_jk²/σ_ik²) + σ_ik²/σ_jk² + (μ_ik - μ_jk)²/σ_jk² - 1]
```

### 1.4 The Nonlinearity Emerges

The gradient of the VFE with respect to beliefs produces nonlinear dynamics through the softmax:

```
∂(β_ij·KL_ij)/∂μ_i = (∂β_ij/∂μ_i)·KL_ij + β_ij·(∂KL_ij/∂μ_i)

where:
∂β_ij/∂μ_i = β_ij·[∂KL_ij/∂μ_i - Σ_k β_ik·∂KL_ik/∂μ_i] / κ
```

This product rule creates **positive feedback**: similar beliefs → higher β → pulled closer → clusters form. This replaces GELU/ReLU with principled nonlinearity.

---

## II. Core Architecture

### 2.1 Agent Representation

Each token position i has:
- **Belief**: q_i = N(μ_qi, σ_qi²) - current posterior estimate
- **Prior**: p_i = N(μ_pi, σ_pi²) - learned expectation for this position

Both are **diagonal Gaussians** for efficiency (no full covariance matrices).

**Shapes:**
- μ_q: (batch, seq_len, embed_dim) - belief means
- σ_q: (batch, seq_len, embed_dim) - belief standard deviations (positive)
- μ_p: (seq_len, embed_dim) - position-dependent prior means (shared across batch)
- σ_p: (seq_len, embed_dim) - position-dependent prior stds

### 2.2 Token Prior Bank

A vocabulary-indexed prior bank serves as BOTH embedding and output:

```python
class TokenPriorBank:
    """
    Each vocabulary token v has a prior: π_v = N(μ_v, σ_v²)

    ENCODING: q_i ← π_{token[i]}  (initialize belief from token prior)
    DECODING: logits_v = -KL(q_i || π_v) / τ  (output via KL to all token priors)
    """
    μ_tokens: (vocab_size, embed_dim)  # learned prior means per token
    σ_tokens: (vocab_size, embed_dim)  # learned prior stds per token
```

**No separate embedding layer or output projection!**

### 2.3 Position-Dependent Priors

Position information is encoded in the **position-dependent priors**, NOT in gauge frames:

```python
class PositionPriors:
    """
    Each layer maintains position-specific priors.
    Position structure EMERGES from VFE minimization, not imposed.
    """
    μ_p: (max_seq_len, embed_dim)  # learnable position prior means
    σ_p: (max_seq_len, embed_dim)  # learnable position prior stds
```

**Why this works:**
- Different positions see different data patterns
- P-flow (prior updates) will differentiate positions based on prediction errors
- No need for sinusoidal encoding - position emerges from learning

### 2.4 Single Layer Structure

```
Input: μ_q, σ_q (beliefs from previous layer or token priors)
       μ_p, σ_p (position-dependent priors for this layer)

1. ATTENTION (from belief geometry):
   KL_ij = KL(q_i || q_j)  for all i,j pairs
   β_ij = softmax_j(-KL_ij / κ)

2. VFE GRADIENT DESCENT (Q-flow, multiple steps):
   for step in range(n_vfe_steps):
       F = α·Σ_i KL(q_i||p_i) + Σ_ij β_ij·KL(q_i||q_j)
       ∂F/∂μ_q = compute_gradient(...)
       μ_q ← μ_q - η_μ · σ_q² · ∂F/∂μ_q  (natural gradient)
       σ_q ← update_variance(...)

       # Optionally recompute β (dynamic attention)
       β_ij = softmax(-KL(q_i||q_j) / κ)

Output: μ_q, σ_q (updated beliefs)
```

### 2.5 Multi-Layer Hierarchy

```
Layer 0: Token priors → Beliefs
Layer 1: Layer 0 beliefs → Layer 1 beliefs
Layer 2: Layer 1 beliefs → Layer 2 beliefs
...
Final: Last layer beliefs → Output logits via TokenPriorBank
```

**Hierarchical prior flow (optional):**
- Parent layer beliefs become child layer priors
- Implements predictive coding / top-down constraint

---

## III. The VFE Components in Detail

### 3.1 Self-Coupling Term: KL(q_i || p_i)

Anchors beliefs to learned priors.

```python
def kl_self_coupling(μ_q, σ_q, μ_p, σ_p, eps=1e-6):
    """
    KL(q || p) for diagonal Gaussians.

    KL = ½ Σ_k [log(σ_pk²/σ_qk²) + σ_qk²/σ_pk² + (μ_qk - μ_pk)²/σ_pk² - 1]
    """
    var_q = σ_q.square() + eps
    var_p = σ_p.square() + eps

    log_ratio = torch.log(var_p / var_q)
    trace_term = var_q / var_p
    mahal_term = (μ_q - μ_p).square() / var_p

    kl = 0.5 * (log_ratio + trace_term + mahal_term - 1.0)
    return kl.sum(dim=-1)  # Sum over embed_dim, shape: (B, N)
```

### 3.2 Belief Alignment Term: Σ β_ij · KL(q_i || q_j)

The core attention mechanism.

```python
def kl_pairwise(μ_q, σ_q, eps=1e-6):
    """
    Compute KL(q_i || q_j) for all pairs.

    Returns: (B, N, N) matrix of KL divergences
    """
    B, N, K = μ_q.shape
    var_q = σ_q.square() + eps  # (B, N, K)

    # Expand for pairwise computation
    μ_i = μ_q.unsqueeze(2)  # (B, N, 1, K)
    μ_j = μ_q.unsqueeze(1)  # (B, 1, N, K)
    var_i = var_q.unsqueeze(2)  # (B, N, 1, K)
    var_j = var_q.unsqueeze(1)  # (B, 1, N, K)

    log_ratio = torch.log(var_j / var_i)  # (B, N, N, K)
    trace_term = var_i / var_j
    mahal_term = (μ_i - μ_j).square() / var_j

    kl = 0.5 * (log_ratio + trace_term + mahal_term - 1.0)
    return kl.sum(dim=-1)  # (B, N, N)

def compute_attention(kl_matrix, kappa=1.0, mask=None):
    """
    Attention from KL divergences.

    β_ij = softmax_j(-KL_ij / κ)
    """
    logits = -kl_matrix / kappa  # (B, N, N)

    if mask is not None:
        logits = logits.masked_fill(~mask, float('-inf'))

    return F.softmax(logits, dim=-1)  # (B, N, N)
```

### 3.3 Observation Likelihood Term

For language modeling, the observation is the target token:

```python
def observation_likelihood(μ_q, σ_q, token_priors, target_ids, tau=1.0):
    """
    p(y | q) ∝ exp(-KL(q || π_y) / τ)

    Returns: (B, N) negative log-likelihood (cross-entropy)
    """
    B, N, K = μ_q.shape
    V = token_priors.μ_tokens.shape[0]

    # KL to all token priors: (B, N, V)
    kl_to_tokens = compute_kl_to_token_priors(μ_q, σ_q, token_priors)

    # Logits: negative KL / temperature
    logits = -kl_to_tokens / tau  # (B, N, V)

    # Cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, V),
        target_ids.view(-1),
        reduction='none'
    ).view(B, N)

    return loss
```

### 3.4 Complete VFE Computation

```python
def compute_vfe(μ_q, σ_q, μ_p, σ_p, β, target_ids, token_priors, config):
    """
    Full variational free energy.

    F = α·Σ_i KL(q_i||p_i) + λ_β·Σ_ij β_ij·KL(q_i||q_j) + Σ_i CE_i
    """
    # Self-coupling
    kl_self = kl_self_coupling(μ_q, σ_q, μ_p, σ_p)  # (B, N)
    F_self = config.alpha * kl_self.sum()

    # Belief alignment
    kl_pairwise_matrix = kl_pairwise(μ_q, σ_q)  # (B, N, N)
    alignment = (β * kl_pairwise_matrix).sum(dim=-1)  # (B, N)
    F_align = config.lambda_beta * alignment.sum()

    # Observation likelihood
    ce_loss = observation_likelihood(μ_q, σ_q, token_priors, target_ids)
    F_obs = ce_loss.sum()

    return F_self + F_align + F_obs, {
        'F_self': F_self.item(),
        'F_align': F_align.item(),
        'F_obs': F_obs.item()
    }
```

---

## IV. Gradient Computation

### 4.1 Mean Gradient: ∂F/∂μ_qi

The gradient has four components:

```python
def compute_mu_gradient(μ_q, σ_q, μ_p, σ_p, β, kl_matrix, token_priors, targets, config):
    """
    ∂F/∂μ_qi = ∂F_self/∂μ_qi + ∂F_align/∂μ_qi + ∂F_obs/∂μ_qi
    """
    B, N, K = μ_q.shape
    var_q = σ_q.square()
    var_p = σ_p.square()

    # 1. Self-coupling gradient: ∂KL(q_i||p_i)/∂μ_qi
    grad_self = (μ_q - μ_p) / var_p  # (B, N, K)

    # 2. Alignment gradient (i aligning to others j)
    # ∂(β_ij·KL_ij)/∂μ_qi = (∂β_ij/∂μ_qi)·KL_ij + β_ij·∂KL_ij/∂μ_qi

    # ∂KL(q_i||q_j)/∂μ_qi = (μ_qi - μ_qj) / var_qj
    var_j = var_q.unsqueeze(1)  # (B, 1, N, K)
    μ_diff = μ_q.unsqueeze(2) - μ_q.unsqueeze(1)  # (B, N, N, K)
    grad_kl_direct = μ_diff / var_j  # (B, N, N, K)

    # β-weighted gradient
    grad_align_direct = (β.unsqueeze(-1) * grad_kl_direct).sum(dim=2)  # (B, N, K)

    # Product rule: gradient through β
    # ∂β_ij/∂μ_qi = -β_ij·[∂KL_ij/∂μ_qi - Σ_k β_ik·∂KL_ik/∂μ_qi] / κ
    mean_grad_kl = (β.unsqueeze(-1) * grad_kl_direct).sum(dim=2, keepdim=True)  # (B, N, 1, K)
    grad_beta = -β.unsqueeze(-1) * (grad_kl_direct - mean_grad_kl) / config.kappa  # (B, N, N, K)
    grad_align_beta = (grad_beta * kl_matrix.unsqueeze(-1)).sum(dim=2)  # (B, N, K)

    # 3. Others aligning to i: Σ_k β_ki · ∂KL(q_k||q_i)/∂μ_qi
    # ∂KL(q_k||q_i)/∂μ_qi = -var_k/var_i² · (μ_k - μ_i) + (μ_k - μ_i)/var_i
    # For diagonal: = -(μ_k - μ_i) / var_i
    grad_others_to_i = -μ_diff.transpose(1, 2) / var_q.unsqueeze(2)  # (B, N, N, K)
    grad_align_others = (β.transpose(1, 2).unsqueeze(-1) * grad_others_to_i).sum(dim=2)

    # 4. Observation gradient (handled separately via autograd or analytically)
    grad_obs = compute_observation_gradient(μ_q, σ_q, token_priors, targets)

    # Total gradient
    total_grad = (
        config.alpha * grad_self
        + config.lambda_beta * (grad_align_direct + grad_align_beta + grad_align_others)
        + grad_obs
    )

    return total_grad
```

### 4.2 Variance Gradient: ∂F/∂σ_qi

```python
def compute_sigma_gradient(μ_q, σ_q, μ_p, σ_p, β, config):
    """
    ∂F/∂σ_qi for diagonal covariance.

    Natural gradient uses: σ_new = σ · exp(-η · σ² · ∂F/∂var)
    """
    var_q = σ_q.square()
    var_p = σ_p.square()

    # Self-coupling: ∂KL(q||p)/∂var_q = ½(1/var_p - 1/var_q)
    grad_var_self = 0.5 * (1.0 / var_p - 1.0 / var_q)

    # Alignment terms (similar structure to mean gradient)
    # ∂KL(q_i||q_j)/∂var_qi = ½(1/var_qj - 1/var_qi)
    # ... (implement similarly)

    # Convert variance gradient to sigma gradient: ∂F/∂σ = 2σ · ∂F/∂var
    grad_sigma = 2 * σ_q * grad_var_self  # simplified

    return grad_sigma
```

### 4.3 Natural Gradient Descent

```python
def natural_gradient_step(μ_q, σ_q, grad_μ, grad_σ, lr_μ=0.1, lr_σ=0.01):
    """
    Natural gradient update respecting Fisher-Rao geometry.

    For Gaussians:
        μ_new = μ - η_μ · Σ · ∇_μ F  (natural gradient on mean)
        σ_new = σ · exp(-η_σ · σ² · ∇_var F)  (retraction on SPD manifold)
    """
    # Natural gradient for mean: precondition by covariance
    var_q = σ_q.square()
    μ_new = μ_q - lr_μ * var_q * grad_μ

    # Retraction-based update for variance (ensures positivity)
    σ_new = σ_q * torch.exp(-lr_σ * grad_σ)
    σ_new = σ_new.clamp(min=1e-4)  # Numerical stability

    return μ_new, σ_new
```

---

## V. Two-Timescale Learning

### 5.1 Q-Flow: Fast Belief Updates (Perception)

Within a single forward pass, beliefs evolve via VFE gradient descent:

```python
def q_flow(μ_q, σ_q, μ_p, σ_p, mask, token_priors, targets, config):
    """
    Fast timescale: Update beliefs to minimize VFE given fixed priors.

    This is the "perception" phase - inferring latent states from observations.
    """
    for step in range(config.n_vfe_steps):
        # Compute attention from current beliefs
        kl_matrix = kl_pairwise(μ_q, σ_q)
        β = compute_attention(kl_matrix, config.kappa, mask)

        # Compute gradients
        grad_μ = compute_mu_gradient(μ_q, σ_q, μ_p, σ_p, β, kl_matrix,
                                      token_priors, targets, config)
        grad_σ = compute_sigma_gradient(μ_q, σ_q, μ_p, σ_p, β, config)

        # Natural gradient step
        μ_q, σ_q = natural_gradient_step(μ_q, σ_q, grad_μ, grad_σ,
                                          config.lr_mu, config.lr_sigma)

    return μ_q, σ_q, β
```

### 5.2 P-Flow: Slow Prior Updates (Learning)

Between training steps, priors evolve based on prediction errors:

```python
def p_flow(μ_p, σ_p, μ_q, σ_q, errors, config):
    """
    Slow timescale: Update priors based on accumulated prediction errors.

    Key principle: Positions with high prediction error need more prior adjustment.
    """
    B, N, K = μ_q.shape

    # Average beliefs across batch
    μ_q_avg = μ_q.mean(dim=0)  # (N, K)
    σ_q_avg = σ_q.mean(dim=0)  # (N, K)

    # Error-weighted learning rate
    # Positions with higher error should update faster
    mean_error = errors.mean()
    relative_error = errors / (mean_error + 1e-6)  # (N,)
    lr_scale = torch.sqrt(relative_error.clamp(0.5, 2.0))  # Bounded scaling

    # Prior mean update: drift toward successful beliefs
    lr_per_pos = config.lr_prior * lr_scale.unsqueeze(-1)  # (N, K)
    μ_p_new = μ_p + lr_per_pos * (μ_q_avg - μ_p)

    # Prior variance update (optional - can keep fixed)
    # σ_p_new = ...

    return μ_p_new, σ_p
```

### 5.3 Token Prior Updates (Critical!)

The token prior bank must also update based on observation gradients:

```python
def update_token_priors(token_priors, μ_q, σ_q, target_ids, config):
    """
    Update token priors based on how well they predict targets.

    For each target token v, move π_v toward the beliefs that successfully
    predicted it.
    """
    B, N, K = μ_q.shape
    V = token_priors.μ_tokens.shape[0]

    # Compute gradient of observation loss w.r.t. token prior means
    # ∂CE/∂μ_πv = -p(v|q) · ∂KL(q||π_v)/∂μ_πv for target tokens

    with torch.no_grad():
        # For each target position, update the corresponding token prior
        for b in range(B):
            for n in range(N):
                v = target_ids[b, n].item()

                # Gradient: push token prior toward belief that predicted it
                delta_mu = μ_q[b, n] - token_priors.μ_tokens[v]
                token_priors.μ_tokens[v] += config.lr_token_prior * delta_mu

                # Similarly for variance (optional)
```

---

## VI. Complete Model Architecture

### 6.1 Configuration

```python
@dataclass
class PureFEPConfig:
    # Architecture
    vocab_size: int = 256
    embed_dim: int = 64
    n_layers: int = 4
    max_seq_len: int = 128

    # VFE weights
    alpha: float = 0.1          # Self-coupling weight
    lambda_beta: float = 1.0    # Alignment weight
    kappa: float = 1.0          # Attention temperature
    tau: float = 1.0            # Output temperature

    # Q-flow (fast timescale)
    n_vfe_steps: int = 10       # VFE iterations per forward pass
    lr_mu: float = 0.1          # Belief mean learning rate
    lr_sigma: float = 0.01      # Belief variance learning rate

    # P-flow (slow timescale)
    lr_prior: float = 0.01      # Position prior learning rate
    lr_token_prior: float = 0.01  # Token prior learning rate

    # Numerical stability
    variance_floor: float = 1e-4
    eps: float = 1e-6
```

### 6.2 Model Definition

```python
class PureFEPTransformer(nn.Module):
    """
    Pure FEP Transformer - NO neural network components.

    Architecture:
    - TokenPriorBank: Encodes/decodes via KL to token priors
    - PositionPriors: Per-layer, per-position learned priors
    - VFE dynamics: Beliefs evolve via natural gradient descent
    """

    def __init__(self, config: PureFEPConfig):
        super().__init__()
        self.config = config

        # Token prior bank (embedding + output)
        self.token_priors = TokenPriorBank(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim
        )

        # Position-dependent priors for each layer
        self.position_priors = nn.ParameterList([
            nn.ParameterDict({
                'mu': nn.Parameter(torch.randn(config.max_seq_len, config.embed_dim) * 0.1),
                'log_sigma': nn.Parameter(torch.zeros(config.max_seq_len, config.embed_dim))
            })
            for _ in range(config.n_layers)
        ])

    def forward(self, input_ids, target_ids=None):
        B, N = input_ids.shape
        device = input_ids.device

        # === ENCODING: Initialize beliefs from token priors ===
        μ_q, σ_q = self.token_priors.encode(input_ids)  # (B, N, K)

        # Causal attention mask
        mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))

        # === PROCESS THROUGH LAYERS ===
        for layer_idx in range(self.config.n_layers):
            # Get position priors for this layer
            μ_p = self.position_priors[layer_idx]['mu'][:N]  # (N, K)
            σ_p = self.position_priors[layer_idx]['log_sigma'][:N].exp()  # (N, K)

            # Q-flow: VFE gradient descent on beliefs
            μ_q, σ_q, β = q_flow(
                μ_q, σ_q, μ_p, σ_p, mask,
                self.token_priors, target_ids, self.config
            )

        # === DECODING: Logits via KL to token priors ===
        logits = self.token_priors.decode(μ_q, σ_q)  # (B, N, V)

        # Compute loss if targets provided
        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                target_ids.view(-1)
            )

        return logits, loss

    def train_step(self, input_ids, target_ids):
        """
        Single training step with P-flow updates.
        """
        # Forward pass (Q-flow happens inside)
        logits, loss = self.forward(input_ids, target_ids)

        # P-flow: Update priors based on errors
        with torch.no_grad():
            per_position_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                target_ids.view(-1),
                reduction='none'
            ).view(input_ids.shape)

            # Update position priors
            for layer_idx in range(self.config.n_layers):
                p_flow(
                    self.position_priors[layer_idx]['mu'],
                    self.position_priors[layer_idx]['log_sigma'].exp(),
                    # ... pass relevant beliefs and errors
                )

            # Update token priors
            update_token_priors(
                self.token_priors,
                # ... pass final layer beliefs
                target_ids,
                self.config
            )

        return loss
```

---

## VII. What We Explicitly AVOID

### 7.1 No Neural Network Components

| Standard Transformer | Our Approach |
|---------------------|--------------|
| nn.Embedding | TokenPriorBank.encode() |
| nn.Linear (Q, K, V projections) | KL-based attention directly on beliefs |
| nn.Linear (output projection) | TokenPriorBank.decode() via KL |
| MLP / FFN | VFE gradient descent |
| GELU/ReLU activations | Softmax attention gradient (product rule) |
| LayerNorm | Not needed (natural gradient is scale-invariant) |

### 7.2 No Ad Hoc Positional Encoding

| Standard Approach | Our Approach |
|-------------------|--------------|
| Sinusoidal encoding | Position-dependent priors |
| Learned position embeddings | Position-dependent priors |
| RoPE / ALiBi | Flat gauge (relative position emerges from prior structure) |
| Gauge frame encoding | NOT USED (flat gauge limit) |

### 7.3 No Arbitrary Hyperparameter Ratios

Each hyperparameter has clear interpretation:
- **α**: How strongly beliefs anchor to priors
- **λ_β**: How strongly beliefs align with neighbors
- **κ**: Attention temperature (sharpness of selection)
- **τ**: Output temperature (confidence calibration)
- **lr_μ, lr_σ**: Natural gradient step sizes
- **lr_prior**: Speed of prior adaptation

---

## VIII. Implementation Phases

### Phase 1: Core VFE Engine (Week 1)
1. Implement diagonal Gaussian KL divergence
2. Implement pairwise KL computation
3. Implement attention from KL
4. Implement VFE and its gradients
5. Test gradient correctness with finite differences

### Phase 2: Single Layer (Week 2)
1. Implement TokenPriorBank
2. Implement Q-flow (VFE gradient descent)
3. Implement single-layer forward pass
4. Test on simple sequence memorization

### Phase 3: Full Model (Week 3)
1. Stack multiple layers
2. Implement P-flow (prior updates)
3. Implement token prior updates
4. Full training loop

### Phase 4: Validation (Week 4)
1. Character-level language modeling (WikiText-2)
2. Compare against standard transformer baseline
3. Analyze attention patterns
4. Profile computational cost

### Phase 5: Extensions (Future)
1. Add gauge structure if needed
2. Hierarchical prior propagation
3. Scale to larger models
4. Explore multi-head via irrep decomposition

---

## IX. Key Equations Summary

| Component | Equation |
|-----------|----------|
| **Belief** | q_i = N(μ_qi, σ_qi²) |
| **Prior** | p_i = N(μ_pi, σ_pi²) |
| **KL Divergence** | KL(q‖p) = ½[log(σ_p²/σ_q²) + σ_q²/σ_p² + (μ_q-μ_p)²/σ_p² - 1] |
| **Attention** | β_ij = softmax_j(-KL(q_i‖q_j) / κ) |
| **VFE** | F = α·Σ KL(q‖p) + λ·Σ β_ij·KL(q_i‖q_j) + CE |
| **Natural Gradient** | μ ← μ - η·σ²·∂F/∂μ |
| **Output** | logits_v = -KL(q‖π_v) / τ |

---

## X. Success Criteria

1. **Theoretical Purity**: Zero MLPs, zero learned projections, zero activations
2. **Functional Correctness**: Gradients verified against finite differences
3. **Competitive Performance**: Within 20% of standard transformer on WikiText-2
4. **Interpretability**: All parameters have geometric meaning
5. **Efficiency**: Reasonable training time (< 10x standard transformer)

---

## Appendix A: Comparison to Current Implementation

| Current (pure_fep_transformer.py) | New Design |
|-----------------------------------|------------|
| Gauge frames for position | Flat gauge (no gauge frames) |
| Complex transport operators | No transport (Ω = I) |
| Multiple position encoding modes | Single approach: position priors |
| Optional MLPs/hybrids | Pure VFE only |
| Ad hoc error scaling | Principled from VFE |
| ~3000 lines of code | Target: ~500 lines |

---

## Appendix B: Mathematical Derivations

### B.1 Why Natural Gradient?

Standard gradient descent treats all parameter directions equally. But on statistical manifolds, the "distance" between distributions is measured by KL divergence, not Euclidean distance.

The Fisher information matrix G defines the Riemannian metric:
```
G_ij = E[∂log p/∂θ_i · ∂log p/∂θ_j]
```

For Gaussians, G_μμ = Σ^(-1), so the natural gradient is:
```
∇̃_μ F = Σ · ∇_μ F
```

This ensures updates are invariant to reparameterization.

### B.2 Why Flat Gauge is Sufficient

The transport operator Ω_ij = exp(φ_i)·exp(-φ_j) serves two purposes:
1. **Frame alignment**: Transform beliefs between different coordinate systems
2. **Position encoding**: Encode relative position in the gauge difference

In the flat gauge limit (all φ_i = 0), we lose frame alignment but can still encode position through **position-dependent priors**. Since standard transformers work without explicit gauge structure, this is sufficient for language modeling.

Gauge structure becomes necessary when:
- Different tokens genuinely use different reference frames (rare in text)
- Equivariance to transformations is required (e.g., 3D point clouds)
- The domain has intrinsic symmetries (e.g., molecular structures)

---

*Document prepared for implementation of Pure FEP Transformer from first principles.*
*Based on theoretical framework from Dennis (2025) and analysis of existing implementations.*
