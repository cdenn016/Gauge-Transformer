# Pure FEP Transformer: Ground-Up Implementation Plan (REVISED)

## Executive Summary

This document provides a detailed plan to implement a **Pure Free Energy Principle (FEP) Transformer** from first principles, with **gauge frames as the core semantic/feature encoding mechanism**.

**Key Design Decisions:**
1. **Gauge Frames for Semantic Encoding**: φ encodes semantic/feature structure, NOT position
2. **Full Transport Operators**: Ω_ij = exp(φ_i)·exp(-φ_j) is used in ALL KL terms
3. **Complete VFE with Prior Coupling**: Includes the γ_ij·KL(p_i||Ω_ij·p_j) term
4. **Position via Priors Only**: Position-dependent priors (μ_p, σ_p), NO position in φ
5. **Two Timescales**: Q-flow (beliefs), P-flow (priors), and **φ-flow** (gauge frames)
6. **No Neural Networks**: Zero MLPs, zero learned projection matrices, zero activation functions

---

## I. Theoretical Foundation

### 1.1 The COMPLETE Variational Free Energy Functional

From the papers, the FULL VFE is:

```
F[{q_i}, {p_i}, {φ_i}] =
    α · Σ_i KL(q_i || p_i)                           [Self-coupling: belief-to-prior]
  + λ_β · Σ_ij β_ij · KL(q_i || Ω_ij·q_j)           [Belief alignment with transport]
  + λ_γ · Σ_ij γ_ij · KL(p_i || Ω_ij·p_j)           [Prior coupling with transport]
  - Σ_i E_{q_i}[log p(y_i | z_i)]                    [Observation likelihood]
```

where:
- **Ω_ij = exp(φ_i) · exp(-φ_j)** is the gauge transport operator
- **β_ij** are belief attention weights
- **γ_ij** are prior (model) attention weights
- **φ_i ∈ 𝔤** are gauge frames in the Lie algebra

### 1.2 Why Gauge Frames are ESSENTIAL

The gauge frames φ_i encode the **semantic reference frame** of each agent/token:

1. **Semantic Orientation**: φ encodes HOW a token "sees" the embedding space
2. **Feature Encoding**: Different tokens have different φ, encoding their semantic role
3. **Transport = Communication**: Ω_ij transforms j's beliefs into i's frame for comparison
4. **Multi-Head from Lie Algebra**: For SO(3), dim(𝔤) = 3 gives 3 natural heads

**Critical Distinction**:
- **φ encodes WHAT** (semantic features, token identity)
- **Position priors encode WHERE** (sequence position)

### 1.3 The Transport Operator

For gauge group G (typically SO(3) or SO(N)), with generators {G_a}:

```
φ_i = Σ_a φ_i^(a) · G_a    ∈ 𝔤 (Lie algebra)

Ω_ij = exp(φ_i) · exp(-φ_j)  ∈ G (Lie group)
```

The transport acts on Gaussian statistics:
```
Ω_ij · N(μ_j, Σ_j) = N(Ω_ij · μ_j, Ω_ij · Σ_j · Ω_ij^T)
```

For diagonal covariances with efficient transport:
```
(Ω · diag(σ) · Ω^T)_kk = Σ_l Ω_kl² · σ_l
```

### 1.4 Attention as Transported Belief Alignment

Attention weights emerge from KL divergence **after transport**:

```
β_ij = softmax_j(-KL(q_i || Ω_ij·q_j) / κ_β)
γ_ij = softmax_j(-KL(p_i || Ω_ij·p_j) / κ_γ)
```

**Why transport matters for attention:**
- Without transport: comparing apples to oranges
- With transport: align j's frame to i's frame, THEN compare
- Tokens with aligned frames (small ||φ_i - φ_j||) have easier communication

### 1.5 Multi-Head Attention from Lie Algebra

For G = SO(3), the Lie algebra 𝔤 = so(3) has dimension 3:
```
Number of heads H = dim(𝔤) = 3
```

Each generator G_a defines a rotation axis, creating 3 natural attention heads:
- Head 1: Rotations around x-axis
- Head 2: Rotations around y-axis
- Head 3: Rotations around z-axis

The embedding space decomposes via irreducible representations (irreps):
```
K = n_0·1 + n_1·3 + n_2·5 + ...
    [scalars] [vectors] [rank-2 tensors]
```

For K=64: could use 10 scalars + 18 vectors = 10·1 + 18·3 = 64

---

## II. Core Architecture

### 2.1 Agent Representation (Complete)

Each token position i has a **full section** of the bundle:

```
Agent i = (q_i, p_i, φ_i)

where:
  q_i = N(μ_qi, σ_qi²)   - belief (posterior)
  p_i = N(μ_pi, σ_pi²)   - prior (generative model)
  φ_i ∈ ℝ^{phi_dim}      - gauge frame (semantic orientation)
```

**Shapes:**
```
μ_q:    (batch, seq_len, embed_dim)  - belief means
σ_q:    (batch, seq_len, embed_dim)  - belief stds
μ_p:    (seq_len, embed_dim)         - position prior means
σ_p:    (seq_len, embed_dim)         - position prior stds
φ:      (batch, seq_len, phi_dim)    - gauge frames
```

For SO(3): phi_dim = 3
For SO(N): phi_dim = N(N-1)/2

### 2.2 Token Prior Bank (with Gauge Frames!)

Each vocabulary token v has a **complete prior section**:

```python
class TokenPriorBank:
    """
    Each token v has: π_v = (μ_v, σ_v, φ_v)

    The gauge frame φ_v encodes the token's SEMANTIC orientation.
    Different tokens "see" the embedding space from different angles.
    """
    μ_tokens: (vocab_size, embed_dim)   # semantic content
    σ_tokens: (vocab_size, embed_dim)   # uncertainty
    φ_tokens: (vocab_size, phi_dim)     # semantic frame
```

**Encoding**: Initialize agent from token prior:
```
q_i ← N(μ_{token[i]}, σ_{token[i]})
φ_i ← φ_{token[i]}
```

**Decoding**: Output via transported KL:
```
logits_v = -KL(q_i || Ω_{iv}·π_v) / τ

where Ω_{iv} = exp(φ_i)·exp(-φ_v) transports token prior to agent's frame
```

### 2.3 Position-Dependent Priors (NO φ for position!)

Position is encoded in priors, NOT in gauge frames:

```python
class PositionPriors:
    """
    Position structure in (μ_p, σ_p) only.
    Gauge frames φ are for SEMANTIC encoding.
    """
    μ_p: (max_seq_len, embed_dim)   # position-dependent means
    σ_p: (max_seq_len, embed_dim)   # position-dependent stds
    # NO φ_position!
```

**Why this separation?**
- φ should be **shift-invariant** (same token → same φ regardless of position)
- Position structure emerges from (μ_p, σ_p) learning different patterns
- Transport Ω_ij depends on semantic frames, not position

### 2.4 Single Layer Structure

```
Input: (μ_q, σ_q, φ) from previous layer or token encoding
       (μ_p, σ_p) position priors for this layer

1. COMPUTE TRANSPORT OPERATORS:
   For all pairs (i,j):
     Ω_ij = exp(φ_i·G) · exp(-φ_j·G)

2. COMPUTE ATTENTION (from transported beliefs):
   KL_ij = KL(q_i || Ω_ij·q_j)
   β_ij = softmax_j(-KL_ij / κ_β)

3. VFE GRADIENT DESCENT (Q-flow):
   for step in range(n_vfe_steps):
     F = α·Σ KL(q||p) + λ_β·Σ β·KL(q||Ω·q) + λ_γ·Σ γ·KL(p||Ω·p) - log p(y|q)

     # Natural gradient updates
     μ_q ← μ_q - η_μ · σ_q² · ∂F/∂μ_q
     σ_q ← σ_q · exp(-η_σ · ∂F/∂log_σ_q)
     φ ← φ - η_φ · ∂F/∂φ

     # Optionally recompute β (dynamic attention)
     β_ij = softmax(-KL(q_i||Ω_ij·q_j) / κ_β)

Output: (μ_q, σ_q, φ) updated beliefs and frames
```

---

## III. The VFE Components in Detail

### 3.1 Self-Coupling: KL(q_i || p_i)

Standard diagonal Gaussian KL (no transport needed - same agent):

```python
def kl_self_coupling(μ_q, σ_q, μ_p, σ_p, eps=1e-6):
    """KL(q || p) for diagonal Gaussians."""
    var_q = σ_q.square() + eps
    var_p = σ_p.square() + eps

    kl = 0.5 * (
        torch.log(var_p / var_q)
        + var_q / var_p
        + (μ_q - μ_p).square() / var_p
        - 1.0
    )
    return kl.sum(dim=-1)  # (B, N)
```

### 3.2 Belief Alignment: Σ β_ij · KL(q_i || Ω_ij·q_j)

**WITH GAUGE TRANSPORT**:

```python
def compute_transport_operators(phi, generators):
    """
    Compute Ω_ij = exp(φ_i·G)·exp(-φ_j·G) for all pairs.

    Args:
        phi: (B, N, phi_dim) gauge frames
        generators: (phi_dim, K, K) Lie algebra generators

    Returns:
        Ω: (B, N, N, K, K) transport operators
    """
    B, N, phi_dim = phi.shape
    K = generators.shape[1]

    # Compute exp(φ·G) for each agent
    phi_dot_G = torch.einsum('bna,aij->bnij', phi, generators)  # (B, N, K, K)
    R = torch.linalg.matrix_exp(phi_dot_G)  # (B, N, K, K)

    # Ω_ij = R_i @ R_j^T
    Omega = torch.einsum('bnik,bnjk->bnijk', R, R)  # (B, N, N, K, K)
    # Note: R_j^T = inv(R_j) for orthogonal matrices

    return Omega

def kl_transported(μ_q, σ_q, Omega, eps=1e-6):
    """
    KL(q_i || Ω_ij·q_j) for all pairs.

    Transported belief: Ω_ij·q_j = N(Ω_ij·μ_j, Ω_ij·Σ_j·Ω_ij^T)
    """
    B, N, K = μ_q.shape
    var_q = σ_q.square() + eps  # (B, N, K)

    # Transport means: Ω_ij @ μ_j
    μ_transported = torch.einsum('bnijk,bjk->bnik', Omega, μ_q)  # (B, N, N, K)

    # Transport variances (diagonal): (Ω @ diag(σ²) @ Ω^T)_kk = Σ_l Ω_kl² · σ_l²
    var_transported = torch.einsum('bnijk,bjk,bnijk->bnik',
                                    Omega, var_q, Omega)  # (B, N, N, K)

    # KL(q_i || transported_j)
    μ_i = μ_q.unsqueeze(2)  # (B, N, 1, K)
    var_i = var_q.unsqueeze(2)  # (B, N, 1, K)

    kl = 0.5 * (
        torch.log(var_transported / var_i)
        + var_i / var_transported
        + (μ_i - μ_transported).square() / var_transported
        - 1.0
    )
    return kl.sum(dim=-1)  # (B, N, N)

def compute_attention(kl_matrix, kappa, mask=None):
    """β_ij = softmax_j(-KL_ij / κ)"""
    logits = -kl_matrix / kappa
    if mask is not None:
        logits = logits.masked_fill(~mask, float('-inf'))
    return F.softmax(logits, dim=-1)
```

### 3.3 Prior Coupling: Σ γ_ij · KL(p_i || Ω_ij·p_j)

**THE MISSING TERM** - ensures priors form a coherent world model:

```python
def prior_coupling_term(μ_p, σ_p, Omega, kappa_gamma, mask=None):
    """
    Σ γ_ij · KL(p_i || Ω_ij·p_j)

    This term ensures priors are mutually consistent under transport.
    """
    # Compute KL between priors with transport
    kl_priors = kl_transported_priors(μ_p, σ_p, Omega)  # (N, N)

    # Compute γ attention weights
    gamma = compute_attention(kl_priors, kappa_gamma, mask)  # (N, N)

    # Weighted sum
    prior_coupling = (gamma * kl_priors).sum()

    return prior_coupling, gamma
```

### 3.4 Observation Likelihood

Output via **transported KL** to token priors:

```python
def observation_likelihood(μ_q, σ_q, φ, token_priors, tau=1.0):
    """
    logits_v = -KL(q_i || Ω_{iv}·π_v) / τ

    Transport each token prior into the agent's frame before comparing.
    """
    B, N, K = μ_q.shape
    V = token_priors.μ_tokens.shape[0]

    # Compute transport from each agent to each token prior
    # Ω_{iv} = exp(φ_i)·exp(-φ_v)
    Omega_to_tokens = compute_agent_to_token_transport(
        φ, token_priors.φ_tokens, generators
    )  # (B, N, V, K, K)

    # Transport token priors
    μ_transported = transport_means(token_priors.μ_tokens, Omega_to_tokens)
    σ_transported = transport_stds(token_priors.σ_tokens, Omega_to_tokens)

    # KL to each transported token prior
    kl_to_tokens = compute_kl_batch(μ_q, σ_q, μ_transported, σ_transported)

    # Logits
    logits = -kl_to_tokens / tau  # (B, N, V)

    return logits
```

### 3.5 Complete VFE Computation

```python
def compute_vfe(μ_q, σ_q, φ, μ_p, σ_p, Omega, target_ids, token_priors, config):
    """
    FULL Variational Free Energy:

    F = α·Σ_i KL(q_i||p_i)
      + λ_β·Σ_ij β_ij·KL(q_i||Ω_ij·q_j)
      + λ_γ·Σ_ij γ_ij·KL(p_i||Ω_ij·p_j)
      - Σ_i log p(y_i|q_i)
    """
    # 1. Self-coupling
    kl_self = kl_self_coupling(μ_q, σ_q, μ_p, σ_p)
    F_self = config.alpha * kl_self.sum()

    # 2. Belief alignment (WITH TRANSPORT)
    kl_beliefs = kl_transported(μ_q, σ_q, Omega)
    beta = compute_attention(kl_beliefs, config.kappa_beta, mask)
    F_belief = config.lambda_beta * (beta * kl_beliefs).sum()

    # 3. Prior coupling (WITH TRANSPORT) - THE MISSING TERM!
    kl_priors = kl_transported_priors(μ_p, σ_p, Omega)
    gamma = compute_attention(kl_priors, config.kappa_gamma, mask)
    F_prior = config.lambda_gamma * (gamma * kl_priors).sum()

    # 4. Observation likelihood
    logits = observation_likelihood(μ_q, σ_q, φ, token_priors, config.tau)
    ce_loss = F.cross_entropy(logits.view(-1, V), target_ids.view(-1))
    F_obs = ce_loss * target_ids.numel()

    F_total = F_self + F_belief + F_prior + F_obs

    return F_total, {
        'F_self': F_self.item(),
        'F_belief': F_belief.item(),
        'F_prior': F_prior.item(),
        'F_obs': F_obs.item(),
        'beta': beta,
        'gamma': gamma
    }
```

---

## IV. Gradient Computation (including ∂F/∂φ!)

### 4.1 Gradient with respect to Gauge Frames: ∂F/∂φ_i

This is CRUCIAL - gauge frames evolve via VFE gradient descent:

```python
def compute_phi_gradient(φ, μ_q, σ_q, μ_p, σ_p, β, γ, generators, config):
    """
    ∂F/∂φ_i includes contributions from:
    1. Belief alignment: Σ_j [∂β_ij/∂φ_i · KL_ij + β_ij · ∂KL_ij/∂φ_i]
    2. Prior coupling:   Σ_j [∂γ_ij/∂φ_i · KL_ij^p + γ_ij · ∂KL_ij^p/∂φ_i]
    3. Others to me:     Σ_k β_ki · ∂KL(q_k||Ω_ki·q_i)/∂φ_i
    4. Priors others:    Σ_k γ_ki · ∂KL(p_k||Ω_ki·p_i)/∂φ_i

    The gradient flows through the transport operator Ω_ij = exp(φ_i)·exp(-φ_j)
    """
    # Use autograd for correctness, then optimize if needed
    φ.requires_grad_(True)

    # Recompute F with gradient tracking
    Omega = compute_transport_operators(φ, generators)
    F, _ = compute_vfe(μ_q, σ_q, φ, μ_p, σ_p, Omega, ...)

    # Gradient via autograd
    grad_phi = torch.autograd.grad(F, φ, retain_graph=True)[0]

    return grad_phi
```

### 4.2 Three-Timescale Updates

```python
def vfe_step(μ_q, σ_q, φ, μ_p, σ_p, generators, config):
    """
    Single VFE gradient descent step updating:
    - μ_q (belief means) - fast
    - σ_q (belief stds) - fast
    - φ (gauge frames) - medium (can be slower than beliefs)
    """
    # Compute transport operators
    Omega = compute_transport_operators(φ, generators)

    # Compute VFE and all gradients
    with torch.enable_grad():
        μ_q.requires_grad_(True)
        σ_q.requires_grad_(True)
        φ.requires_grad_(True)

        F, metrics = compute_vfe(μ_q, σ_q, φ, μ_p, σ_p, Omega, ...)

        grad_μ = torch.autograd.grad(F, μ_q, retain_graph=True)[0]
        grad_σ = torch.autograd.grad(F, σ_q, retain_graph=True)[0]
        grad_φ = torch.autograd.grad(F, φ)[0]

    # Natural gradient updates
    var_q = σ_q.square()
    μ_q_new = μ_q - config.lr_mu * var_q * grad_μ
    σ_q_new = σ_q * torch.exp(-config.lr_sigma * grad_σ * σ_q)
    φ_new = φ - config.lr_phi * grad_φ

    # Clamp for stability
    σ_q_new = σ_q_new.clamp(min=config.variance_floor)
    φ_new = clamp_phi_norm(φ_new, config.phi_max_norm)  # e.g., π

    return μ_q_new, σ_q_new, φ_new
```

---

## V. Why Semantic Encoding in φ Works

### 5.1 Token Identity via Gauge Frame

Different tokens have different "orientations" in semantic space:

```
Token "cat" → φ_cat = [0.3, -0.1, 0.5]   (some orientation)
Token "dog" → φ_dog = [0.4, -0.2, 0.6]   (similar orientation - similar semantics!)
Token "run" → φ_run = [-0.5, 0.8, 0.1]   (different orientation - different category)
```

When computing attention:
- cat attending to dog: Ω_{cat,dog} ≈ I (small rotation, easy transport)
- cat attending to run: Ω_{cat,run} = large rotation (harder transport)

This creates **semantic clustering** in attention patterns!

### 5.2 Transport Cost as Semantic Distance

The KL divergence after transport:
```
KL(q_cat || Ω_{cat,run}·q_run)
```
includes an implicit cost for the transport itself. Even if the beliefs (μ, σ) are similar, if the frames are misaligned, attention is reduced.

### 5.3 Multi-Head = Multiple Semantic Axes

For SO(3) with 3 generators:
- Head 1 (G_x): Captures one axis of semantic variation
- Head 2 (G_y): Captures another axis
- Head 3 (G_z): Captures third axis

Different heads attend to different aspects of semantic similarity.

---

## VI. Position Encoding (WITHOUT φ)

### 6.1 Position in Priors Only

```python
class LayerPriors:
    """
    Position structure emerges from position-dependent priors.
    NOT from gauge frames.
    """
    def __init__(self, max_seq_len, embed_dim):
        # Position-dependent prior means
        self.μ_p = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.1)
        # Position-dependent prior stds
        self.log_σ_p = nn.Parameter(torch.zeros(max_seq_len, embed_dim))

        # NO φ_position - gauge frames come from TOKEN priors only!
```

### 6.2 Why Position Emerges

Through P-flow, position priors learn:
- Position 0 sees beginning-of-sequence patterns
- Position N-1 sees end-of-sequence patterns
- Middle positions learn their characteristic patterns

The causal mask ensures positional asymmetry. Priors naturally differentiate.

---

## VII. Complete Model Architecture

### 7.1 Configuration

```python
@dataclass
class PureFEPConfig:
    # Architecture
    vocab_size: int = 256
    embed_dim: int = 64           # K
    n_layers: int = 4
    max_seq_len: int = 128

    # Gauge structure
    gauge_group: str = 'SO3'      # 'SO3' or 'SON'
    phi_dim: int = 3              # dim(𝔤): 3 for SO(3), N(N-1)/2 for SO(N)
    n_heads: int = 3              # = phi_dim for SO(3)

    # VFE weights
    alpha: float = 0.1            # Self-coupling
    lambda_beta: float = 1.0      # Belief alignment
    lambda_gamma: float = 0.1     # Prior coupling (NEW!)
    kappa_beta: float = 1.0       # Belief attention temperature
    kappa_gamma: float = 1.0      # Prior attention temperature
    tau: float = 1.0              # Output temperature

    # Q-flow (fast timescale)
    n_vfe_steps: int = 10
    lr_mu: float = 0.1
    lr_sigma: float = 0.01
    lr_phi: float = 0.05          # Gauge frame learning rate

    # P-flow (slow timescale)
    lr_prior: float = 0.01
    lr_token_prior: float = 0.01

    # Stability
    variance_floor: float = 1e-4
    phi_max_norm: float = 3.14159  # π radians
    eps: float = 1e-6
```

### 7.2 Model Definition

```python
class PureFEPTransformer(nn.Module):
    """
    Pure FEP Transformer with FULL gauge structure.

    - Gauge frames φ encode SEMANTIC features
    - Transport Ω_ij = exp(φ_i)·exp(-φ_j) in ALL KL terms
    - Position encoded in priors (μ_p, σ_p), NOT in φ
    - Complete VFE includes prior coupling term
    """

    def __init__(self, config: PureFEPConfig):
        super().__init__()
        self.config = config

        # Generate Lie algebra generators
        if config.gauge_group == 'SO3':
            self.generators = generate_so3_generators()  # (3, K, K)
        else:
            self.generators = generate_soN_generators(config.phi_dim)
        self.register_buffer('generators_buf', self.generators)

        # Token prior bank (μ, σ, φ for each token)
        self.token_priors = TokenPriorBank(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            phi_dim=config.phi_dim,
            generators=self.generators
        )

        # Position priors for each layer (μ, σ only - NO φ!)
        self.position_priors = nn.ModuleList([
            PositionPriors(config.max_seq_len, config.embed_dim)
            for _ in range(config.n_layers)
        ])

    def forward(self, input_ids, target_ids=None):
        B, N = input_ids.shape
        device = input_ids.device

        # === ENCODING ===
        # Initialize (μ_q, σ_q, φ) from token priors
        μ_q, σ_q, φ = self.token_priors.encode(input_ids)

        # Causal mask
        mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))

        # === LAYERS ===
        for layer_idx in range(self.config.n_layers):
            μ_p = self.position_priors[layer_idx].μ_p[:N]
            σ_p = self.position_priors[layer_idx].σ_p[:N]

            # Q-flow with gauge evolution
            μ_q, σ_q, φ = self.q_flow(
                μ_q, σ_q, φ, μ_p, σ_p, mask, target_ids
            )

        # === DECODING ===
        logits = self.token_priors.decode(μ_q, σ_q, φ)

        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                target_ids.view(-1)
            )

        return logits, loss

    def q_flow(self, μ_q, σ_q, φ, μ_p, σ_p, mask, targets):
        """VFE gradient descent on beliefs AND gauge frames."""
        for step in range(self.config.n_vfe_steps):
            μ_q, σ_q, φ = vfe_step(
                μ_q, σ_q, φ, μ_p, σ_p,
                self.generators_buf, mask, targets,
                self.token_priors, self.config
            )
        return μ_q, σ_q, φ
```

---

## VIII. What We KEEP vs AVOID

### KEEP (Core FEP with Gauge Structure)

| Component | Role |
|-----------|------|
| Gauge frames φ | Semantic/feature encoding |
| Transport Ω_ij | Frame alignment for comparison |
| KL(q_i \|\| Ω_ij·q_j) | Transported belief alignment |
| KL(p_i \|\| Ω_ij·p_j) | Prior coupling (world model coherence) |
| ∂F/∂φ | Gauge frame evolution |
| Multi-head from dim(𝔤) | Natural head structure |

### AVOID (Ad Hoc / Neural)

| Eliminated | Reason |
|------------|--------|
| Position in φ | φ is for semantics, not position |
| Sinusoidal encoding | Position emerges from priors |
| W_Q, W_K, W_V matrices | Attention from KL geometry |
| MLPs / FFN | VFE gradient descent |
| GELU/ReLU | Softmax gradient nonlinearity |
| Learned projections | All from VFE |

---

## IX. Implementation Phases (Revised)

### Phase 1: Gauge Infrastructure (Week 1)
1. Implement SO(3) generators
2. Implement transport operator computation
3. Implement transported KL divergence
4. Test gauge equivariance properties

### Phase 2: Complete VFE (Week 2)
1. Implement all four VFE terms
2. Implement gradient computation (including ∂F/∂φ)
3. Validate gradients with finite differences
4. Test on simple examples

### Phase 3: Token & Position Priors (Week 3)
1. Implement TokenPriorBank with φ_tokens
2. Implement PositionPriors (μ, σ only)
3. Encoding/decoding with transport
4. P-flow updates

### Phase 4: Full Model (Week 4)
1. Stack layers
2. Training loop with Q-flow + P-flow
3. WikiText-2 experiments
4. Compare to standard transformer

### Phase 5: Analysis (Week 5)
1. Visualize learned φ structure
2. Analyze attention patterns
3. Study semantic clustering
4. Multi-head decomposition

---

## X. Key Equations Summary (REVISED)

| Component | Equation |
|-----------|----------|
| **Agent** | (q_i, p_i, φ_i) = (N(μ_qi, σ_qi²), N(μ_pi, σ_pi²), φ_i ∈ 𝔤) |
| **Transport** | Ω_ij = exp(φ_i·G) · exp(-φ_j·G) |
| **Transported Mean** | μ̃_j = Ω_ij · μ_j |
| **Transported Var** | σ̃_j² = diag(Ω_ij · diag(σ_j²) · Ω_ij^T) |
| **Belief Attention** | β_ij = softmax_j(-KL(q_i \|\| Ω_ij·q_j) / κ_β) |
| **Prior Attention** | γ_ij = softmax_j(-KL(p_i \|\| Ω_ij·p_j) / κ_γ) |
| **VFE** | F = α·Σ KL(q\|\|p) + λ_β·Σ β·KL(q\|\|Ω·q) + λ_γ·Σ γ·KL(p\|\|Ω·p) - log p(y) |
| **Natural Gradient μ** | μ ← μ - η_μ · σ² · ∂F/∂μ |
| **Gauge Update** | φ ← φ - η_φ · ∂F/∂φ |

---

*Revised plan incorporating gauge frames as CORE semantic encoding mechanism.*
*φ encodes WHAT (semantics), priors encode WHERE (position).*
*Full transport Ω_ij in ALL KL terms including prior coupling.*
