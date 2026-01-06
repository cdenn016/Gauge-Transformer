# Pure FEP Transformer: Ground-Up Implementation Plan (REVISED v2)

## Executive Summary

This document provides a detailed plan to implement a **Pure Free Energy Principle (FEP) Transformer** from first principles, with **gauge frames as the core semantic/feature encoding mechanism**.

**Key Design Decisions:**
1. **Gauge Frames for Semantic Encoding**: φ encodes semantic/feature structure, NOT position
2. **Full Transport via BCH**: Ω_ij computed using Baker-Campbell-Hausdorff formula, NOT naive subtraction
3. **SO(N) Compatible**: Works with fundamental and higher irreps of any SO(N)
4. **Block-Diagonal Covariance**: Preserves correlations within irrep blocks
5. **Complete VFE with Prior Coupling**: Includes the γ_ij·KL(p_i||Ω_ij·p_j) term
6. **Position via Priors Only**: Position-dependent priors (μ_p, σ_p), NO position in φ
7. **Haar Initialization**: Break symmetry with proper group-theoretic initialization
8. **Ouroboros Tower**: Optional Phase 2 extension for long-range memory
9. **No Neural Networks**: Zero MLPs, zero learned projection matrices, zero activation functions

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
  + Σ_i Σ_d decay^d · KL(p_i || h_i^d)              [Ouroboros Tower - PHASE 2]
```

where:
- **Ω_ij = exp(φ_i) · exp(-φ_j)** is the gauge transport operator
- **β_ij** are belief attention weights
- **γ_ij** are prior (model) attention weights
- **φ_i ∈ 𝔤** are gauge frames in the Lie algebra
- **h_i^d** are hyperpriors from ancestor depth d (Ouroboros)

### 1.2 Why Gauge Frames are ESSENTIAL

The gauge frames φ_i encode the **semantic reference frame** of each agent/token:

1. **Semantic Orientation**: φ encodes HOW a token "sees" the embedding space
2. **Feature Encoding**: Different tokens have different φ, encoding their semantic role
3. **Transport = Communication**: Ω_ij transforms j's beliefs into i's frame for comparison
4. **Multi-Head from Lie Algebra**: For SO(N), dim(𝔤) = N(N-1)/2 gives natural heads

**Critical Distinction**:
- **φ encodes WHAT** (semantic features, token identity)
- **Position priors encode WHERE** (sequence position)

### 1.3 Multi-Head: The Geometry IS Multi-Headed

**Important**: We do NOT treat generators as separate "heads" with separate β matrices.

The VFE is a **single scalar**. The gradient ∂F/∂φ naturally decomposes:

```
∂F/∂φ_i = Σ_a (∂F/∂φ_i^(a)) · e_a
```

where e_a are basis vectors in ℝ^{dim(𝔤)}. Each generator component captures a different axis of semantic variation automatically.

---

## II. Efficient Transport via Baker-Campbell-Hausdorff

### 2.1 Why Naive Subtraction is WRONG

For non-abelian groups like SO(N):
```
exp(φ_i) · exp(-φ_j) ≠ exp(φ_i - φ_j)   ← WRONG!
```

The group is non-commutative. We MUST use the BCH formula.

### 2.2 The BCH Formula

```
exp(X) · exp(Y) = exp(X + Y + ½[X,Y] + 1/12[X,[X,Y]] - 1/12[Y,[X,Y]] + ...)
```

For Ω_ij = exp(X_i) · exp(-X_j) where X_i = φ_i^(a) G_a:

```
Ω_ij = exp(φ_ij · G)

where φ_ij = φ_i - φ_j - ½[φ_i, φ_j]_𝔤 + O(φ³)
```

### 2.3 Structure Constants for SO(N)

The Lie bracket in coordinates uses structure constants f_abc:

```
[G_a, G_b] = Σ_c f_abc G_c
```

For SO(N), the generators are antisymmetric N×N matrices indexed by pairs (p,q) with p < q:
```
(G_pq)_ij = δ_pi δ_qj - δ_pj δ_qi
```

The structure constants are:
```
f_{(pq)(rs)(tu)} = δ_qr δ_pt δ_su - δ_qr δ_ps δ_tu - δ_pr δ_qt δ_su + δ_pr δ_qs δ_tu + ...
```

```python
def compute_soN_structure_constants(N):
    """
    Compute structure constants f_abc for so(N).

    dim(so(N)) = N(N-1)/2

    Returns:
        f: (dim_g, dim_g, dim_g) antisymmetric tensor
    """
    dim_g = N * (N - 1) // 2
    f = torch.zeros(dim_g, dim_g, dim_g)

    # Index mapping: (p,q) with p<q -> linear index
    def pair_to_idx(p, q):
        # Row-major upper triangular indexing
        return p * N - p * (p + 1) // 2 + (q - p - 1)

    def idx_to_pair(k):
        # Inverse mapping
        p = 0
        while pair_to_idx(p, N-1) < k:
            p += 1
        if p > 0:
            p -= 1
        while pair_to_idx(p, p+1) + (N - p - 2) < k:
            p += 1
        q = k - pair_to_idx(p, p+1) + p + 1
        return p, q

    # Compute [G_ab, G_cd] for all pairs
    for k1 in range(dim_g):
        a, b = idx_to_pair(k1)  # G_ab
        for k2 in range(dim_g):
            c, d = idx_to_pair(k2)  # G_cd

            # [G_ab, G_cd] = δ_bc G_ad - δ_ac G_bd - δ_bd G_ac + δ_ad G_bc
            contributions = []

            if b == c and a != d:
                p, q = (a, d) if a < d else (d, a)
                sign = 1.0 if a < d else -1.0
                contributions.append((pair_to_idx(p, q), sign))

            if a == c and b != d:
                p, q = (b, d) if b < d else (d, b)
                sign = -1.0 if b < d else 1.0
                contributions.append((pair_to_idx(p, q), sign))

            if b == d and a != c:
                p, q = (a, c) if a < c else (c, a)
                sign = -1.0 if a < c else 1.0
                contributions.append((pair_to_idx(p, q), sign))

            if a == d and b != c:
                p, q = (b, c) if b < c else (c, b)
                sign = 1.0 if b < c else -1.0
                contributions.append((pair_to_idx(p, q), sign))

            for idx, sign in contributions:
                f[k1, k2, idx] += sign

    return f
```

### 2.4 BCH Combination in Lie Algebra Coordinates

```python
def bch_combine(phi_i, phi_j, structure_constants, order=2):
    """
    Compute φ_ij such that exp(φ_ij·G) ≈ exp(φ_i·G)·exp(-φ_j·G)

    Uses BCH formula in Lie algebra coordinates - NO matrix exponentials!

    Args:
        phi_i: (B, N, dim_g) - source gauge frames
        phi_j: (B, N, dim_g) - target gauge frames
        structure_constants: (dim_g, dim_g, dim_g) - f_abc
        order: BCH truncation (1=naive, 2=first commutator, 3=second order)

    Returns:
        phi_ij: (B, N, N, dim_g) - combined Lie algebra elements
    """
    B, N, dim_g = phi_i.shape

    # Expand for pairwise computation
    phi_i_exp = phi_i.unsqueeze(2)  # (B, N, 1, dim_g)
    phi_j_exp = phi_j.unsqueeze(1)  # (B, 1, N, dim_g)

    # Order 1: naive difference (only correct for abelian groups!)
    phi_ij = phi_i_exp - phi_j_exp  # (B, N, N, dim_g)

    if order >= 2:
        # First commutator: -½[φ_i, φ_j]
        # [X_i, X_j]^c = Σ_{a,b} φ_i^a φ_j^b f_abc
        commutator = torch.einsum('bnia,bnjb,abc->bnijc',
                                   phi_i_exp, phi_j_exp,
                                   structure_constants)  # (B, N, N, dim_g)
        phi_ij = phi_ij - 0.5 * commutator

    if order >= 3:
        # Second order: +1/12[φ_i,[φ_i,φ_j]] - 1/12[φ_j,[φ_i,φ_j]]
        # [φ_i, commutator]^c = Σ_{a,b} φ_i^a comm^b f_abc
        comm_i_comm = torch.einsum('bnia,bnijb,abc->bnijc',
                                    phi_i_exp, commutator,
                                    structure_constants)
        comm_j_comm = torch.einsum('bnjb,bnija,abc->bnijc',
                                    phi_j_exp, commutator,
                                    structure_constants)
        phi_ij = phi_ij + (1.0/12.0) * comm_i_comm - (1.0/12.0) * comm_j_comm

    return phi_ij
```

### 2.5 Efficient Rotation via Rodrigues (SO(3)) and Series (SO(N))

**Key insight**: We never materialize the full (B, N, N, K, K) transport tensor!

#### For SO(3): Rodrigues Formula

```python
def rodrigues_rotate(phi_ij, v):
    """
    Rodrigues formula: R(φ)·v = v·cos(θ) + (k×v)·sin(θ) + k(k·v)(1-cos(θ))

    where θ = ||φ||, k = φ/θ (unit axis)

    Args:
        phi_ij: (B, N, N, 3) - axis-angle in so(3)
        v: (B, N, 3) - vectors to rotate

    Returns:
        Rv: (B, N, N, 3) - rotated vectors

    Complexity: O(B × N² × K) instead of O(B × N² × K²)
    """
    B, N1, N2, _ = phi_ij.shape

    # Compute angle and axis
    theta = phi_ij.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, N, N, 1)
    k = phi_ij / theta  # unit axis (B, N, N, 3)

    cos_t = torch.cos(theta)  # (B, N, N, 1)
    sin_t = torch.sin(theta)

    # Expand v for broadcasting: (B, 1, N, 3)
    v_exp = v.unsqueeze(1)

    # k · v (dot product)
    k_dot_v = (k * v_exp).sum(dim=-1, keepdim=True)  # (B, N, N, 1)

    # k × v (cross product)
    k_cross_v = torch.cross(k, v_exp.expand(-1, N1, -1, -1), dim=-1)  # (B, N, N, 3)

    # Rodrigues formula
    Rv = v_exp * cos_t + k_cross_v * sin_t + k * k_dot_v * (1 - cos_t)

    return Rv  # (B, N, N, 3)
```

#### For SO(N): Truncated Exponential Series

```python
def soN_rotate(phi_ij, v, generators, max_terms=4):
    """
    Rotate v by exp(φ_ij · G) using truncated series.

    exp(X) ≈ I + X + X²/2! + X³/3! + ...

    Args:
        phi_ij: (B, N, N, dim_g) - Lie algebra elements
        v: (B, N, K) - vectors in representation space
        generators: (dim_g, K, K) - Lie algebra generators
        max_terms: truncation order

    Returns:
        Rv: (B, N, N, K)
    """
    B, N1, N2, dim_g = phi_ij.shape
    K = v.shape[-1]

    # Compute X = φ · G as (B, N, N, K, K) matrix
    # But DON'T materialize - apply term by term

    v_exp = v.unsqueeze(1)  # (B, 1, N, K)
    result = v_exp.clone()  # Identity term

    # X·v term
    Xv = torch.einsum('bnija,ajk,bnk->bnij', phi_ij, generators, v_exp.squeeze(1))
    result = result + Xv.unsqueeze(-1) if K == 1 else result + Xv

    # Higher order terms: X^n·v / n!
    current = Xv
    for n in range(2, max_terms + 1):
        # X · current
        current = torch.einsum('bnija,ajk,bnijk->bnijk',
                               phi_ij, generators, current.unsqueeze(-1)).squeeze(-1)
        result = result + current / math.factorial(n)

    return result
```

#### For Reducible Representations (Multiple Irreps)

```python
def rotate_reducible(phi_ij, v, irrep_structure, generators_per_irrep):
    """
    Rotate vectors in a reducible representation K = ⊕_ℓ n_ℓ · (2ℓ+1)

    Args:
        phi_ij: (B, N, N, dim_g)
        v: (B, N, K) where K = Σ n_ℓ · dim(irrep_ℓ)
        irrep_structure: [(irrep_dim, multiplicity, start_idx), ...]
        generators_per_irrep: dict mapping irrep_dim -> generators

    Returns:
        Rv: (B, N, N, K)
    """
    B, N1, N2, _ = phi_ij.shape
    K = v.shape[-1]

    Rv = torch.zeros(B, N1, N2, K, device=v.device, dtype=v.dtype)

    for irrep_dim, mult, start in irrep_structure:
        end = start + irrep_dim * mult
        v_block = v[..., start:end].reshape(B, -1, mult, irrep_dim)

        if irrep_dim == 1:
            # Scalars are invariant under SO(N)
            Rv[..., start:end] = v_block.unsqueeze(1).expand(-1, N1, N2, -1, -1).reshape(B, N1, N2, -1)
        else:
            # Apply appropriate rotation
            gens = generators_per_irrep[irrep_dim]
            for m in range(mult):
                v_m = v_block[..., m, :]  # (B, N, irrep_dim)
                if irrep_dim == 3:
                    Rv_m = rodrigues_rotate(phi_ij, v_m)
                else:
                    Rv_m = soN_rotate(phi_ij, v_m, gens)
                Rv[..., start + m*irrep_dim : start + (m+1)*irrep_dim] = Rv_m

    return Rv
```

---

## III. Block-Diagonal Covariance (Mean-Field Fix)

### 3.1 The Problem with Diagonal Covariance

When we transport diagonal covariance:
```
Σ' = Ω · diag(σ²) · Ω^T
```

The result Σ' is generally **full**, not diagonal. Projecting back to diagonal via:
```
σ'_k² = (Ω · diag(σ²) · Ω^T)_kk = Σ_l Ω_kl² · σ_l²
```

**discards the off-diagonal correlations** generated by the rotation.

### 3.2 Block-Diagonal Solution

Use block-diagonal covariance aligned with irreps:

```
Σ = diag(Σ_scalar₁, ..., Σ_scalar_n, Σ_vector₁, ..., Σ_vector_m, ...)
```

where:
- Scalar blocks: 1×1 (trivially diagonal)
- Vector blocks: 3×3 (full covariance within the 3D subspace)
- Rank-2 blocks: 5×5, etc.

Transport preserves block structure because irreps don't mix:
```
Ω · Σ_block · Ω^T = Σ'_block  (same shape!)
```

```python
@dataclass
class IrrepStructure:
    """Defines how K decomposes into irreps."""
    irreps: List[Tuple[int, int]]  # [(dim, multiplicity), ...]

    @property
    def total_dim(self):
        return sum(dim * mult for dim, mult in self.irreps)

    def get_block_indices(self):
        """Return (start, end, dim) for each irrep block."""
        indices = []
        pos = 0
        for dim, mult in self.irreps:
            for _ in range(mult):
                indices.append((pos, pos + dim, dim))
                pos += dim
        return indices


class BlockDiagonalCovariance(nn.Module):
    """
    Block-diagonal covariance respecting irrep structure.
    """
    def __init__(self, irrep_structure: IrrepStructure):
        super().__init__()
        self.irrep_structure = irrep_structure

        # Store Cholesky factors for each block (ensures positive definiteness)
        self.register_parameter('log_diag', None)  # Diagonal elements
        self.register_parameter('off_diag', None)  # Off-diagonal (lower triangular)

        # Initialize parameters
        self._init_params()

    def get_covariance_blocks(self, sigma_params):
        """
        Convert parameters to list of covariance blocks.

        Args:
            sigma_params: (B, N, n_params) packed parameters

        Returns:
            blocks: list of (B, N, dim, dim) covariance matrices
        """
        blocks = []
        param_idx = 0

        for start, end, dim in self.irrep_structure.get_block_indices():
            if dim == 1:
                # Scalar: just variance
                var = sigma_params[..., param_idx:param_idx+1].exp()
                blocks.append(var.unsqueeze(-1))  # (B, N, 1, 1)
                param_idx += 1
            else:
                # Full block: Cholesky parameterization
                n_params = dim + dim * (dim - 1) // 2
                L = self._unpack_cholesky(sigma_params[..., param_idx:param_idx+n_params], dim)
                blocks.append(L @ L.transpose(-1, -2))  # (B, N, dim, dim)
                param_idx += n_params

        return blocks

    def transport_blocks(self, blocks, phi_ij, generators_per_irrep):
        """
        Transport each covariance block: Σ' = Ω · Σ · Ω^T
        """
        transported = []
        block_idx = 0

        for start, end, dim in self.irrep_structure.get_block_indices():
            block = blocks[block_idx]  # (B, N, dim, dim)

            if dim == 1:
                # Scalars are invariant
                transported.append(block.unsqueeze(1).expand(-1, phi_ij.shape[1], -1, -1, -1))
            else:
                # Compute Ω for this irrep and transport
                Omega = compute_irrep_rotation(phi_ij, dim, generators_per_irrep[dim])
                # Σ' = Ω @ Σ @ Ω^T
                block_exp = block.unsqueeze(1)  # (B, 1, N, dim, dim)
                transported_block = Omega @ block_exp @ Omega.transpose(-1, -2)
                transported.append(transported_block)

            block_idx += 1

        return transported
```

---

## IV. Symmetry Breaking: Haar Initialization

### 4.1 The Cold Start Problem

If all φ_i = 0 initially, transport is trivial (Ω_ij = I) and belief alignment provides no semantic differentiation. The model gets "stuck."

### 4.2 Haar-Distributed Initialization

Initialize token gauge frames from the Haar measure on SO(N):

```python
def haar_so3_init(n_tokens, dim_g=3):
    """
    Sample φ uniformly over SO(3) via axis-angle.

    Haar measure on SO(3): uniform axis, uniform angle in [0, π]
    (with proper density correction)
    """
    # Random unit axes
    axes = torch.randn(n_tokens, 3)
    axes = F.normalize(axes, dim=-1)

    # Uniform angle in [0, π] with Haar density ∝ sin²(θ/2)
    # Use inverse CDF sampling
    u = torch.rand(n_tokens)
    # CDF of sin²(θ/2) on [0,π]: F(θ) = (θ - sin(θ))/π
    # Approximate inverse via Newton's method or lookup table
    angles = inverse_haar_cdf_so3(u)

    return axes * angles.unsqueeze(-1)


def haar_soN_init(n_tokens, N):
    """
    Sample φ uniformly over SO(N).

    Generate random orthogonal matrix via QR decomposition of Gaussian matrix.
    Then extract Lie algebra element via matrix logarithm.
    """
    dim_g = N * (N - 1) // 2

    # Random orthogonal matrices via QR
    A = torch.randn(n_tokens, N, N)
    Q, R = torch.linalg.qr(A)
    # Ensure det(Q) = +1
    Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-1)

    # Matrix logarithm to get Lie algebra element
    phi_matrix = torch.linalg.matrix_log(Q)  # Antisymmetric

    # Extract coordinates in basis
    phi = extract_lie_coords(phi_matrix, N)  # (n_tokens, dim_g)

    return phi
```

---

## V. Complete VFE with Efficient Transport

### 5.1 Transported KL Divergence

```python
def kl_transported_efficient(mu_q, sigma_blocks, phi, structure_constants,
                              irrep_structure, generators, bch_order=2, eps=1e-6):
    """
    Compute KL(q_i || Ω_ij·q_j) efficiently using BCH.

    NEVER materializes (B, N, N, K, K) tensor!

    Complexity: O(B × N² × K × dim_g) instead of O(B × N² × K³)
    """
    B, N, K = mu_q.shape

    # Step 1: BCH to get combined Lie algebra elements
    phi_ij = bch_combine(phi, phi, structure_constants, order=bch_order)  # (B, N, N, dim_g)

    # Step 2: Transport means via Rodrigues/series (NOT matrix multiply)
    mu_transported = rotate_reducible(phi_ij, mu_q, irrep_structure, generators)

    # Step 3: Transport covariance blocks
    sigma_transported_blocks = transport_covariance_blocks(
        sigma_blocks, phi_ij, irrep_structure, generators
    )

    # Step 4: Compute KL divergence
    # For block-diagonal: KL = Σ_blocks KL_block
    kl = compute_block_kl(
        mu_q, sigma_blocks,
        mu_transported, sigma_transported_blocks,
        irrep_structure, eps
    )

    return kl  # (B, N, N)


def compute_block_kl(mu_i, sigma_i_blocks, mu_j_transported, sigma_j_blocks,
                     irrep_structure, eps):
    """
    KL divergence for block-diagonal Gaussians.

    KL = Σ_blocks [ ½(log|Σ_j|/|Σ_i| + tr(Σ_j⁻¹Σ_i) + (μ_i-μ_j)^T Σ_j⁻¹ (μ_i-μ_j) - dim) ]
    """
    B, N1, N2 = mu_j_transported.shape[:3]
    kl_total = torch.zeros(B, N1, N2, device=mu_i.device)

    block_idx = 0
    for start, end, dim in irrep_structure.get_block_indices():
        mu_i_block = mu_i[..., start:end].unsqueeze(2)  # (B, N, 1, dim)
        mu_j_block = mu_j_transported[..., start:end]    # (B, N, N, dim)

        Sigma_i = sigma_i_blocks[block_idx].unsqueeze(2)  # (B, N, 1, dim, dim)
        Sigma_j = sigma_j_blocks[block_idx]                # (B, N, N, dim, dim)

        # Add eps for stability
        Sigma_j_safe = Sigma_j + eps * torch.eye(dim, device=Sigma_j.device)

        # Compute block KL
        Sigma_j_inv = torch.linalg.inv(Sigma_j_safe)

        log_det_ratio = torch.linalg.slogdet(Sigma_j_safe)[1] - torch.linalg.slogdet(Sigma_i + eps * torch.eye(dim, device=Sigma_i.device))[1]
        trace_term = torch.einsum('...ij,...ji->...', Sigma_j_inv, Sigma_i)

        mu_diff = mu_i_block - mu_j_block
        mahal_term = torch.einsum('...i,...ij,...j->...', mu_diff, Sigma_j_inv, mu_diff)

        kl_block = 0.5 * (log_det_ratio + trace_term + mahal_term - dim)
        kl_total = kl_total + kl_block

        block_idx += 1

    return kl_total
```

### 5.2 Complete VFE Computation

```python
def compute_vfe(mu_q, sigma_blocks, phi, mu_p, sigma_p_blocks,
                target_ids, token_priors, config, mask=None):
    """
    FULL Variational Free Energy with efficient BCH transport.

    F = α·Σ_i KL(q_i||p_i)
      + λ_β·Σ_ij β_ij·KL(q_i||Ω_ij·q_j)
      + λ_γ·Σ_ij γ_ij·KL(p_i||Ω_ij·p_j)
      - Σ_i log p(y_i|q_i)
    """
    B, N, K = mu_q.shape

    # Precompute BCH-combined gauge frames
    phi_ij = bch_combine(phi, phi, config.structure_constants,
                         order=config.bch_order)  # (B, N, N, dim_g)

    # 1. Self-coupling: KL(q_i || p_i) - no transport needed
    kl_self = kl_block_diagonal(mu_q, sigma_blocks, mu_p, sigma_p_blocks)
    F_self = config.alpha * kl_self.sum()

    # 2. Belief alignment with BCH transport
    kl_beliefs = kl_transported_efficient(
        mu_q, sigma_blocks, phi, config.structure_constants,
        config.irrep_structure, config.generators, config.bch_order
    )
    beta = compute_attention(kl_beliefs, config.kappa_beta, mask)
    F_belief = config.lambda_beta * (beta * kl_beliefs).sum()

    # 3. Prior coupling with BCH transport
    kl_priors = kl_transported_efficient(
        mu_p, sigma_p_blocks, phi, config.structure_constants,
        config.irrep_structure, config.generators, config.bch_order
    )
    gamma = compute_attention(kl_priors, config.kappa_gamma, mask)
    F_prior = config.lambda_gamma * (gamma * kl_priors).sum()

    # 4. Observation likelihood (with transport to token priors)
    logits = compute_output_logits(mu_q, sigma_blocks, phi, token_priors, config)
    ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size),
                               target_ids.view(-1), reduction='sum')
    F_obs = ce_loss

    F_total = F_self + F_belief + F_prior + F_obs

    return F_total, {
        'F_self': F_self.item(),
        'F_belief': F_belief.item(),
        'F_prior': F_prior.item(),
        'F_obs': F_obs.item(),
        'beta': beta.detach(),
        'gamma': gamma.detach(),
    }
```

---

## VI. Ouroboros Tower (Phase 2 Optional)

### 6.1 The Non-Markovian Memory Term

```
F_ouroboros = Σ_i Σ_d decay^d · KL(p_i || h_i^d)
```

where h_i^d is the hyperprior from ancestor depth d.

### 6.2 Implementation (Phase 2)

```python
class OuroborosTower:
    """
    Hierarchical hyperprior memory for long-range dependencies.

    Enable this AFTER core VFE is validated.
    """
    def __init__(self, config):
        self.decay = config.ouroboros_decay
        self.max_depth = config.ouroboros_depth
        self.history_buffer = []  # List of (mu_h, sigma_h) at each depth

    def compute_ouroboros_term(self, mu_p, sigma_p_blocks):
        """
        Σ_d decay^d · KL(p_i || h_i^d)
        """
        if not self.history_buffer:
            return 0.0

        F_ouro = 0.0
        for d, (mu_h, sigma_h) in enumerate(self.history_buffer):
            weight = self.decay ** d
            kl_to_hyperprior = kl_block_diagonal(mu_p, sigma_p_blocks, mu_h, sigma_h)
            F_ouro += weight * kl_to_hyperprior.sum()

        return F_ouro

    def update_history(self, mu_p, sigma_p_blocks):
        """
        Shift history and add current priors as new hyperprior.
        """
        self.history_buffer.insert(0, (mu_p.detach().clone(),
                                        [b.detach().clone() for b in sigma_p_blocks]))
        if len(self.history_buffer) > self.max_depth:
            self.history_buffer.pop()
```

**When to enable:**
- After Phase 1-4 are working
- For long-context tasks (WikiText-103, etc.)
- When the model needs "gravitational pull" from distant history

---

## VII. Complete Configuration

```python
@dataclass
class PureFEPConfig:
    # Architecture
    vocab_size: int = 256
    embed_dim: int = 64           # K (must match irrep decomposition)
    n_layers: int = 4
    max_seq_len: int = 128

    # Gauge structure
    gauge_group: str = 'SO3'      # 'SO3' or 'SON'
    N: int = 3                    # N for SO(N)
    phi_dim: int = 3              # dim(𝔤): 3 for SO(3), N(N-1)/2 for SO(N)
    bch_order: int = 2            # BCH truncation order

    # Irrep decomposition of embed_dim
    # Example for K=64 under SO(3): 10 scalars + 18 vectors = 10×1 + 18×3 = 64
    irrep_structure: IrrepStructure = field(default_factory=lambda:
        IrrepStructure([(1, 10), (3, 18)]))  # [(dim, multiplicity), ...]

    # Covariance structure
    covariance_mode: str = 'block_diagonal'  # 'diagonal' or 'block_diagonal'

    # VFE weights
    alpha: float = 0.1            # Self-coupling
    lambda_beta: float = 1.0      # Belief alignment
    lambda_gamma: float = 0.1     # Prior coupling
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

    # Initialization
    phi_init: str = 'haar'        # 'zeros', 'haar', 'uniform'

    # Ouroboros (Phase 2)
    enable_ouroboros: bool = False
    ouroboros_decay: float = 0.9
    ouroboros_depth: int = 4

    # Stability
    variance_floor: float = 1e-4
    phi_max_norm: float = 3.14159  # π radians
    eps: float = 1e-6
```

---

## VIII. Token Prior Bank (with Haar Init)

```python
class TokenPriorBank(nn.Module):
    """
    Each token v has: π_v = (μ_v, Σ_v, φ_v)

    φ_v initialized from Haar measure for symmetry breaking.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token prior means
        init_std = 1.0 / math.sqrt(config.embed_dim)
        self.mu_tokens = nn.Parameter(
            torch.randn(config.vocab_size, config.embed_dim) * init_std
        )

        # Token prior covariances (block-diagonal parameters)
        n_cov_params = self._count_cov_params(config.irrep_structure)
        self.sigma_params = nn.Parameter(
            torch.zeros(config.vocab_size, n_cov_params)
        )

        # Token gauge frames - HAAR INITIALIZED!
        if config.gauge_group == 'SO3':
            phi_init = haar_so3_init(config.vocab_size)
        else:
            phi_init = haar_soN_init(config.vocab_size, config.N)
        self.phi_tokens = nn.Parameter(phi_init)

    def encode(self, input_ids):
        """Initialize agent beliefs from token priors."""
        mu_q = self.mu_tokens[input_ids]            # (B, N, K)
        sigma_params = self.sigma_params[input_ids]  # (B, N, n_params)
        phi = self.phi_tokens[input_ids]             # (B, N, phi_dim)

        # Convert sigma_params to block covariances
        sigma_blocks = self._params_to_blocks(sigma_params)

        return mu_q, sigma_blocks, phi

    def decode(self, mu_q, sigma_blocks, phi):
        """
        Output logits via transported KL to all token priors.

        logits_v = -KL(q_i || Ω_{iv}·π_v) / τ
        """
        # Compute transport from each position to each token
        # This is expensive but necessary for output
        logits = compute_output_logits_efficient(
            mu_q, sigma_blocks, phi,
            self.mu_tokens, self._params_to_blocks(self.sigma_params), self.phi_tokens,
            self.config
        )
        return logits
```

---

## IX. Implementation Phases (Revised)

### Phase 1: Gauge Infrastructure
1. Implement SO(N) structure constants
2. Implement BCH combination (order 2-3)
3. Implement Rodrigues formula (SO3) and series expansion (SON)
4. Implement Haar initialization
5. **Test**: Verify group properties (closure, associativity)

### Phase 2: Block-Diagonal Covariance
1. Implement IrrepStructure and block indexing
2. Implement Cholesky parameterization for blocks
3. Implement block transport under rotation
4. Implement block KL divergence
5. **Test**: Verify positive-definiteness preservation

### Phase 3: Complete VFE
1. Implement all four VFE terms with efficient transport
2. Implement gradient computation via autograd
3. Implement natural gradient updates
4. **Test**: Gradient check with finite differences

### Phase 4: Token & Position Priors
1. Implement TokenPriorBank with Haar init
2. Implement PositionPriors (μ, σ only)
3. Implement encoding/decoding with transport
4. Implement P-flow updates
5. **Test**: Verify symmetry breaking

### Phase 5: Full Model & Training
1. Stack layers
2. Training loop with Q-flow + P-flow
3. WikiText-2 character-level experiments
4. Compare to standard transformer baseline

### Phase 6: Ouroboros Extension (Optional)
1. Implement OuroborosTower
2. Add to VFE computation
3. Test on long-context tasks
4. Tune decay and depth parameters

---

## X. Complexity Analysis

| Operation | Naive | With BCH + Rodrigues |
|-----------|-------|---------------------|
| Transport operators | O(B×N²×K³) | O(B×N²×K×dim_g) |
| Mean transport | O(B×N²×K²) | O(B×N²×K) |
| Covariance transport | O(B×N²×K³) | O(B×N²×Σblock_dim³) |
| Memory for Ω | O(B×N²×K²) | O(B×N²×dim_g) |

For K=64, N=1024, dim_g=3:
- Naive Ω memory: 64×1024²×64² ≈ 270GB per batch item 😱
- BCH approach: 1024²×3 ≈ 3MB per batch item ✓

---

## XI. Key Equations Summary (Final)

| Component | Equation |
|-----------|----------|
| **BCH Combination** | φ_ij = φ_i - φ_j - ½[φ_i, φ_j]_𝔤 + O(φ³) |
| **Lie Bracket** | [φ_i, φ_j]^c = Σ_{ab} φ_i^a φ_j^b f_abc |
| **Rodrigues (SO3)** | Ω·v = v cos θ + (k×v) sin θ + k(k·v)(1-cos θ) |
| **Block Covariance Transport** | Σ'_block = Ω_block · Σ_block · Ω_block^T |
| **Belief Attention** | β_ij = softmax_j(-KL(q_i \|\| Ω_ij·q_j) / κ_β) |
| **VFE** | F = α·KL(q\|\|p) + λ_β·β·KL + λ_γ·γ·KL - log p(y) [+ Ouroboros] |
| **Haar Init (SO3)** | φ ~ Uniform(S²) × HaarAngle([0,π]) |

---

*Plan v2: Proper BCH transport, SO(N) compatible, block-diagonal covariance.*
*φ encodes WHAT (semantics), priors encode WHERE (position).*
*Ouroboros tower deferred to Phase 6 after core validation.*
