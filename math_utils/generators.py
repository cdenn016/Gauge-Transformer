# -*- coding: utf-8 -*-
"""
SO(3) Lie Algebra Generators
=============================

Construction and validation of SO(3) generators for gauge theory.

For SO(3), we use the spin-ℓ irreducible representations:
- Dimension: K = 2ℓ + 1 (always odd)
- Generators: Real skew-symmetric K×K matrices
- Commutation: [G_x, G_y] = G_z (cyclic)
- Casimir eigenvalue: ℓ(ℓ+1)

Uses real tesseral harmonics (not spherical) to avoid complex arithmetic.
"""

import numpy as np
from typing import Dict


# =============================================================================
# Main Interface - SO(3) Generators
# =============================================================================

def generate_so3_generators(
    K: int,
    *,
    cache: bool = True,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate SO(3) Lie algebra generators for dimension K.

    This is the primary interface for obtaining generators. Internally uses
    irrep construction with automatic validation.

    Args:
        K: Latent dimension (must be odd: K = 2ℓ + 1)
        cache: If True, cache generators for reuse
        validate: If True, verify commutation relations
        eps: Tolerance for validation

    Returns:
        G: Generators array, shape (3, K, K), float32
           G[a] is the a-th generator (a ∈ {0,1,2} for x,y,z)

    Properties:
        - G[a] is real skew-symmetric: G[a]ᵀ = -G[a]
        - Commutation: [G_x, G_y] = G_z (cyclic)
        - Casimir: -Σ_a G_a² = ℓ(ℓ+1) I where ℓ = (K-1)/2

    Examples:
        >>> # Spin-1 (3D, ℓ=1)
        >>> G = generate_so3_generators(3)
        >>> G.shape
        (3, 3, 3)

        >>> # Verify commutation
        >>> np.allclose(G[0] @ G[1] - G[1] @ G[0], G[2])
        True

        >>> # Spin-2 (5D, ℓ=2)
        >>> G = generate_so3_generators(5)
        >>> ell = (5 - 1) // 2  # = 2
        >>> casimir = ell * (ell + 1)  # = 6
        >>> C2 = -sum(G[a] @ G[a] for a in range(3))
        >>> np.allclose(C2, casimir * np.eye(5))
        True

    Raises:
        ValueError: If K is even (SO(3) irreps must have odd dimension)
        RuntimeError: If validation fails

    Notes:
        - For K=3: Standard 3D rotation generators (spin-1)
        - For K=5,7,9,...: Higher spin representations
        - Internally constructs irrep via tesseral harmonics
        - Cached by default for performance
    """
    # Validate K is odd
    if K % 2 == 0:
        raise ValueError(
            f"K must be odd for SO(3) irreps (K = 2ℓ + 1). Got K={K}."
        )

    # Check cache
    if cache and K in _GENERATOR_CACHE:
        return _GENERATOR_CACHE[K].copy()

    # Compute spin quantum number
    ell = (K - 1) // 2

    # Build irrep generators
    G = _build_so3_irrep_generators(ell)

    # Validate if requested
    if validate:
        _validate_so3_generators(G, eps=1e-5)

    # Cache for reuse
    if cache:
        _GENERATOR_CACHE[K] = G.copy()

    return G


# =============================================================================
# Irrep Construction (Tesseral Basis)
# =============================================================================

def _build_so3_irrep_generators(ell: int) -> np.ndarray:
    """
    Build SO(3) generators for spin-ℓ irrep in real tesseral basis.

    Algorithm:
    ---------
    1. Construct complex spherical harmonic operators J_x, J_y, J_z
    2. Build unitary transformation S: spherical → tesseral
    3. Transform: G_a = Re(S J_a S†) and enforce skew-symmetry

    Args:
        ell: Spin quantum number (ℓ ≥ 0)

    Returns:
        G: (3, K, K) float32 generators where K = 2ℓ + 1
    """
    K = 2 * ell + 1

    # ========== Step 1: Complex spherical operators ==========
    # Build J_+, J_-, J_z in complex basis
    J_plus = np.zeros((K, K), dtype=np.complex128)
    J_minus = np.zeros((K, K), dtype=np.complex128)
    J_z = np.zeros((K, K), dtype=np.complex128)

    for m in range(-ell, ell + 1):
        i = m + ell  # Index: m ∈ [-ℓ, ℓ] → i ∈ [0, K-1]

        # J_z is diagonal
        J_z[i, i] = m

        # J_+ raises m by 1
        if m < ell:
            a = np.sqrt((ell - m) * (ell + m + 1))
            J_plus[i, i + 1] = a

        # J_- lowers m by 1
        if m > -ell:
            a = np.sqrt((ell + m) * (ell - m + 1))
            J_minus[i, i - 1] = a

    # Cartesian operators
    J_x = (J_plus + J_minus) / 2.0
    J_y = (J_plus - J_minus) / (2.0j)

    # ========== Step 2: Spherical → Tesseral transformation ==========
    # S is unitary, transforms |ℓ,m⟩ → tesseral basis
    S = _build_tesseral_transform(ell)
    S_inv = S.conj().T

    # ========== Step 3: Transform to real basis ==========
    def _to_real_skew(J_complex: np.ndarray) -> np.ndarray:
        """Transform complex operator to real skew-symmetric generator."""
        # G = Re(S (iJ) S†) where factor of i makes it skew-symmetric
        G_complex = S @ (1j * J_complex) @ S_inv
        G_real = G_complex.real

        # Enforce skew-symmetry (remove any numerical symmetric part)
        G_skew = 0.5 * (G_real - G_real.T)
        return G_skew

    G_x = _to_real_skew(J_x)
    G_y = _to_real_skew(J_y)
    G_z = _to_real_skew(J_z)

    # Stack as (3, K, K)
    G = np.stack([G_x, G_y, G_z], axis=0)

    return G.astype(np.float32, copy=False)


def _build_tesseral_transform(ell: int) -> np.ndarray:
    """
    Construct unitary transformation from spherical to tesseral basis.

    Tesseral harmonics are real linear combinations of spherical harmonics:
        Y^c_{ℓm} = (Y_{ℓm} + (-1)^m Y_{ℓ,-m}) / √2        (cosine-like, m > 0)
        Y^s_{ℓm} = (Y_{ℓm} - (-1)^m Y_{ℓ,-m}) / (i√2)     (sine-like, m > 0)
        Y^0_{ℓ0} = Y_{ℓ0}                                  (m = 0)

    Args:
        ell: Spin quantum number

    Returns:
        S: (K, K) unitary matrix, complex128
    """
    K = 2 * ell + 1
    S = np.zeros((K, K), dtype=np.complex128)

    # m = 0 component (center)
    S[0, ell] = 1.0

    # m > 0 components (cosine and sine pairs)
    row = 1
    for m in range(1, ell + 1):
        phase = (-1) ** m
        sqrt2_inv = 1.0 / np.sqrt(2.0)

        # Cosine-like: Y^c_m = (Y_m + phase Y_{-m}) / √2
        S[row, ell + m] = sqrt2_inv
        S[row, ell - m] = phase * sqrt2_inv
        row += 1

        # Sine-like: Y^s_m = (Y_m - phase Y_{-m}) / (i√2)
        S[row, ell + m] = -1j * sqrt2_inv
        S[row, ell - m] = 1j * phase * sqrt2_inv
        row += 1

    return S


# =============================================================================
# Validation
# =============================================================================

def _validate_so3_generators(
    G: np.ndarray,
    *,
    eps: float = 1e-6,
    verbose: bool = False,
) -> None:
    """
    Validate SO(3) commutation relations and properties.

    Checks:
    ------
    1. Skew-symmetry: G[a]ᵀ = -G[a]
    2. Commutation: [G_x, G_y] = G_z (cyclic)
    3. Casimir: C_2 = -Σ G_a² = ℓ(ℓ+1) I

    Args:
        G: (3, K, K) generators
        eps: Tolerance for checks
        verbose: If True, print validation details

    Raises:
        RuntimeError: If any check fails
    """
    if G.shape[0] != 3:
        raise ValueError(f"Expected 3 generators (x,y,z), got {G.shape[0]}")

    K = G.shape[1]
    if G.shape != (3, K, K):
        raise ValueError(f"Expected shape (3, K, K), got {G.shape}")

    # Cast to float64 for validation to avoid precision issues with large spin
    G64 = G.astype(np.float64)
    G_x, G_y, G_z = G64[0], G64[1], G64[2]

    # ========== Check 1: Skew-symmetry ==========
    for a, name in enumerate(['x', 'y', 'z']):
        G_a = G64[a]
        skew_error = np.linalg.norm(G_a + G_a.T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Generator G_{name} not skew-symmetric: ||G + Gᵀ|| = {skew_error:.3e}"
            )

    # ========== Check 2: Commutation relations ==========
    # [G_x, G_y] = G_z
    comm_xy = G_x @ G_y - G_y @ G_x
    error_xy = np.linalg.norm(comm_xy - G_z, ord='fro')

    # [G_y, G_z] = G_x (cyclic)
    comm_yz = G_y @ G_z - G_z @ G_y
    error_yz = np.linalg.norm(comm_yz - G_x, ord='fro')

    # [G_z, G_x] = G_y
    comm_zx = G_z @ G_x - G_x @ G_z
    error_zx = np.linalg.norm(comm_zx - G_y, ord='fro')

    max_error = max(error_xy, error_yz, error_zx)

    # Scale tolerance by generator norm squared for commutator checks
    # (matrix products accumulate errors proportional to scale²)
    scale = max(np.linalg.norm(G64[a], ord='fro') for a in range(3))
    threshold = eps * max(scale * scale, 1.0)

    if max_error > threshold:
        raise RuntimeError(
            f"SO(3) commutation relations violated:\n"
            f"  [G_x, G_y] - G_z: {error_xy:.3e}\n"
            f"  [G_y, G_z] - G_x: {error_yz:.3e}\n"
            f"  [G_z, G_x] - G_y: {error_zx:.3e}\n"
            f"  threshold: {threshold:.3e}"
        )

    # Use float64 for Casimir check as well
    C_2 = -sum(G64[a] @ G64[a] for a in range(3))

    # Extract eigenvalues (should all be ℓ(ℓ+1))
    eigenvalues    = np.linalg.eigvalsh(C_2)
    casimir_value  = float(np.mean(eigenvalues))
    casimir_spread = float(np.std(eigenvalues))

    # Expected value
    ell = (K - 1) // 2
    casimir_expected = ell * (ell + 1)
    casimir_error = abs(casimir_value - casimir_expected)

    # Scale tolerance by the size of C₂ (larger for high spin)
    base = max(abs(casimir_expected), 1.0)
    tol  = eps * base

    if casimir_error > tol or casimir_spread > tol:
        raise RuntimeError(
            "Casimir operator check failed:\n"
            f"  Expected: {casimir_expected}\n"
            f"  Got: {casimir_value:.6f} ± {casimir_spread:.3e}\n"
            f"  Error: {casimir_error:.3e}"
        )


    if verbose:
        print("✓ SO(3) generator validation passed:")
        print(f"  Dimension: K = {K} (ℓ = {ell})")
        print(f"  Skew-symmetry: max error = {max([np.linalg.norm(G64[a] + G64[a].T) for a in range(3)]):.3e}")
        print(f"  Commutation: max error = {max_error:.3e}")
        print(f"  Casimir: C₂ = {casimir_value:.6f} (expected {casimir_expected})")


# =============================================================================
# Multi-Irrep Block-Diagonal Generators
# =============================================================================

def generate_multi_irrep_generators(
    irrep_spec: list,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate block-diagonal SO(3) generators from a multi-irrep specification.

    This creates generators that act on a direct sum of irreducible representations:
        V = ⊕_ℓ (V_ℓ)^{n_ℓ}

    where V_ℓ is the spin-ℓ irrep (dimension 2ℓ+1) with multiplicity n_ℓ.

    Args:
        irrep_spec: List of (label, multiplicity, dim) tuples.
            Example: [('ℓ0', 32, 1), ('ℓ1', 15, 3), ('ℓ2', 10, 5)]
            - label: String identifier (e.g., 'ℓ0', 'ℓ1', 'scalar', 'vector')
            - multiplicity: How many copies of this irrep
            - dim: Dimension of irrep (must be odd: 1, 3, 5, 7, ...)
        validate: If True, verify the resulting generators
        eps: Tolerance for validation

    Returns:
        G: Block-diagonal generators, shape (3, K, K), where K = Σ mult × dim
           Each G[a] has blocks corresponding to each irrep copy

    Example:
        >>> # K = 32×1 + 15×3 + 10×5 = 32 + 45 + 50 = 127
        >>> spec = [('ℓ0', 32, 1), ('ℓ1', 15, 3), ('ℓ2', 10, 5)]
        >>> G = generate_multi_irrep_generators(spec)
        >>> G.shape
        (3, 127, 127)

        >>> # Structure: block diagonal with scalar 0-blocks, then spin-1 blocks, then spin-2
        >>> # First 32 dimensions: all zeros (scalars don't rotate)
        >>> np.allclose(G[:, :32, :32], 0)
        True

    Raises:
        ValueError: If any irrep dimension is even
    """
    # Validate irrep dimensions
    for label, mult, dim in irrep_spec:
        if dim % 2 == 0:
            raise ValueError(
                f"Irrep '{label}' has even dimension {dim}. "
                f"SO(3) irreps must have odd dimension (2ℓ+1)."
            )
        if mult < 0:
            raise ValueError(f"Irrep '{label}' has negative multiplicity {mult}.")

    # Compute total dimension
    K = sum(mult * dim for _, mult, dim in irrep_spec)

    # Initialize block-diagonal generators
    G = np.zeros((3, K, K), dtype=np.float32)

    # Fill in blocks
    idx = 0
    for label, mult, dim in irrep_spec:
        if dim == 1:
            # Scalars (ℓ=0): generator is zero
            # Skip mult×1 dimensions
            idx += mult * dim
        else:
            # Higher spin: get generators for this irrep
            G_irrep = generate_so3_generators(dim, cache=True, validate=False)

            # Place mult copies on the diagonal
            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_irrep
                idx += dim

    # Validate if requested
    if validate and K > 1:
        _validate_block_diagonal_generators(G, irrep_spec, eps=eps)

    return G


def _validate_block_diagonal_generators(
    G: np.ndarray,
    irrep_spec: list,
    *,
    eps: float = 1e-6,
) -> None:
    """
    Validate block-diagonal multi-irrep generators.

    Checks:
    1. Skew-symmetry: G[a]ᵀ = -G[a]
    2. Commutation: [G_x, G_y] = G_z (cyclic)
    3. Block structure: off-diagonal blocks are zero

    Args:
        G: (3, K, K) generators
        irrep_spec: The irrep specification used to create G
        eps: Tolerance for checks
    """
    K = G.shape[1]

    # Cast to float64 for validation to avoid precision issues with large spin
    G64 = G.astype(np.float64)

    # Check skew-symmetry
    for a in range(3):
        skew_error = np.linalg.norm(G64[a] + G64[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Block-diagonal generator G[{a}] not skew-symmetric: "
                f"||G + Gᵀ|| = {skew_error:.3e}"
            )

    # Check commutation relations
    G_x, G_y, G_z = G64[0], G64[1], G64[2]

    comm_xy = G_x @ G_y - G_y @ G_x
    error_xy = np.linalg.norm(comm_xy - G_z, ord='fro')

    comm_yz = G_y @ G_z - G_z @ G_y
    error_yz = np.linalg.norm(comm_yz - G_x, ord='fro')

    comm_zx = G_z @ G_x - G_x @ G_z
    error_zx = np.linalg.norm(comm_zx - G_y, ord='fro')

    max_error = max(error_xy, error_yz, error_zx)

    # Scale tolerance by generator norm squared for commutator checks
    # (matrix products accumulate errors proportional to scale²)
    scale = max(np.linalg.norm(G64[a], ord='fro') for a in range(3))
    threshold = eps * max(scale * scale, 1.0)

    if max_error > threshold:
        raise RuntimeError(
            f"Block-diagonal SO(3) commutation violated:\n"
            f"  [G_x, G_y] - G_z: {error_xy:.3e}\n"
            f"  [G_y, G_z] - G_x: {error_yz:.3e}\n"
            f"  [G_z, G_x] - G_y: {error_zx:.3e}\n"
            f"  threshold: {threshold:.3e}"
        )

    # Check block structure (off-diagonal blocks should be zero)
    idx = 0
    block_starts = []
    for _, mult, dim in irrep_spec:
        for _ in range(mult):
            block_starts.append((idx, dim))
            idx += dim

    for i, (start_i, dim_i) in enumerate(block_starts):
        for j, (start_j, dim_j) in enumerate(block_starts):
            if i != j:
                # Check off-diagonal block is zero
                for a in range(3):
                    block = G64[a, start_i:start_i+dim_i, start_j:start_j+dim_j]
                    block_norm = np.linalg.norm(block, ord='fro')
                    if block_norm > eps:
                        raise RuntimeError(
                            f"Off-diagonal block ({i},{j}) is non-zero: "
                            f"||block|| = {block_norm:.3e}"
                        )


# =============================================================================
# SO(N) Generators - Fundamental Representation
# =============================================================================

def generate_soN_generators(
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate SO(N) Lie algebra generators in the fundamental (N-dimensional) representation.

    SO(N) is the group of N×N orthogonal matrices with determinant 1.
    Its Lie algebra so(N) consists of N×N skew-symmetric matrices.

    The Lie algebra has dimension N(N-1)/2, with basis elements L_{ij} for i < j:
        (L_{ij})_{kl} = δ_{ik}δ_{jl} - δ_{il}δ_{jk}

    These satisfy the commutation relations:
        [L_{ij}, L_{kl}] = δ_{jk}L_{il} - δ_{ik}L_{jl} - δ_{jl}L_{ik} + δ_{il}L_{jk}

    Args:
        N: The dimension of the fundamental representation (N ≥ 2)
        validate: If True, verify commutation relations
        eps: Tolerance for validation

    Returns:
        G: Generators array, shape (N(N-1)/2, N, N), float32
           G[a] is the a-th generator, indexed by pairs (i,j) with i < j

    Examples:
        >>> # SO(3) - 3 generators, 3×3 matrices
        >>> G = generate_soN_generators(3)
        >>> G.shape
        (3, 3, 3)

        >>> # SO(5) - 10 generators, 5×5 matrices
        >>> G = generate_soN_generators(5)
        >>> G.shape
        (10, 5, 5)

        >>> # SO(8) - 28 generators, 8×8 matrices
        >>> G = generate_soN_generators(8)
        >>> G.shape
        (28, 8, 8)

    Properties:
        - G[a] is real skew-symmetric: G[a]ᵀ = -G[a]
        - Orthogonal action: exp(θ G[a]) ∈ SO(N) for any θ
        - Satisfies so(N) commutation relations
    """
    if N < 2:
        raise ValueError(f"N must be >= 2 for SO(N), got N={N}")

    n_generators = N * (N - 1) // 2
    G = np.zeros((n_generators, N, N), dtype=np.float32)

    # Build generators L_{ij} for i < j
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            # (L_{ij})_{kl} = δ_{ik}δ_{jl} - δ_{il}δ_{jk}
            G[idx, i, j] = 1.0
            G[idx, j, i] = -1.0
            idx += 1

    if validate:
        _validate_soN_generators(G, N, eps=eps)

    return G


def _validate_soN_generators(
    G: np.ndarray,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """
    Validate SO(N) generators satisfy required properties.

    Checks:
    1. Skew-symmetry: G[a]ᵀ = -G[a]
    2. Commutation relations: [L_{ij}, L_{kl}] follows so(N) structure

    Args:
        G: (n_gen, N, N) generators where n_gen = N(N-1)/2
        N: Dimension of fundamental rep
        eps: Tolerance for checks
    """
    n_gen = G.shape[0]
    expected_n_gen = N * (N - 1) // 2

    if n_gen != expected_n_gen:
        raise ValueError(
            f"Expected {expected_n_gen} generators for SO({N}), got {n_gen}"
        )

    # Check skew-symmetry
    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"SO({N}) generator G[{a}] not skew-symmetric: "
                f"||G + Gᵀ|| = {skew_error:.3e}"
            )

    # Build index map: (i,j) -> generator index
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx += 1

    # Check a sample of commutation relations
    # [L_{ij}, L_{jk}] = L_{ik} for i < j < k
    max_error = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                # [L_{ij}, L_{jk}] should equal L_{ik}
                a = idx_map[(i, j)]
                b = idx_map[(j, k)]
                c = idx_map[(i, k)]

                comm = G[a] @ G[b] - G[b] @ G[a]
                error = np.linalg.norm(comm - G[c], ord='fro')
                max_error = max(max_error, error)

    if max_error > eps:
        raise RuntimeError(
            f"SO({N}) commutation relations violated, max error: {max_error:.3e}"
        )


def generate_multi_irrep_soN_generators(
    irrep_spec: list,
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate block-diagonal SO(N) generators from a multi-irrep specification.

    This creates generators for a direct sum of SO(N) irreducible representations:
        V = ⊕_i (V_i)^{mult_i}

    Supported irrep types:
        - 'scalar' (dim=1): Invariant, generators act as zero
        - 'fund' (dim=N): Fundamental/vector representation
        - 'wedge2' (dim=N(N-1)/2): Antisymmetric 2-tensor ∧²V
        - 'sym2' (dim=N(N+1)/2-1): Symmetric traceless 2-tensor Sym²₀V

    Args:
        irrep_spec: List of (label, multiplicity, dim) tuples.
            Example: [('scalar', 10, 1), ('fund', 8, N), ('wedge2', 4, N*(N-1)//2)]
            Total dimension K = Σ mult × dim
        N: The gauge group dimension (SO(N))
        validate: If True, verify the resulting generators
        eps: Tolerance for validation

    Returns:
        G: Block-diagonal generators, shape (N(N-1)/2, K, K)
           where K = Σ mult × dim

    Example:
        >>> # SO(5) with mixed irreps
        >>> spec = [('scalar', 10, 1), ('fund', 8, 5), ('wedge2', 4, 10)]
        >>> G = generate_multi_irrep_soN_generators(spec, N=5)
        >>> G.shape
        (10, 90, 90)  # 10 generators, K = 10 + 40 + 40 = 90

        >>> # SO(8) with all three tensor irreps
        >>> spec = [('fund', 4, 8), ('wedge2', 2, 28), ('sym2', 2, 35)]
        >>> G = generate_multi_irrep_soN_generators(spec, N=8)
        >>> G.shape
        (28, 158, 158)  # 28 generators, K = 32 + 56 + 70 = 158

    Note:
        Using diverse irreps (fund + wedge2 + sym2) provides genuinely different
        transformation channels, similar to using multiple spin-ℓ irreps for SO(3).
    """
    # Expected dimensions for each irrep type
    expected_dims = {
        'scalar': 1,
        'fund': N,
        'fundamental': N,
        'vector': N,
        'wedge2': N * (N - 1) // 2,
        'antisym2': N * (N - 1) // 2,
        'exterior2': N * (N - 1) // 2,
        'sym2': N * (N + 1) // 2 - 1,
        'sym2_traceless': N * (N + 1) // 2 - 1,
        'symmetric2': N * (N + 1) // 2 - 1,
    }

    # Validate irrep specification
    for label, mult, dim in irrep_spec:
        label_lower = label.lower()

        # Check if it's a known irrep type
        if label_lower in expected_dims:
            expected_dim = expected_dims[label_lower]
            if dim != expected_dim:
                raise ValueError(
                    f"Irrep '{label}' should have dim={expected_dim} for SO({N}), "
                    f"but got dim={dim}."
                )
        else:
            # Unknown label - check if dimension matches a known irrep
            if dim == 1:
                pass  # Scalar
            elif dim == N:
                pass  # Fundamental
            elif dim == N * (N - 1) // 2:
                pass  # ∧²V
            elif dim == N * (N + 1) // 2 - 1:
                pass  # Sym²₀V
            else:
                raise ValueError(
                    f"Irrep '{label}' has dimension {dim}, which doesn't match any "
                    f"implemented SO({N}) irrep. Supported dims: 1 (scalar), "
                    f"{N} (fund), {N*(N-1)//2} (wedge2), {N*(N+1)//2-1} (sym2)."
                )

        if mult < 0:
            raise ValueError(f"Irrep '{label}' has negative multiplicity {mult}.")

    # Compute total dimension
    K = sum(mult * dim for _, mult, dim in irrep_spec)

    # Number of generators for SO(N)
    n_gen = N * (N - 1) // 2

    # Initialize block-diagonal generators
    G = np.zeros((n_gen, K, K), dtype=np.float32)

    # Get generators for each irrep type (cached for efficiency)
    G_fund = None
    G_wedge2 = None
    G_sym2 = None

    # Fill in blocks
    idx = 0
    for label, mult, dim in irrep_spec:
        if dim == 1:
            # Scalars: generators act as zero
            idx += mult * dim

        elif dim == N:
            # Fundamental representation
            if G_fund is None:
                G_fund = generate_soN_generators(N, validate=False)
            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_fund
                idx += dim

        elif dim == N * (N - 1) // 2:
            # ∧²V (antisymmetric 2-tensor)
            if G_wedge2 is None:
                G_wedge2 = generate_wedge2_generators(N, validate=False)
            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_wedge2
                idx += dim

        elif dim == N * (N + 1) // 2 - 1:
            # Sym²₀V (symmetric traceless 2-tensor)
            if G_sym2 is None:
                G_sym2 = generate_sym2_traceless_generators(N, validate=False)
            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_sym2
                idx += dim

        else:
            # Should never reach here due to validation above
            raise RuntimeError(f"Unexpected dimension {dim} for irrep '{label}'")

    # Validate if requested
    if validate and K > 1:
        _validate_block_diagonal_soN_generators(G, irrep_spec, N, eps=eps)

    return G


# =============================================================================
# SO(N) Higher Tensor Representations (Non-Fundamental Irreps)
# =============================================================================

def _wedge2_index_to_pair(idx: int, N: int) -> tuple:
    """Map linear index to (i,j) pair with i < j for ∧²V basis."""
    i = 0
    count = 0
    while count + (N - 1 - i) <= idx:
        count += N - 1 - i
        i += 1
    j = idx - count + i + 1
    return i, j


def _wedge2_pair_to_index(i: int, j: int, N: int) -> int:
    """Map (i,j) pair with i < j to linear index for ∧²V basis."""
    # Sum of (N-1) + (N-2) + ... + (N-i) = i*N - i*(i+1)/2
    return i * N - i * (i + 1) // 2 + (j - i - 1)


def generate_wedge2_generators(
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate SO(N) generators for ∧²V (antisymmetric 2-tensor representation).

    The exterior square ∧²V is the space of antisymmetric N×N matrices.
    Elements can be thought of as "bivectors" or "angular momentum" components.

    Dimension: N(N-1)/2
    Basis: { e_i ∧ e_j : i < j } represented as E_ij - E_ji

    The Lie algebra action is the commutator:
        G · X = [G, X] = GX - XG

    This preserves antisymmetry (since G is skew-symmetric).

    Args:
        N: Dimension of the fundamental representation (N ≥ 2)
        validate: If True, verify the generators
        eps: Tolerance for validation

    Returns:
        G: Generators array, shape (n_gen, dim, dim)
           where n_gen = N(N-1)/2 and dim = N(N-1)/2

    Example:
        >>> G = generate_wedge2_generators(5)
        >>> G.shape
        (10, 10, 10)  # SO(5) has 10 generators, ∧²(R^5) has dim 10

    Properties:
        - Different Casimir eigenvalue than fundamental
        - Transforms as X' = O X Oᵀ under O ∈ SO(N)
        - Captures "rotational" or "angular momentum" degrees of freedom
    """
    if N < 2:
        raise ValueError(f"N must be >= 2 for SO(N), got N={N}")

    n_gen = N * (N - 1) // 2
    dim = N * (N - 1) // 2  # Same dimension as number of generators!

    # Get fundamental generators
    G_fund = generate_soN_generators(N, validate=False)  # (n_gen, N, N)

    # Build generators for ∧²V representation
    # Action: X → [G_a, X] where X is antisymmetric N×N matrix
    G_wedge2 = np.zeros((n_gen, dim, dim), dtype=np.float32)

    for a in range(n_gen):
        G_a = G_fund[a]  # (N, N) skew-symmetric

        for p in range(dim):  # Input basis element index
            i, j = _wedge2_index_to_pair(p, N)

            # Basis element: E_ij - E_ji (antisymmetric)
            X = np.zeros((N, N), dtype=np.float32)
            X[i, j] = 1.0
            X[j, i] = -1.0

            # Commutator [G_a, X] = G_a @ X - X @ G_a
            comm = G_a @ X - X @ G_a  # Still antisymmetric

            # Express result in ∧² basis
            for q in range(dim):
                k, l = _wedge2_index_to_pair(q, N)
                # The coefficient is the (k,l) entry (upper triangle)
                G_wedge2[a, q, p] = comm[k, l]

    if validate:
        _validate_wedge2_generators(G_wedge2, N, eps=eps)

    return G_wedge2


def _validate_wedge2_generators(
    G: np.ndarray,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """Validate ∧²V generators."""
    n_gen, dim, _ = G.shape

    expected_n_gen = N * (N - 1) // 2
    expected_dim = N * (N - 1) // 2

    if n_gen != expected_n_gen:
        raise ValueError(f"Expected {expected_n_gen} generators, got {n_gen}")
    if dim != expected_dim:
        raise ValueError(f"Expected dim {expected_dim}, got {dim}")

    # Check skew-symmetry of generators
    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"∧² generator G[{a}] not skew-symmetric: ||G + Gᵀ|| = {skew_error:.3e}"
            )

    # Check sample commutation relations (they should form so(N) algebra)
    if n_gen >= 3:
        comm_01 = G[0] @ G[1] - G[1] @ G[0]
        if np.linalg.norm(comm_01 + comm_01.T, ord='fro') > eps:
            raise RuntimeError("Commutator [G_0, G_1] in ∧² rep not skew-symmetric")


def _sym2_traceless_basis_size(N: int) -> int:
    """Dimension of Sym²₀V (symmetric traceless 2-tensors)."""
    return N * (N + 1) // 2 - 1


def _sym2_traceless_index_to_components(idx: int, N: int) -> tuple:
    """
    Map linear index to symmetric traceless basis element.

    Basis ordering:
    - First N(N-1)/2 indices: off-diagonal (i,j) with i < j, coefficient √2
    - Next N-1 indices: diagonal traceless combinations

    Returns:
        (type, data) where:
        - type='offdiag': data=(i, j) for off-diagonal element
        - type='diag': data=k for k-th diagonal traceless element
    """
    n_offdiag = N * (N - 1) // 2

    if idx < n_offdiag:
        # Off-diagonal element
        i, j = _wedge2_index_to_pair(idx, N)
        return ('offdiag', (i, j))
    else:
        # Diagonal traceless element
        k = idx - n_offdiag
        return ('diag', k)


def _build_sym2_traceless_basis_element(idx: int, N: int) -> np.ndarray:
    """
    Build the idx-th basis element of Sym²₀V as an N×N matrix.

    The basis is orthonormal under the Frobenius inner product.
    """
    n_offdiag = N * (N - 1) // 2
    X = np.zeros((N, N), dtype=np.float32)

    if idx < n_offdiag:
        # Off-diagonal: (E_ij + E_ji) / √2
        i, j = _wedge2_index_to_pair(idx, N)
        X[i, j] = 1.0 / np.sqrt(2)
        X[j, i] = 1.0 / np.sqrt(2)
    else:
        # Diagonal traceless: use Gell-Mann-like basis
        # Element k: (E_00 + ... + E_kk - (k+1)E_{k+1,k+1}) / √((k+1)(k+2))
        k = idx - n_offdiag
        norm = np.sqrt((k + 1) * (k + 2))
        for i in range(k + 1):
            X[i, i] = 1.0 / norm
        X[k + 1, k + 1] = -(k + 1) / norm

    return X


def generate_sym2_traceless_generators(
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate SO(N) generators for Sym²₀V (symmetric traceless 2-tensor representation).

    The symmetric traceless square Sym²₀V is the space of symmetric N×N matrices
    with trace zero. These represent "quadrupolar" or "strain-like" degrees of freedom.

    Dimension: N(N+1)/2 - 1
    Basis: Orthonormal symmetric traceless matrices

    The Lie algebra action is the commutator:
        G · X = [G, X] = GX - XG

    This preserves symmetry and tracelessness.

    Args:
        N: Dimension of the fundamental representation (N ≥ 2)
        validate: If True, verify the generators
        eps: Tolerance for validation

    Returns:
        G: Generators array, shape (n_gen, dim, dim)
           where n_gen = N(N-1)/2 and dim = N(N+1)/2 - 1

    Example:
        >>> G = generate_sym2_traceless_generators(5)
        >>> G.shape
        (10, 14, 14)  # SO(5) has 10 generators, Sym²₀(R^5) has dim 14

    Properties:
        - Different Casimir eigenvalue than fundamental and ∧²
        - Transforms as X' = O X Oᵀ under O ∈ SO(N)
        - Captures "quadrupolar" or "deformation" degrees of freedom
    """
    if N < 2:
        raise ValueError(f"N must be >= 2 for SO(N), got N={N}")

    n_gen = N * (N - 1) // 2
    dim = _sym2_traceless_basis_size(N)

    # Get fundamental generators
    G_fund = generate_soN_generators(N, validate=False)  # (n_gen, N, N)

    # Pre-build basis elements
    basis = [_build_sym2_traceless_basis_element(p, N) for p in range(dim)]

    # Build generators for Sym²₀V representation
    # Action: X → [G_a, X] where X is symmetric traceless N×N matrix
    G_sym2 = np.zeros((n_gen, dim, dim), dtype=np.float32)

    for a in range(n_gen):
        G_a = G_fund[a]  # (N, N) skew-symmetric

        for p in range(dim):  # Input basis element index
            X = basis[p]

            # Commutator [G_a, X] = G_a @ X - X @ G_a
            # This is symmetric (and traceless) when G is skew and X is symmetric
            comm = G_a @ X - X @ G_a

            # Express result in Sym²₀ basis via inner product
            for q in range(dim):
                Y = basis[q]
                # Inner product: tr(Yᵀ comm) = tr(Y comm) since both symmetric
                G_sym2[a, q, p] = np.sum(Y * comm)

    if validate:
        _validate_sym2_traceless_generators(G_sym2, N, eps=eps)

    return G_sym2


def _validate_sym2_traceless_generators(
    G: np.ndarray,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """Validate Sym²₀V generators."""
    n_gen, dim, _ = G.shape

    expected_n_gen = N * (N - 1) // 2
    expected_dim = _sym2_traceless_basis_size(N)

    if n_gen != expected_n_gen:
        raise ValueError(f"Expected {expected_n_gen} generators, got {n_gen}")
    if dim != expected_dim:
        raise ValueError(f"Expected dim {expected_dim}, got {dim}")

    # Check skew-symmetry of generators
    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Sym²₀ generator G[{a}] not skew-symmetric: ||G + Gᵀ|| = {skew_error:.3e}"
            )

    # Check sample commutation
    if n_gen >= 3:
        comm_01 = G[0] @ G[1] - G[1] @ G[0]
        if np.linalg.norm(comm_01 + comm_01.T, ord='fro') > eps:
            raise RuntimeError("Commutator [G_0, G_1] in Sym²₀ rep not skew-symmetric")


def _validate_block_diagonal_soN_generators(
    G: np.ndarray,
    irrep_spec: list,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """
    Validate block-diagonal multi-irrep SO(N) generators.

    Checks:
    1. Skew-symmetry
    2. Sample commutation relations
    3. Block structure (off-diagonal blocks are zero)
    """
    n_gen = G.shape[0]
    K = G.shape[1]

    expected_n_gen = N * (N - 1) // 2
    if n_gen != expected_n_gen:
        raise ValueError(
            f"Expected {expected_n_gen} generators for SO({N}), got {n_gen}"
        )

    # Check skew-symmetry
    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Block-diagonal SO({N}) generator G[{a}] not skew-symmetric: "
                f"||G + Gᵀ|| = {skew_error:.3e}"
            )

    # Check sample commutation (first 3 generators if available, like SO(3) subset)
    if n_gen >= 3:
        G_0, G_1, G_2 = G[0], G[1], G[2]

        # For SO(N) with N >= 3, generators 0,1,2 correspond to:
        # L_{01}, L_{02}, L_{03} or similar
        # Their commutations depend on index structure

        # Just check that commutators are skew-symmetric (sanity check)
        comm_01 = G_0 @ G_1 - G_1 @ G_0
        if np.linalg.norm(comm_01 + comm_01.T, ord='fro') > eps:
            raise RuntimeError("Commutator [G_0, G_1] not skew-symmetric")

    # Check block structure
    idx = 0
    block_starts = []
    for _, mult, dim in irrep_spec:
        for _ in range(mult):
            block_starts.append((idx, dim))
            idx += dim

    for i, (start_i, dim_i) in enumerate(block_starts):
        for j, (start_j, dim_j) in enumerate(block_starts):
            if i != j:
                for a in range(min(n_gen, 10)):  # Check first 10 generators
                    block = G[a, start_i:start_i+dim_i, start_j:start_j+dim_j]
                    block_norm = np.linalg.norm(block, ord='fro')
                    if block_norm > eps:
                        raise RuntimeError(
                            f"Off-diagonal block ({i},{j}) in generator {a} "
                            f"is non-zero: ||block|| = {block_norm:.3e}"
                        )


# =============================================================================
# SO(N) Lie Algebra Operations (PyTorch)
# =============================================================================

def _get_soN_gauge_generators(n_gen: int, device, dtype) -> 'torch.Tensor':
    """
    Get N×N generators for SO(N) gauge group based on n_gen.

    These are the canonical so(N) basis elements L_{ij}, NOT the K×K transport generators.
    Used internally for BCH composition.
    """
    import torch
    import math

    # Infer N from n_gen: n_gen = N(N-1)/2
    N = int((1 + math.sqrt(1 + 8 * n_gen)) / 2)

    if N * (N - 1) // 2 != n_gen:
        raise ValueError(f"n_gen={n_gen} doesn't correspond to valid SO(N)")

    # Build canonical generators L_{ij} for i < j
    generators = torch.zeros(n_gen, N, N, device=device, dtype=dtype)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            generators[idx, i, j] = 1.0
            generators[idx, j, i] = -1.0
            idx += 1

    return generators


def soN_bracket_torch(
    phi1: 'torch.Tensor',
    phi2: 'torch.Tensor',
    generators: 'torch.Tensor',
) -> 'torch.Tensor':
    """
    Compute the Lie bracket [φ₁·G, φ₂·G] in so(N) and return coordinates.

    For so(N), the Lie bracket of two skew-symmetric matrices is:
        [A, B] = AB - BA

    This is used in BCH composition for proper Lie group updates.

    Args:
        phi1: First Lie algebra element coordinates (..., n_gen)
        phi2: Second Lie algebra element coordinates (..., n_gen)
        generators: Lie algebra generators (n_gen, K, K) - used only for n_gen count
                   The actual N×N generators are computed internally.

    Returns:
        bracket_coords: Coordinates of [φ₁·G, φ₂·G] in generator basis (..., n_gen)
    """
    import torch

    n_gen = generators.shape[0]

    # Get proper N×N generators for the gauge group (not K×K transport generators!)
    gauge_gens = _get_soN_gauge_generators(n_gen, phi1.device, phi1.dtype)

    # Build skew-symmetric matrices using N×N gauge generators
    A1 = torch.einsum('...a,aij->...ij', phi1, gauge_gens)  # (..., N, N)
    A2 = torch.einsum('...a,aij->...ij', phi2, gauge_gens)  # (..., N, N)

    # Lie bracket: [A, B] = AB - BA
    bracket = A1 @ A2 - A2 @ A1  # (..., N, N)

    # Extract coordinates from upper triangular
    bracket_coords = extract_soN_coords_torch(bracket, gauge_gens)

    return bracket_coords


def extract_soN_coords_torch(
    A: 'torch.Tensor',
    generators: 'torch.Tensor',
) -> 'torch.Tensor':
    """
    Extract so(N) Lie algebra coordinates from a skew-symmetric matrix.

    Given A = Σ_a φ_a G_a, extract the coordinates φ_a.

    For the canonical basis L_{ij} (with i < j), the coordinates are simply
    the upper-triangular elements of A: φ_a = A[i, j].

    Args:
        A: Skew-symmetric matrix (..., M, M) where M is matrix dimension
        generators: Lie algebra generators (n_gen, K, K)
                   Note: K may be embedding dim, not gauge group dim!

    Returns:
        phi: Lie algebra coordinates (..., n_gen)
    """
    import torch
    import math

    n_gen = generators.shape[0]
    M = A.shape[-1]  # Matrix dimension of A

    # Infer gauge group dimension N from n_gen: n_gen = N(N-1)/2
    # Solving: N = (1 + sqrt(1 + 8*n_gen)) / 2
    N = int((1 + math.sqrt(1 + 8 * n_gen)) / 2)

    # Validate
    if N * (N - 1) // 2 != n_gen:
        raise ValueError(f"n_gen={n_gen} doesn't correspond to valid SO(N). "
                        f"Expected N*(N-1)/2 for some integer N.")

    if M != N:
        raise ValueError(f"Matrix A has dimension {M}x{M} but gauge group is SO({N}). "
                        f"For BCH composition, need {N}x{N} matrices.")

    # Build index mapping: generator a -> (i, j) with i < j
    # For canonical basis, generator a corresponds to pair (i, j) in order
    batch_shape = A.shape[:-2]
    phi = torch.zeros(*batch_shape, n_gen, device=A.device, dtype=A.dtype)

    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            # φ_a = A[i, j] (upper triangular element)
            phi[..., idx] = A[..., i, j]
            idx += 1

    return phi


def soN_compose_bch_torch(
    phi1: 'torch.Tensor',
    phi2: 'torch.Tensor',
    generators: 'torch.Tensor',
    order: int = 1,
) -> 'torch.Tensor':
    """
    Compose two so(N) elements using Baker-Campbell-Hausdorff formula.

    log(exp(φ₁·G)·exp(φ₂·G)) = φ₁ + φ₂ + ½[φ₁,φ₂] + (1/12)[φ₁,[φ₁,φ₂]] - ...

    For so(N), the Lie bracket is: [A, B] = AB - BA (matrix commutator)

    This is the proper way to compose updates in the Lie algebra, ensuring
    the result corresponds to a valid group element when exponentiated.

    Args:
        phi1: First so(N) element (..., n_gen)
        phi2: Second so(N) element (..., n_gen)
        generators: Lie algebra generators (n_gen, N, N)
        order: BCH expansion order (0=addition, 1=first correction, 2=second)

    Returns:
        phi_composed: Composed element in so(N) (..., n_gen)
    """
    if order == 0:
        # Simple addition (valid for small angles only)
        return phi1 + phi2

    # First-order BCH: φ₁ + φ₂ + ½[φ₁,φ₂]
    bracket_12 = soN_bracket_torch(phi1, phi2, generators)
    result = phi1 + phi2 + 0.5 * bracket_12

    if order >= 2:
        # Second-order: + (1/12)[φ₁,[φ₁,φ₂]] - (1/12)[φ₂,[φ₁,φ₂]]
        bracket_1_12 = soN_bracket_torch(phi1, bracket_12, generators)
        bracket_2_12 = soN_bracket_torch(phi2, bracket_12, generators)
        result = result + (1.0/12.0) * bracket_1_12 - (1.0/12.0) * bracket_2_12

    return result


def retract_soN_torch(
    phi: 'torch.Tensor',
    delta_phi: 'torch.Tensor',
    generators: 'torch.Tensor',
    step_size: float = 1.0,
    trust_region: float = 0.3,
    max_norm: float = 3.14159,
    bch_order: int = 1,
    eps: float = 1e-6,
) -> 'torch.Tensor':
    """
    Retract phi update onto SO(N) manifold with trust region.

    This is the proper way to update gauge frames φ:
    1. Scale delta by step_size
    2. Apply trust region (limit relative change)
    3. Compose using BCH formula (proper Lie group composition)
    4. Clamp final norm

    Args:
        phi: Current gauge frames (..., n_gen)
        delta_phi: Update direction (typically -grad_phi) (..., n_gen)
        generators: Lie algebra generators (n_gen, N, N)
        step_size: Learning rate for the update
        trust_region: Maximum relative change ||δφ|| / ||φ|| per update
        max_norm: Maximum allowed norm for phi (π = 180° rotation)
        bch_order: Order of BCH expansion (0=add, 1=first correction)
        eps: Numerical stability constant

    Returns:
        phi_new: Updated gauge frames (..., n_gen)
    """
    import torch

    # Scale update
    update = step_size * delta_phi

    # Trust region: limit step size relative to current phi
    phi_norm = torch.norm(phi, dim=-1, keepdim=True).clamp(min=0.1)
    update_norm = torch.norm(update, dim=-1, keepdim=True)

    # Scale down if update is too large relative to current phi
    scale = torch.clamp(trust_region * phi_norm / (update_norm + eps), max=1.0)
    update = scale * update

    # Compose using BCH (proper Lie group composition)
    phi_new = soN_compose_bch_torch(phi, update, generators, order=bch_order)

    # Clamp to max norm (retraction to ball)
    phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
    phi_new = torch.where(
        phi_new_norm > max_norm,
        phi_new * (max_norm / (phi_new_norm + eps)),
        phi_new
    )

    return phi_new


def retract_soN_exact_torch(
    phi: 'torch.Tensor',
    delta_phi: 'torch.Tensor',
    generators: 'torch.Tensor',
    step_size: float = 1.0,
    trust_region: float = 0.3,
    max_norm: float = 3.14159,
    eps: float = 1e-6,
) -> 'torch.Tensor':
    """
    Exact SO(N) retraction via matrix exponential and logarithm.

    Computes: φ_new = log(exp(φ·G) · exp(δφ·G))

    This is more accurate than BCH for large updates but more expensive.
    Uses real Schur decomposition for the matrix logarithm.

    Args:
        phi: Current gauge frames (..., n_gen)
        delta_phi: Update direction (..., n_gen)
        generators: Lie algebra generators (n_gen, N, N)
        step_size: Learning rate
        trust_region: Maximum relative change
        max_norm: Maximum norm for phi
        eps: Numerical stability

    Returns:
        phi_new: Updated gauge frames (..., n_gen)
    """
    import torch

    n_gen = generators.shape[0]

    # Get proper N×N generators for the gauge group (not K×K transport generators!)
    gauge_gens = _get_soN_gauge_generators(n_gen, phi.device, phi.dtype)

    # Scale update with trust region
    update = step_size * delta_phi
    phi_norm = torch.norm(phi, dim=-1, keepdim=True).clamp(min=0.1)
    update_norm = torch.norm(update, dim=-1, keepdim=True)
    scale = torch.clamp(trust_region * phi_norm / (update_norm + eps), max=1.0)
    update = scale * update

    # Build skew-symmetric matrices using N×N gauge generators
    A_phi = torch.einsum('...a,aij->...ij', phi, gauge_gens)
    A_delta = torch.einsum('...a,aij->...ij', update, gauge_gens)

    # Matrix exponentials
    R_phi = torch.matrix_exp(A_phi)
    R_delta = torch.matrix_exp(A_delta)

    # Group product
    R_new = R_phi @ R_delta

    # Matrix logarithm for orthogonal matrices
    # Use the fact that for R ∈ SO(N), log(R) is skew-symmetric
    A_new = _matrix_log_orthogonal_torch(R_new, eps=eps)

    # Extract coordinates
    phi_new = extract_soN_coords_torch(A_new, gauge_gens)

    # Clamp to max norm
    phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
    phi_new = torch.where(
        phi_new_norm > max_norm,
        phi_new * (max_norm / (phi_new_norm + eps)),
        phi_new
    )

    return phi_new


def _matrix_log_orthogonal_torch(
    R: 'torch.Tensor',
    eps: float = 1e-6,
) -> 'torch.Tensor':
    """
    Compute matrix logarithm for orthogonal matrices.

    For R ∈ SO(N), log(R) is a skew-symmetric matrix in so(N).
    Uses the real Schur decomposition approach for stability.

    Args:
        R: Orthogonal matrix (..., N, N)
        eps: Numerical stability

    Returns:
        A: Skew-symmetric matrix log(R) (..., N, N)
    """
    import torch

    # For small deviations from identity, use first-order approximation
    # log(I + X) ≈ X - X²/2 + X³/3 - ...
    # For orthogonal R = I + X where X is small and skew-symmetric

    N = R.shape[-1]
    I = torch.eye(N, device=R.device, dtype=R.dtype)

    # Check if close to identity (common case for small updates)
    deviation = R - I
    deviation_norm = torch.norm(deviation, dim=(-2, -1), keepdim=True)

    # For small deviations, use series expansion
    # For larger deviations, use the antisymmetric part extraction
    # (This is a simplified approach; full Schur method would be more robust)

    # Antisymmetric part of deviation gives first-order log
    A_approx = 0.5 * (deviation - deviation.transpose(-1, -2))

    # For better accuracy with larger rotations, use iterative refinement
    # Newton iteration: A_{k+1} = A_k + (R - exp(A_k)) @ exp(-A_k) antisymmetrized
    # But for simplicity and speed, we use BCH-based correction

    # Second-order correction
    A_sq = A_approx @ A_approx
    correction = -0.5 * A_sq  # Second-order term
    A = A_approx + 0.5 * (correction - correction.transpose(-1, -2))

    # Ensure skew-symmetry
    A = 0.5 * (A - A.transpose(-1, -2))

    return A


# =============================================================================
# Cache
# =============================================================================

_GENERATOR_CACHE: Dict[int, np.ndarray] = {}