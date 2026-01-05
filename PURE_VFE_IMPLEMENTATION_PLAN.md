# Pure VFE Implementation: Agent as Bundle Section

## Theoretical Foundation

Each agent (token position) is a **section of an associated bundle** to a principal G-bundle with statistical fibers.

Each agent has local coordinates:
- **q** (belief) - coordinates on statistical fiber
- **p** (prior) - reference distribution on fiber
- **φ** (gauge frame) - connection on principal bundle

ALL evolve via VFE gradient descent:
```
dq/dt = -∂F/∂q
dp/dt = -∂F/∂p
dφ/dt = -∂F/∂φ
```

## Full Variational Free Energy

```
F = Σ_i α·KL(q_i||p_i)                      [Self-coupling]
  + Σ_ij β_ij·KL(q_i||Ω_ij·q_j)            [Belief alignment]
  + Σ_ij γ_ij·KL(p_i||Ω_ij·p_j)            [Prior coupling - CRITICAL!]
  + Σ_i E_q_i[-log p(o_i|c)]                [Observation at base point c]
  + Σ_i KL(p_i||h_i)                        [Hyperprior - optional]
```

Where:
- Ω_ij = exp(φ_i)·exp(-φ_j) depends on gauge frames
- β_ij, γ_ij from attention weights
- c is base manifold point (for token priors)

## Current vs Principled Implementation

### ❌ Current (WRONG):
```python
# Position "encoded" from position index
self.gauge_position = GaugePositionEncoder(...)
phi = self.gauge_position(positions)  # Lookup/function!

# phi initialized to zeros each forward pass
phi = torch.zeros(B, N, phi_dim)

# Prior coupling DISABLED by default
prior_coupling_enabled: bool = False
```

**Problems:**
1. φ is "encoded" not learned
2. φ re-initialized every forward (not persistent)
3. Prior coupling disabled (no ∂F/∂φ from priors!)
4. Position structure IMPOSED not EMERGENT

### ✅ Principled (CORRECT):
```python
# Each layer has persistent gauge frames (per position)
class PureFEPLayer:
    def __init__(...):
        # Position-dependent parameters (ALL persistent)
        self.prior_mu = nn.Parameter(...)     # (seq_len, K)
        self.prior_sigma = nn.Parameter(...)  # (seq_len, K)
        self.phi = nn.Parameter(...)          # (seq_len, phi_dim) NEW!

    def forward(...):
        # Use persistent phi (not zeros!)
        phi = self.phi[:N]  # Slice to sequence length

        # VFE includes prior coupling
        F = (α * KL(q||p) +
             Σ β_ij * KL(q_i||Ω_ij·q_j) +
             Σ γ_ij * KL(p_i||Ω_ij·p_j) +  # ENABLED!
             E_q[-log p(o|c)])

        # Update phi via VFE gradient
        grad_phi = autograd.grad(F, phi)[0]
        phi.data -= phi_lr * grad_phi  # dφ/dt = -∂F/∂φ
```

**Benefits:**
1. φ evolves via VFE (not encoded)
2. φ persistent across batches (learns structure)
3. Prior coupling enabled (∂F/∂φ includes priors!)
4. Position structure EMERGES from minimizing F

## Implementation Steps

### 1. Add phi as Parameter in PureFEPLayer

**File:** `transformer/pure_fep_transformer.py` (around line 920)

```python
# After prior_mu and prior_sigma initialization
# Add persistent gauge frames (per position)
self.phi = nn.Parameter(
    torch.randn(config.seq_length, config.phi_dim) * 0.1
)
```

### 2. Remove GaugePositionEncoder

**File:** `transformer/pure_fep_transformer.py` (lines 2131-2139)

```python
# DELETE:
if config.position_mode in ['gauge_frame', 'both']:
    self.gauge_position = GaugePositionEncoder(...)
else:
    self.gauge_position = None

# Position mode is now OBSOLETE - phi evolves via VFE
```

### 3. Use Persistent phi in init_beliefs

**File:** `transformer/pure_fep_transformer.py` (line 1042)

```python
# BEFORE:
phi = torch.zeros(B, N, phi_dim, device=device)

# AFTER:
# Use persistent learned phi (slice to sequence length)
N_phi = min(N, self.phi.shape[0])
phi = self.phi[:N_phi].unsqueeze(0).expand(B, -1, -1).clone()

# Pad if needed
if N > N_phi:
    phi_pad = torch.zeros(B, N - N_phi, phi_dim, device=device)
    phi = torch.cat([phi, phi_pad], dim=1)

# Enable gradients for VFE descent
phi = phi.detach().requires_grad_(True)
```

### 4. Enable Prior Coupling by Default

**File:** `transformer/pure_fep_transformer.py` (line 622)

```python
# BEFORE:
prior_coupling_enabled: bool = False

# AFTER:
prior_coupling_enabled: bool = True  # ESSENTIAL for dφ/dt!
```

### 5. Update phi via VFE Gradient

**File:** `transformer/pure_fep_transformer.py` (around line 2020)

```python
# After VFE steps complete, update persistent phi
if self.config.gauge_evolution_enabled and phi.grad is not None:
    with torch.no_grad():
        # Average gradient across batch
        grad_phi_batch = phi.grad.mean(dim=0)  # (N, phi_dim)

        # Update persistent phi: dφ/dt = -∂F/∂φ
        N_phi = min(N, self.phi.shape[0])
        self.phi[:N_phi] -= self.config.gauge_lr * grad_phi_batch[:N_phi]

        # Optional: normalize to prevent explosion
        phi_norm = self.phi.norm(dim=-1, keepdim=True)
        max_norm = self.config.phi_max_norm  # e.g., π
        self.phi.data = torch.where(
            phi_norm > max_norm,
            self.phi * (max_norm / phi_norm),
            self.phi
        )
```

### 6. Update Config Defaults

**File:** `transformer/pure_fep_transformer.py` (lines 622-633)

```python
# Prior coupling: ENABLE by default (essential for φ evolution)
prior_coupling_enabled: bool = True  # Was False
lambda_prior: float = 0.1

# Gauge evolution: ENABLE by default
gauge_evolution_enabled: bool = True  # Was False
gauge_lr: float = 0.01

# Position mode: OBSOLETE (phi evolves, not encoded)
# Keep for backward compat but document as deprecated
position_mode: str = 'none'  # Was 'gauge_frame'
```

### 7. Update Train Config

**File:** `transformer/train_pure_FEP.py` (lines 103-112)

```python
# Prior coupling: essential for VFE
'prior_coupling_enabled': True,  # Was False
'lambda_prior': 0.1,

# Gradient-based updates: all from VFE
'gradient_prior_updates': True,
'prior_grad_lr': 0.01,

# Gauge evolution: φ evolves via VFE
'gauge_evolution_enabled': True,
'gauge_lr': 0.01,
```

## Expected Behavior After Changes

### Emergence Dynamics

**Fast timescale (Q-flow):**
- Beliefs minimize F via natural gradient
- Converge in ~20 VFE steps per forward

**Slow timescale (P-flow + φ-flow):**
- Priors drift toward successful beliefs
- φ evolves to minimize transport costs
- Position structure emerges over many batches

**Meta-agent formation:**
1. Tokens at similar positions see similar patterns
2. Prior coupling makes p_i ≈ p_j for similar positions
3. Gauge alignment makes φ_i ≈ φ_j
4. β_ij increases → beliefs align → cluster forms
5. **Emergent meta-agent** with coherent (q, p, φ)

### Position Information

**Not encoded** - position structure emerges because:
1. Causal masking (token i can't see j > i)
2. Sequential processing (earlier tokens processed first)
3. VFE minimization finds φ configuration that reduces alignment costs
4. φ_i ≠ φ_j naturally from minimizing Σ γ_ij·KL(p_i||Ω_ij·p_j)

**Position "at the end" vs "at the start"** emerges from different φ values minimizing F!

## Testing Plan

### 1. Verify φ Persistence
```python
# Check phi changes across batches
phi_init = model.layers[0].phi.data.clone()
# Train 10 batches
phi_final = model.layers[0].phi.data
assert (phi_final != phi_init).any()  # Should evolve!
```

### 2. Verify Prior Coupling Active
```python
# Check VFE includes γ term
assert config.prior_coupling_enabled == True
assert config.lambda_prior > 0
```

### 3. Monitor φ Evolution
```python
# Log phi norm and diversity
phi_norm = model.layers[0].phi.norm(dim=-1).mean()
phi_std = model.layers[0].phi.std(dim=0).mean()
print(f"φ norm: {phi_norm:.3f}, diversity: {phi_std:.3f}")
```

### 4. Check Meta-Agent Formation
```python
# Compute prior similarity matrix
p = model.layers[0].prior_mu  # (N, K)
similarity = F.cosine_similarity(p.unsqueeze(1), p.unsqueeze(0), dim=-1)
# Should see block structure emerge over training!
```

## Summary

**Current implementation is NOT pure VFE:**
- φ "encoded" from position → violates dφ/dt = -∂F/∂φ
- Prior coupling disabled → no ∂F/∂φ from priors
- φ re-initialized → not persistent learning

**Principled pure VFE:**
- φ is parameter that evolves via ∂F/∂φ
- Prior coupling enabled → ∂F/∂φ includes alignment
- φ persistent → structure emerges over time
- Position NOT imposed → emerges from minimizing F

This is the **true fiber bundle picture** - each agent has local (q, p, φ) coordinates that minimize the global free energy!
