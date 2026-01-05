# Comparison: VFE_dynamic vs pure_fep_transformer

**Date:** 2026-01-05
**Purpose:** Understand architectural differences and identify simplifications from vfe_dynamic

---

## Architecture Overview

### VFE_dynamic (variational_ffn.py)

**What it is:** An FFN layer that can be used in standard transformer architectures

**Structure:**
```
GaugeTransformer
├── Embedding (nn.Embedding)
├── Transformer Blocks
│   ├── Attention (KL-based)
│   └── FFN (VariationalFFNDynamic) ← VFE_dynamic mode
└── Output (nn.Linear or KL-to-prior)
```

**Key features:**
- FFN layer replaces standard MLP
- Can be used in any transformer
- Dynamic β: recomputes attention at each VFE step
- Optional pure_fep_mode for learning via prior evolution
- Can use PriorBank (token-dependent priors)

---

### pure_fep_transformer (pure_fep_transformer.py)

**What it is:** A complete transformer architecture built entirely around VFE

**Structure:**
```
PureFEPTransformer
├── PriorBank (unified embedding + output)
├── GaugePositionEncoder (φ for position)
├── PureFEPLayers (hierarchical)
│   ├── Self-attention via KL
│   ├── VFE gradient descent (Q-flow)
│   └── Prior updates (P-flow)
└── KL-to-prior decoding
```

**Key features:**
- No separate embedding/output layers
- PriorBank serves both purposes
- Hierarchical layers with parent-child prior flow
- Position via gauge frames φ
- Pure gradient-based learning (as of commit 72eeafc)

---

## Key Architectural Differences

### 1. **Position Encoding**

| Feature | VFE_dynamic | pure_fep_transformer |
|---------|-------------|---------------------|
| Position encoding | **None** (emergent from data!) | Gauge frames φ ∈ so(N) |
| Transport operators | Not used in FFN | Ω_ij = exp(φ_i)·exp(-φ_j) |
| Position dependence | Via attention mask only | Via gauge transport |

**Insight from vfe_dynamic:** Position can be **emergent** - no explicit encoding needed!
- Causal masking provides temporal order
- Beliefs evolve sequentially
- Position information implicit in processing order

**This is what you meant!** VFE_dynamic proved position encoding isn't necessary.

---

### 2. **Embedding & Output**

| Component | VFE_dynamic | pure_fep_transformer |
|-----------|-------------|---------------------|
| Embedding | nn.Embedding OR PriorBank | PriorBank only |
| Output | nn.Linear OR KL-to-prior | KL-to-prior only |
| Unified? | Optional | Always |

**VFE_dynamic flexibility:**
- Can use standard embedding + linear (for backprop training)
- Can use PriorBank + KL (for pure FEP)
- Modular design

**pure_fep_transformer purity:**
- Always uses PriorBank
- No learned W_out
- Principled but less flexible

---

### 3. **Prior Updates**

| Aspect | VFE_dynamic | pure_fep_transformer |
|--------|-------------|---------------------|
| Token priors | PriorBank (if enabled) | PriorBank always |
| Position priors | In FFN layer | In each PureFEPLayer |
| Hierarchical flow | No | Yes (parent → child) |
| Update mechanism | VFE gradient OR EMA | VFE gradient (as of 72eeafc) |

**VFE_dynamic simplicity:**
- Single-level priors
- No parent-child relationships
- Simpler P-flow

**pure_fep_transformer complexity:**
- Hierarchical prior structure
- Parent beliefs → child priors
- Multiple timescales

---

### 4. **Attention**

| Feature | VFE_dynamic | pure_fep_transformer |
|---------|-------------|---------------------|
| Attention | Computed in transformer block | Computed in layer |
| Dynamic β | Recomputed each VFE step | Recomputed each VFE step |
| Transport | Ω from gauge frames | Ω from gauge frames |
| Self-masking | Optional (mask_self_attention) | Always masked |

**Similar approach** - both use dynamic KL-based attention.

---

### 5. **Learning**

| Mode | VFE_dynamic | pure_fep_transformer |
|------|-------------|---------------------|
| Backprop | Default (pure_fep_mode=False) | Never (pure_fep_mode=True always) |
| Prior evolution | Optional (pure_fep_mode=True) | Always |
| VFE gradients | For Q-flow | For Q-flow AND P-flow |
| Hybrid capable? | Yes | No |

**VFE_dynamic flexibility:**
- Can train with backprop (faster, proven)
- Can switch to pure FEP (experimental)

**pure_fep_transformer purity:**
- No backprop path
- Theoretically pure
- Less battle-tested

---

## What Can Be Simplified in pure_fep_transformer?

Based on VFE_dynamic's proven simplicity:

### ✅ 1. **Remove Gauge Position Encoding**

VFE_dynamic **doesn't use** gauge frames for position - and it works!

**Current (complex):**
```python
# Position via gauge frames
self.gauge_position = GaugePositionEncoder(...)
phi = self.gauge_position(positions)
Omega = compute_transport(phi)
beta = softmax(-KL(q_i, Omega @ q_j))
```

**Simplified (like VFE_dynamic):**
```python
# NO position encoding!
# Just causal masking
beta = softmax(-KL(q_i, q_j)) * causal_mask
```

**Benefits:**
- No φ parameters to learn/update
- No matrix exponentials
- No transport operators
- Faster forward pass
- Position emerges from sequential processing

---

### ✅ 2. **Simplify to Single-Level Priors**

VFE_dynamic uses **flat** prior structure - no hierarchy.

**Current (complex):**
```python
# Hierarchical layers with parent-child flow
for layer in layers:
    layer.update_prior_from_parent(parent_beliefs)
```

**Simplified (like VFE_dynamic):**
```python
# Single level - position or token priors only
# No parent-child relationships
```

**Benefits:**
- Simpler P-flow
- Easier to debug
- Proven to work in VFE_dynamic

---

### ✅ 3. **Make Architecture Modular**

VFE_dynamic can be **dropped into** any transformer.

**Current:** Monolithic PureFEPTransformer class

**Simplified:** Modular components
- PureFEPLayer can work standalone
- Can integrate with standard transformer
- Easier testing and ablation

---

## Recommended Simplifications

Based on VFE_dynamic success, I recommend:

### **Phase 1: Remove Position Encoding**

```python
class SimplifiedPureFEPLayer:
    def __init__(self, seq_len, embed_dim):
        # Position-dependent priors (no gauge frames!)
        self.prior_mu = nn.Parameter(torch.randn(seq_len, embed_dim))
        self.prior_sigma = nn.Parameter(torch.ones(seq_len, embed_dim))
        # NO phi, NO generators

    def compute_attention(self, q_mu, q_sigma):
        # Direct KL without transport
        kl_pairwise = kl_divergence(q_mu, q_sigma)  # (B, N, N)
        beta = softmax(-kl_pairwise / kappa) * causal_mask
        return beta
```

### **Phase 2: Flatten Hierarchy**

```python
# Instead of parent-child prior flow
# Just: token priors + position priors

# Token priors (semantic)
prior_bank.prior_mu  # (vocab_size, K)

# Position priors (positional)
layer.prior_mu  # (seq_len, K)

# Both updated via VFE gradient
# No hierarchical relationships
```

### **Phase 3: Make Optional Features Truly Optional**

Like VFE_dynamic:
- `use_gauge_frames: bool = False`  # Enable/disable position encoding
- `use_hierarchy: bool = False`  # Enable/disable parent-child flow
- `use_prior_bank: bool = True`  # Token vs position priors

This allows:
- Minimal configuration (VFE_dynamic style)
- Full configuration (current pure_fep_transformer)
- Easy ablation studies

---

## Key Insight from VFE_dynamic

**Position encoding is not needed!**

The user was right - VFE_dynamic proved this empirically:
- No gauge frames φ
- No position embeddings
- No learned position encoding

Position information comes from:
1. **Causal masking** (can't attend to future)
2. **Sequential processing** (token i before i+1)
3. **Position-dependent priors** (each position has its prior)

The gauge frame machinery was **over-engineering** - beautiful theoretically, but unnecessary practically!

---

## Action Items

Should I:

1. **Remove gauge frames from pure_fep_transformer?**
   - Simplify to direct KL attention
   - Keep position-dependent priors
   - Match VFE_dynamic simplicity

2. **Flatten hierarchy?**
   - Single-level prior structure
   - No parent-child relationships
   - Proven design from VFE_dynamic

3. **Make it modular?**
   - PureFEPLayer that works standalone
   - Can integrate with any transformer
   - Easier testing

Let me know which simplifications you want!

