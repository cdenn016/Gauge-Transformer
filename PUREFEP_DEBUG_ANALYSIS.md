# PureFEP Implementation Debug Analysis

**Date:** 2026-01-05
**Status:** Learning is not working
**Branch:** `claude/debug-purefep-ceet0`

---

## Executive Summary

The pureFEP implementation is **theoretically sound but has critical implementation issues** that prevent learning. The main problems are:

1. **Discriminative learning rate is too small** (0.0001 = 0.01 × 0.01)
2. **Observation gradients are incorrectly normalized** (divided by B×N)
3. **Token prior updates happen with stale beliefs**
4. **Belief-chasing update is disabled**, leaving only discriminative updates
5. **Position-dependent vs token-dependent prior confusion**

---

## Architecture Overview

### Core Concept

PureFEP learns entirely through **Variational Free Energy (VFE) minimization** without backpropagation:

```
F = α·Σ_i KL(q_i||p_i)                          [Self-coupling]
  + λ_β·Σ_{i,j} β_ij·KL(q_i||Ω_{ij}·q_j)       [Belief alignment]
  + λ_obs·Σ_i E_{q_i}[-log p(y_i|z_i)]         [Observation likelihood]
```

### Two-Timescale Dynamics

**Fast (Q-flow):** Beliefs minimize VFE via gradient descent (perception)
**Slow (P-flow):** Priors adapt toward beliefs (learning)

---

## Critical Issues Found

### Issue #1: Discriminative Learning Rate Too Small

**Location:** `pure_fep_transformer.py:2797`

```python
disc_lr = base_blend * 0.01  # base_blend ≈ 0.01 → disc_lr ≈ 0.0001
```

**Problem:**
- The discriminative update pushes confused priors apart
- Learning rate is 0.0001 (0.01 × 0.01) - **100x smaller than intended**
- At this rate, priors barely move even with large errors
- This was introduced in commit `652facb` to prevent "explosion"

**Impact:** 🔴 CRITICAL - Priors don't learn token distinctions

**Evidence:**
- Commit message says "0.01x lr" meaning 1% of base rate
- Base rate is already slow (0.01), so 0.01 × 0.01 = 0.0001
- For typical embedding distances O(1), update is ~0.0001 per step
- Would need 10,000 steps to move distance 1.0

---

### Issue #2: Observation Gradient Normalization

**Location:** `pure_fep_transformer.py:1230, 1233`

```python
grad_sigma = grad_sigma + lambda_obs * grad_sigma_ce / (B * N)
grad_mu = grad_mu + lambda_obs * grad_mu_ce / (B * N)
```

**Problem:**
- CE gradient is divided by batch size × sequence length
- For B=24, N=24: division by 576!
- This makes the observation term negligible compared to self-coupling
- **The VFE loss itself uses reduction='sum'** (line 1223), so gradient is already summed
- Dividing by B×N essentially applies reduction='mean' TWICE

**Impact:** 🔴 CRITICAL - Beliefs don't track targets

**Correct approach:**
- Option A: Use reduction='mean' and don't divide
- Option B: Use reduction='sum' and divide by B (not B×N)
- Option C: Don't divide at all (natural gradient rescales anyway)

---

### Issue #3: Token Prior Updates Use Stale Beliefs

**Location:** `pure_fep_transformer.py:2762-2828`

**Problem Flow:**
1. Model runs forward pass → computes beliefs q(z|x)
2. Beliefs are frozen (no_grad context)
3. Model computes logits from **frozen beliefs**
4. Discriminative update pushes priors based on **predictions from frozen beliefs**

**The Issue:**
- By the time discriminative update happens, beliefs are stale
- Priors are updated based on OLD predictions, not current state
- This creates a temporal mismatch in learning

**Impact:** 🟡 MODERATE - Slows learning convergence

**Better approach:**
- Update priors during VFE steps, not after
- OR: Run one more VFE step after prior update to sync

---

### Issue #4: Belief-Chasing Disabled

**Location:** `pure_fep_transformer.py:2755-2759`

```python
# DISABLED for now - this update makes priors converge together
# self.prior_bank.prior_mu.data[mask] = (
#     (1 - blend_rate) * self.prior_bank.prior_mu.data[mask] +
#     blend_rate * avg_beliefs[mask]
# )
```

**Problem:**
- Original update: `π_target ← (1-lr)·π_target + lr·q`
- This was disabled because it made priors converge (all tokens → same embedding)
- **Root cause:** When q ≈ π_input (input prior), pulling π_target toward q makes target ≈ input

**Current state:**
- Only discriminative update remains
- No attractive force, only repulsive
- Priors can drift without grounding

**Impact:** 🟡 MODERATE - Priors lack attraction to data

**Fix needed:**
- Keep discriminative repulsion
- Add attraction ONLY from correct predictions
- Or: Use contrastive loss properly (pull correct, push incorrect)

---

### Issue #5: Position vs Token Prior Confusion

**Location:** Multiple files

**The Confusion:**

1. **Layer priors** (`PureFEPLayer.prior_mu`) are **position-dependent**: shape `(seq_len, embed_dim)`
   - Updated in `_update_priors_with_prediction_error()` via hierarchical flow
   - Represent "what should be believed at each position"

2. **Token priors** (`PriorBank.prior_mu`) are **token-dependent**: shape `(vocab_size, embed_dim)`
   - Updated in discriminative update section
   - Represent "what each token means"

**The Issue:**
- These are orthogonal concepts but both called "priors"
- Layer priors learn position-specific patterns (e.g., "2nd token is usually a noun")
- Token priors learn token semantics (e.g., "word 'cat' means X")
- **Both** need to learn, but current code focuses on token priors only

**Impact:** 🟡 MODERATE - Layer priors not leveraged

**Evidence:**
- Layer prior updates at lines 2405-2426
- Token prior updates at lines 2762-2828
- They operate independently with different mechanisms

---

## Issue #6: Observation Gradient Computation Issues

**Location:** `pure_fep_transformer.py:1200-1233`

**Current Flow:**
```python
# 1. Detach beliefs
mu_q_grad = mu_q.detach().requires_grad_(True)

# 2. Compute CE
ce_for_grad = F.cross_entropy(logits_grad.view(-1, V), targets.view(-1), reduction='sum')

# 3. Backprop
grad_mu_ce = torch.autograd.grad(ce_for_grad, mu_q_grad, retain_graph=True)[0]

# 4. Normalize (PROBLEM!)
grad_mu = grad_mu + lambda_obs * grad_mu_ce / (B * N)
```

**Problems:**
1. **Double normalization:** CE already summed, then divided by B×N
2. **Detach breaks computational graph:** Natural gradients can't flow through
3. **Separate gradient computation:** Creates second forward pass overhead

**Impact:** 🔴 CRITICAL - Observation term too weak

---

## Issue #7: Natural Gradient Scaling

**Location:** `pure_fep_transformer.py:1257-1259`

```python
nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
    grad_mu, grad_sigma, sigma_q, eps=self.config.eps
)
```

**The natural gradient formula:**
```python
nat_grad_μ = Σ_q · grad_μ  # Multiply by covariance
```

**Problem:**
- If σ_q ≈ 1.0 (initial scale), then nat_grad ≈ grad (no effect)
- If σ_q << 1.0 (confident), then nat_grad << grad (too small)
- If σ_q >> 1.0 (uncertain), then nat_grad >> grad (too large)

**Current behavior:**
- Initial σ ≈ 1.0 from `init_sigma_scale=1.0`
- Natural gradient ≈ vanilla gradient initially
- As σ changes, scaling becomes unpredictable

**Impact:** 🟡 MODERATE - Learning dynamics unstable

---

## Recommended Fixes

### Priority 1: Fix Discriminative Learning Rate

**File:** `transformer/pure_fep_transformer.py:2797`

```python
# CURRENT (TOO SMALL):
disc_lr = base_blend * 0.01  # ≈ 0.0001

# FIX: Make it meaningful relative to base_blend
disc_lr = base_blend * 0.5  # ≈ 0.005 (50% of base prior lr)
```

**Rationale:**
- Discriminative updates need to be comparable to generative updates
- 0.5× base rate allows priors to separate while maintaining stability
- Still conservative enough to prevent explosion

---

### Priority 2: Fix Observation Gradient Normalization

**File:** `transformer/pure_fep_transformer.py:1230-1233`

**Option A: Remove the division entirely**
```python
# Natural gradient already handles scaling
grad_sigma = grad_sigma + lambda_obs * grad_sigma_ce
grad_mu = grad_mu + lambda_obs * grad_mu_ce
```

**Option B: Use mean reduction in CE**
```python
# Change line 1220-1224:
ce_for_grad = F.cross_entropy(
    logits_grad.view(-1, self.config.vocab_size),
    targets.view(-1),
    reduction='mean'  # Changed from 'sum'
)
# Then don't divide by B×N
grad_mu = grad_mu + lambda_obs * grad_mu_ce
```

**Recommended:** Option B - cleaner and matches standard practice

---

### Priority 3: Re-enable Attraction with Correct Predictions

**File:** `transformer/pure_fep_transformer.py:2750-2760`

```python
# NEW: Separate updates for correct vs incorrect
with torch.no_grad():
    # Get predictions
    logits = self.prior_bank.decode(final_mu_q, final_sigma_q, tau=1.0)
    predictions = logits.argmax(dim=-1).view(-1)
    targets_flat = targets.view(-1)

    # CORRECT predictions: ATTRACT prior to belief
    correct_mask = predictions == targets_flat
    if correct_mask.any():
        # ... pull π_target toward q when prediction is correct ...

    # INCORRECT predictions: REPULSE confused priors
    wrong_mask = ~correct_mask
    if wrong_mask.any():
        # ... existing discriminative update ...
```

**Rationale:**
- Correct predictions: Prior is working → reinforce it
- Incorrect predictions: Prior is confused → separate them
- Combines attraction and repulsion properly

---

### Priority 4: Increase lambda_obs

**File:** `transformer/train_pure_FEP.py:75`

```python
# CURRENT:
'lambda_obs': 1.0,

# FIX: Make observation term stronger
'lambda_obs': 10.0,  # Compensates for weak gradients
```

**Rationale:**
- If gradient normalization can't be fixed immediately
- Increase weight to compensate
- Standard in VFE: observation term should dominate during inference

---

## Testing Strategy

### Test 1: Minimal Overfitting Test
```python
# Single batch, 10 epochs
# Should memorize perfectly if learning works
batch_size = 4
seq_length = 8
vocab_size = 16
epochs = 100
```

**Success criteria:**
- Train loss → 0.0
- All predictions correct on training batch
- Token priors visibly separate in embedding space

### Test 2: Learning Rate Sweep
```python
for disc_mult in [0.01, 0.1, 0.5, 1.0]:
    disc_lr = base_blend * disc_mult
    # Train and measure convergence speed
```

### Test 3: Gradient Magnitude Monitoring
```python
# Add logging in vfe_step:
print(f"grad_mu_kl: {grad_mu_kl.abs().mean():.6f}")
print(f"grad_mu_align: {grad_mu_align.abs().mean():.6f}")
print(f"grad_mu_ce: {grad_mu_ce.abs().mean():.6f}")
print(f"Ratio CE/KL: {(grad_mu_ce.abs().mean() / grad_mu_kl.abs().mean()):.3f}")
```

**Expected healthy ratios:**
- CE/KL should be 1-10× during learning
- If CE/KL < 0.1, observation gradient too weak

---

## Summary of Root Causes

1. **Discriminative lr = 0.0001** → Priors barely move
2. **Observation grad ÷ (B×N)** → CE signal ~500× too weak
3. **Attraction disabled** → No pull toward data, only push
4. **Stale beliefs** → Updates lag behind current state

**Combined effect:** Model can't learn because:
- Beliefs don't chase targets (weak obs gradient)
- Priors don't learn from beliefs (too-small disc_lr)
- No positive reinforcement (attraction disabled)

---

## Fixes Applied

### ✅ Fix 1: Discriminative Learning Rate (Priority 1)

**File:** `transformer/pure_fep_transformer.py:2795-2798`

**Changed:**
```python
# BEFORE:
disc_lr = base_blend * 0.01  # ≈ 0.0001

# AFTER:
disc_lr = base_blend * 0.5  # ≈ 0.005 (50× larger!)
```

**Impact:** Discriminative updates now provide **50× stronger learning signal**

---

### ✅ Fix 2: Observation Gradient Normalization (Priority 2)

**File:** `transformer/pure_fep_transformer.py:1220-1236`

**Changed:**
```python
# BEFORE:
ce_for_grad = F.cross_entropy(..., reduction='sum')
grad_mu = grad_mu + lambda_obs * grad_mu_ce / (B * N)  # Double normalization!

# AFTER:
ce_for_grad = F.cross_entropy(..., reduction='mean')
grad_mu = grad_mu + lambda_obs * grad_mu_ce  # No division
```

**Impact:** Observation gradient now **~500× stronger** (for typical B=24, N=24)

---

### ✅ Fix 3: Contrastive Learning (Priority 3)

**File:** `transformer/pure_fep_transformer.py:2777-2807`

**Added:**
```python
# CORRECT predictions: Attract prior to belief
if correct_mask.any():
    attr_lr = base_blend * 0.5
    π_correct ← (1 - lr)·π_correct + lr·belief

# INCORRECT predictions: Push confused priors apart
if wrong_mask.any():
    disc_lr = base_blend * 0.5
    π_target ← π_target + lr·(π_target - π_predicted)
```

**Impact:** Now has both **attraction** (reinforcement) and **repulsion** (discrimination)

---

### ✅ Fix 4: Gradient Logging

**File:** `transformer/pure_fep_transformer.py:1238-1250`

**Added:**
- Config option: `debug_gradient_logging: bool = False`
- Logs gradient magnitudes during VFE steps
- Shows CE/Self gradient ratio for diagnosis

**Usage:**
```python
config = PureFEPConfig(
    ...,
    debug_gradient_logging=True  # Enable logging
)
```

---

## Next Steps

1. ✅ **Applied Priority 1 fix** (disc_lr: 50× stronger)
2. ✅ **Applied Priority 2 fix** (obs gradient: ~500× stronger)
3. ✅ **Applied Priority 3 fix** (contrastive learning)
4. ✅ **Added gradient monitoring**
5. 🔄 **Run minimal overfitting test** (pending)
6. 🔄 **Validate fixes with training** (pending)
7. 🔄 **Hyperparameter sweep** (pending)

---

## References

- **Commit 652facb:** "Fix discriminative update: use raw diff with clipping and 0.01x lr"
  - Introduced the 0.01× multiplier (too conservative)
- **Commit 8551755:** "Replace belief-chasing with discriminative prior updates"
  - Disabled the EMA attraction update
- **Lines 2762-2829:** Discriminative update implementation
- **Lines 1200-1233:** Observation gradient computation
- **Lines 1257-1295:** Natural gradient and belief update

