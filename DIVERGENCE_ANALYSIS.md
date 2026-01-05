# PureFEP Divergence Analysis

**Date:** 2026-01-05
**Issue:** Model exploding (PPL: 115K → 382 quintillion in 50 steps)

---

## What Happened

After applying the "fixes" from commit 203cca4, the model **catastrophically diverged** instead of learning.

### Training Output:
```
Step 10: PPL = 115,483
Step 20: PPL = 784,579
Step 30: PPL = 213,195,696
Step 40: PPL = 349 trillion
Step 50: PPL = 382 quintillion
```

**Exponential explosion** - each step makes it ~5-10× worse.

---

## Root Cause: TOO AGGRESSIVE UPDATES

### Issue 1: Observation Gradient "Fix" Was a No-Op

**My analysis was WRONG:**

```python
# BEFORE:
ce = F.cross_entropy(..., reduction='sum')     # Sum over all elements
grad = autograd.grad(ce, mu)[0]                # Gradient of sum
grad_mu = grad_mu + lambda_obs * grad / (B*N)  # Divide by count

# AFTER:
ce = F.cross_entropy(..., reduction='mean')    # Mean over all elements
grad = autograd.grad(ce, mu)[0]                # Gradient of mean
grad_mu = grad_mu + lambda_obs * grad          # No division
```

**Mathematical equivalence:**
- `∂(sum)/∂x = sum of ∂loss_i/∂x`
- `∂(mean)/∂x = ∂(sum/N)/∂x = (∂sum/∂x) / N`
- Therefore: `∂(sum)/∂x / N` = `∂(mean)/∂x`

**Conclusion:** The two versions are **identical**! This "fix" did nothing.

---

### Issue 2: Discriminative LR 50× Too Large

**The Real Problem:**

```python
# BEFORE (too small):
disc_lr = base_blend * 0.01  # ≈ 0.0001

# "FIX" (WAY too large):
disc_lr = base_blend * 0.5   # ≈ 0.005 (50× increase!)
```

**Why this causes explosion:**
- Priors move 50× faster
- Each wrong prediction shifts embeddings by ~0.005
- With hundreds of wrong predictions per batch: 0.005 × 500 = 2.5 total shift
- Priors diverge from reasonable values
- KL divergences explode
- Logits become huge
- Perplexity → infinity

---

### Issue 3: Attraction Compounds the Problem

**Added this:**
```python
# Attraction for correct predictions
attr_lr = base_blend * 0.5  # Same 50× too-large rate!
```

**Problem:**
- Now priors move in BOTH directions (attract + repel)
- Double the movement speed
- Compounds the instability

---

## Why Observation Gradient Wasn't the Issue

**The original code was already correct:**

```python
ce_loss = F.cross_entropy(..., reduction='sum')
grad = autograd.grad(ce_loss, mu)[0]
final_grad = lambda_obs * grad / (B * N)
```

This gives **per-token average gradient**, which is appropriate for VFE.

The issue was never the observation gradient normalization. It was:
1. **Discriminative lr too small** (original issue)
2. **My "fix" overcorrected** by 50× (new issue)

---

## Correct Fix Strategy

### Principle: **Conservative Incremental Changes**

Don't jump from 0.01× to 0.5× - that's a 50× leap!

**Better approach:**
1. Start with 2× increase: `disc_lr = base_blend * 0.02`
2. Test for stability
3. If stable, try 5×: `disc_lr = base_blend * 0.05`
4. If stable, try 10×: `disc_lr = base_blend * 0.1`
5. Find the maximum stable rate

---

## Revised Fixes

### Fix 1: Conservative Discriminative LR

```python
# Balanced: 10× stronger than original, not 50×
disc_lr = base_blend * 0.1  # ≈ 0.001 (was 0.0001)
```

### Fix 2: Even More Conservative Attraction

```python
# Attraction should be gentler than repulsion
attr_lr = base_blend * 0.05  # ≈ 0.0005 (2× original, not 50×)
```

### Fix 3: Revert Observation Gradient (It Was Fine!)

```python
# Keep original - it was already correct
ce_for_grad = F.cross_entropy(..., reduction='sum')
grad_mu = grad_mu + lambda_obs * grad_mu_ce / (B * N)
```

### Fix 4: Add Prior Update Clipping

```python
# Clip prior updates to prevent explosion
prior_update = prior_update.clamp(-0.1, 0.1)  # Max 0.1 shift per step
```

---

## Testing Strategy

### Phase 1: Minimal Change
- `disc_lr = base_blend * 0.02` (2× original)
- No attraction (keep disabled)
- Monitor for stability

### Phase 2: Gradual Increase
- If stable, try `0.05`, then `0.1`, then `0.2`
- Stop when divergence starts
- Use highest stable value

### Phase 3: Add Attraction
- Once repulsion is stable
- Start with `attr_lr = disc_lr * 0.5`
- Gradually increase

---

## Key Lesson

**"Too weak" doesn't mean "make it 50× stronger"** - that causes instability.

**Proper debugging:**
1. Increase by 2×, test
2. If stable, increase by 2× again
3. Repeat until divergence
4. Use last stable value

**Binary search for optimal lr:**
- Too low: slow learning
- Too high: divergence
- Sweet spot: somewhere in between

---

## Next Steps

1. ✅ Revert observation gradient "fix" (it was a no-op anyway)
2. ✅ Scale back disc_lr to 0.1× (not 0.5×)
3. ✅ Scale back attr_lr to 0.05× (not 0.5×)
4. ✅ Add prior update clipping
5. 🔄 Test with conservative settings
6. 🔄 Gradually increase if stable

