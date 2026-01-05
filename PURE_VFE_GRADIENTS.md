# Pure FEP: True Gradient-Based Learning

**STATUS: ✅ IMPLEMENTED**
**Commit: TBD**

## Architecture Fix Required

### Current (HYBRID):
```python
# Position-dependent priors (layers)
if gradient_prior_updates:
    dp/dt = -grad_p VFE  ✓ (CORRECT!)
else:
    p ← (1-lr)*p + lr*q  ✗ (hand-crafted EMA)

# Token priors (PriorBank)
π_target ← π_target + disc_lr * (π_target - π_pred)  ✗ (hand-crafted discriminative)
π_correct ← (1-lr)*π + lr*q_correct  ✗ (hand-crafted attractive)
```

### Should Be (PURE):
```python
# ALL parameters via VFE gradient
dq/dt = -grad_q VFE
dp/dt = -grad_p VFE
dφ/dt = -grad_φ VFE
d(token_priors)/dt = -grad_{token_priors} VFE
```

---

## Implementation Plan

### 1. Token Priors Must Be Differentiable

Token priors appear in VFE through observation term:
```
F = ... + λ_obs * E_q[-log p(y|z)]

where p(y=v|z) = softmax(-KL(q || π_v) / τ)  # π_v = token priors
```

So: ∂F/∂π_v pulls token priors to minimize prediction error!

### 2. Remove Hand-Crafted Updates

**Delete lines 2761-2879** (discriminative/attractive updates)

Replace with:
```python
# Token priors updated via VFE gradient (same as position priors)
if self.config.gradient_prior_updates:
    # Observation term is differentiable w.r.t. token priors
    # Backprop through: CE(decode(q, token_priors), targets)
    grad_token_priors = autograd.grad(ce_loss, self.prior_bank.prior_mu)[0]

    # Gradient descent
    self.prior_bank.prior_mu.sub_(self.config.prior_grad_lr * grad_token_priors)
```

### 3. Ensure Differentiability

**Requirements:**
- Token priors must have `requires_grad=True`
- VFE loss must not detach token priors
- Use `with torch.enable_grad()` even in pure_fep_mode

**Check PriorBank:**
```python
# In PriorBank.__init__:
self.prior_mu = nn.Parameter(...)  # ✓ Already a Parameter!
self.log_prior_sigma = nn.Parameter(...)  # ✓ Already a Parameter!
```

**Check VFE computation:**
- Must NOT detach during observation gradient computation
- Keep computational graph alive

---

## Expected Behavior

With pure VFE gradients:

**Position priors:**
```
∂F/∂p = α * (p - q)/σ_p²  [pulls p toward q]
      + prior_coupling_gradient  [aligns priors]
      + ouroboros_gradient  [ancestral influence]
```

**Token priors:**
```
∂F/∂π_v = ∂/∂π_v [λ_obs * CE(decode(q, π), y)]
        = gradient through KL(q || π_v) in decode
        = pulls π_correct toward q_correct
        = pushes π_incorrect away from q_incorrect (naturally!)
```

**The contrastive behavior emerges automatically from the VFE!**

No need for hand-crafted:
- Discriminative updates (push apart)
- Attractive updates (pull together)
- Error weighting heuristics

Just: minimize free energy!

---

## Code Changes Required

### File: `transformer/pure_fep_transformer.py`

**1. Delete lines 2761-2879:**
```python
# REMOVE: All discriminative/attractive update code
```

**2. Add VFE gradient for token priors (around line 2700):**
```python
# Update token priors via VFE gradient
if self.config.gradient_prior_updates and targets is not None:
    # Ensure token priors are differentiable
    with torch.enable_grad():
        # Get final beliefs after VFE convergence
        final_mu_q = info['layer_infos'][-1]['beliefs'][0]
        final_sigma_q = info['layer_infos'][-1]['beliefs'][1]

        # Compute observation loss (differentiable w.r.t. token priors)
        logits = self.prior_bank.decode(final_mu_q, final_sigma_q, tau=1.0)
        ce_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
            reduction='sum'
        ) / (B * N)

        # Backprop to token priors
        grad_token_priors = torch.autograd.grad(
            ce_loss,
            self.prior_bank.prior_mu,
            retain_graph=False
        )[0]

        # Gradient descent on token priors
        self.prior_bank.prior_mu.sub_(
            self.config.prior_grad_lr * grad_token_priors
        )
```

**3. Ensure VFE doesn't detach in pure_fep_mode:**

Currently pure_fep_mode uses `with torch.no_grad()` which breaks gradients to token priors!

Change train_step:
```python
# BEFORE:
if self.config.pure_fep_mode:
    with torch.no_grad():  # ✗ Breaks gradient flow!
        logits, info = self(...)

# AFTER:
if self.config.pure_fep_mode:
    # Keep gradients for prior updates, but don't backprop through VFE
    logits, info = self(...)  # No no_grad!
    # Gradients flow to priors, but not through VFE dynamics
```

---

## Why This Will Work

1. **Simpler:** No hand-crafted update rules
2. **Principled:** Everything minimizes the same VFE
3. **Automatic:** Contrastive behavior emerges naturally
4. **Stable:** VFE landscape is well-defined
5. **Scalable:** Same mechanism for all parameters

**The key insight:**
VFE gradient w.r.t. token priors ALREADY implements:
- Attraction (via self-coupling term pulling priors to beliefs)
- Repulsion (via observation term separating confused priors)
- No need to hand-code these!

---

## Testing

After changes:
1. Check token prior gradients are non-zero
2. Verify priors move toward minimizing VFE
3. Confirm no explosion (use grad clipping)
4. Monitor: do confused priors separate? (should happen automatically)

