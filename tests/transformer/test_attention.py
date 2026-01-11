# -*- coding: utf-8 -*-
"""
Attention Tests
===============

Tests for transformer.core.attention module.
"""

import pytest
import torch
import math


class TestComputeAttentionWeights:
    """Test compute_attention_weights function."""

    def test_basic_computation(self, cpu_device):
        """Test basic attention weight computation."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        kappa = 1.0

        beta = compute_attention_weights(mu, sigma, kappa=kappa)

        # Check shape
        assert beta.shape == (B, N, N)

    def test_output_is_probability(self, cpu_device):
        """Test attention weights are valid probabilities."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        kappa = 1.0

        beta = compute_attention_weights(mu, sigma, kappa=kappa)

        # Check non-negative
        assert (beta >= 0).all(), "Attention weights should be non-negative"

        # Check sums to 1 along last dim
        sums = beta.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Attention weights should sum to 1"

    def test_output_finite(self, cpu_device):
        """Test output contains no NaN/Inf."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        kappa = 1.0

        beta = compute_attention_weights(mu, sigma, kappa=kappa)

        assert torch.isfinite(beta).all(), "Output contains NaN or Inf"

    def test_with_mask(self, cpu_device):
        """Test attention with causal mask."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        kappa = 1.0

        # Create causal mask
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        beta = compute_attention_weights(mu, sigma, kappa=kappa, mask=mask)

        # Check masked positions are zero
        upper_tri = torch.triu(torch.ones(N, N, device=cpu_device), diagonal=1).bool()
        assert (beta[:, upper_tri] == 0).all(), "Masked positions should be zero"

    def test_different_kappa_values(self, cpu_device):
        """Test with different temperature values."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1

        for kappa in [0.1, 1.0, 10.0]:
            beta = compute_attention_weights(mu, sigma, kappa=kappa)
            assert torch.isfinite(beta).all()
            assert (beta >= 0).all()

    def test_kappa_temperature_effect(self, cpu_device):
        """Test that lower kappa gives sharper distributions."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1

        beta_low = compute_attention_weights(mu, sigma, kappa=0.1)
        beta_high = compute_attention_weights(mu, sigma, kappa=10.0)

        # Lower kappa should give higher max attention (sharper)
        max_low = beta_low.max(dim=-1).values.mean()
        max_high = beta_high.max(dim=-1).values.mean()

        assert max_low > max_high, "Lower kappa should give sharper attention"


class TestComputeKLMatrix:
    """Test compute_kl_matrix function."""

    def test_basic_computation(self, cpu_device):
        """Test basic KL matrix computation."""
        from transformer.core.attention import compute_kl_matrix

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1

        kl_matrix = compute_kl_matrix(mu, sigma)

        # Check shape
        assert kl_matrix.shape == (B, N, N)

    def test_self_kl_is_zero(self, cpu_device):
        """Test KL(p||p) = 0 on diagonal."""
        from transformer.core.attention import compute_kl_matrix

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1

        kl_matrix = compute_kl_matrix(mu, sigma)

        # Diagonal should be zero (KL divergence to self)
        diag = torch.diagonal(kl_matrix, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5), \
            "KL(p||p) should be 0"

    def test_kl_non_negative(self, cpu_device):
        """Test KL divergence is non-negative."""
        from transformer.core.attention import compute_kl_matrix

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1

        kl_matrix = compute_kl_matrix(mu, sigma)

        assert (kl_matrix >= -1e-5).all(), "KL divergence should be non-negative"


class TestCreateAttentionMask:
    """Test create_attention_mask function."""

    def test_full_causal_mask(self, cpu_device):
        """Test full causal attention mask."""
        from transformer.core.attention import create_attention_mask

        N = 8
        mask = create_attention_mask(N, pattern='full', causal=True, device=cpu_device)

        # Should be lower triangular
        expected = torch.tril(torch.ones(N, N, device=cpu_device))
        assert torch.allclose(mask, expected)

    def test_full_bidirectional_mask(self, cpu_device):
        """Test full bidirectional attention mask."""
        from transformer.core.attention import create_attention_mask

        N = 8
        mask = create_attention_mask(N, pattern='full', causal=False, device=cpu_device)

        # Should be all ones
        expected = torch.ones(N, N, device=cpu_device)
        assert torch.allclose(mask, expected)

    def test_local_attention_mask(self, cpu_device):
        """Test local attention mask."""
        from transformer.core.attention import create_attention_mask

        N = 16
        window = 4
        mask = create_attention_mask(
            N, pattern='local', window=window, causal=True, device=cpu_device
        )

        # Check shape
        assert mask.shape == (N, N)

        # Check it's a valid mask (0 or 1)
        assert ((mask == 0) | (mask == 1)).all()


class TestIrrepMultiHeadAttention:
    """Test IrrepMultiHeadAttention module."""

    @pytest.fixture
    def attention_module(self, cpu_device):
        """Create attention module."""
        from transformer.core.attention import IrrepMultiHeadAttention

        K = 16
        n_heads = 2
        kappa = 1.0

        # Create simple generators
        generators = torch.randn(3, K, K, device=cpu_device)
        # Make antisymmetric
        generators = generators - generators.transpose(-1, -2)

        attention = IrrepMultiHeadAttention(
            embed_dim=K,
            n_heads=n_heads,
            generators=generators,
            kappa=kappa,
        )
        return attention.to(cpu_device)

    def test_creation(self, cpu_device):
        """Test creating attention module."""
        from transformer.core.attention import IrrepMultiHeadAttention

        K = 16
        generators = torch.randn(3, K, K, device=cpu_device)
        generators = generators - generators.transpose(-1, -2)

        attention = IrrepMultiHeadAttention(
            embed_dim=K,
            n_heads=2,
            generators=generators,
            kappa=1.0,
        )
        assert attention is not None

    def test_forward_pass(self, attention_module, cpu_device):
        """Test forward pass through attention."""
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1

        # Create causal mask
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        mu_out, sigma_out, phi_out, beta = attention_module(
            mu, sigma, phi, mask=mask
        )

        # Check output shapes
        assert mu_out.shape == (B, N, K)
        assert sigma_out.shape == sigma.shape
        assert beta.shape[-2:] == (N, N)

    def test_forward_output_finite(self, attention_module, cpu_device):
        """Test forward outputs are finite."""
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        mu_out, sigma_out, phi_out, beta = attention_module(
            mu, sigma, phi, mask=mask
        )

        assert torch.isfinite(mu_out).all(), "mu contains NaN/Inf"
        assert torch.isfinite(sigma_out).all(), "sigma contains NaN/Inf"
        assert torch.isfinite(beta).all(), "beta contains NaN/Inf"


class TestAggregateMessages:
    """Test aggregate_messages function."""

    def test_basic_aggregation(self, cpu_device):
        """Test basic message aggregation."""
        from transformer.core.attention import aggregate_messages

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        beta = torch.softmax(torch.randn(B, N, N, device=cpu_device), dim=-1)

        mu_agg = aggregate_messages(mu, beta)

        # Check shape
        assert mu_agg.shape == (B, N, K)

    def test_aggregation_with_identity_beta(self, cpu_device):
        """Test aggregation with identity attention (copies input)."""
        from transformer.core.attention import aggregate_messages

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)

        # Identity attention: each position attends only to itself
        beta = torch.eye(N, device=cpu_device).unsqueeze(0).expand(B, -1, -1)

        mu_agg = aggregate_messages(mu, beta)

        # Should be same as input
        assert torch.allclose(mu_agg, mu, atol=1e-5)


class TestComputeTransportOperators:
    """Test compute_transport_operators function."""

    def test_basic_computation(self, cpu_device):
        """Test basic transport operator computation."""
        from transformer.core.attention import compute_transport_operators

        B, N = 2, 8
        K = 16
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1

        # Create generators
        generators = torch.randn(3, K, K, device=cpu_device)
        generators = generators - generators.transpose(-1, -2)

        omega = compute_transport_operators(phi, generators)

        # Check shape: (B, N, N, K, K)
        assert omega.shape == (B, N, N, K, K)

    def test_identity_for_zero_phi(self, cpu_device):
        """Test transport is identity when phi=0."""
        from transformer.core.attention import compute_transport_operators

        B, N = 2, 8
        K = 16
        phi = torch.zeros(B, N, 3, device=cpu_device)

        generators = torch.randn(3, K, K, device=cpu_device)
        generators = generators - generators.transpose(-1, -2)

        omega = compute_transport_operators(phi, generators)

        # When phi_i = phi_j = 0, transport should be identity
        identity = torch.eye(K, device=cpu_device)
        diag_omega = omega[:, range(N), range(N)]  # Self-transport

        for b in range(B):
            for n in range(N):
                assert torch.allclose(diag_omega[b, n], identity, atol=1e-4), \
                    "Self-transport should be identity"
