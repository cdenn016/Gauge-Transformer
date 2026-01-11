# -*- coding: utf-8 -*-
"""
Blocks Tests
============

Tests for transformer.core.blocks module.
"""

import pytest
import torch


class TestGaugeTransformerBlock:
    """Test GaugeTransformerBlock class."""

    @pytest.fixture
    def block(self, cpu_device):
        """Create a transformer block."""
        from transformer.core.blocks import GaugeTransformerBlock

        K = 16
        hidden_dim = 32
        kappa = 1.0

        # Create generators
        generators = torch.randn(3, K, K)
        generators = generators - generators.transpose(-1, -2)

        block = GaugeTransformerBlock(
            embed_dim=K,
            hidden_dim=hidden_dim,
            kappa=kappa,
            generators=generators,
            ffn_mode='learned',
        )
        return block.to(cpu_device)

    def test_creation(self, cpu_device):
        """Test creating transformer block."""
        from transformer.core.blocks import GaugeTransformerBlock

        K = 16
        generators = torch.randn(3, K, K)
        generators = generators - generators.transpose(-1, -2)

        block = GaugeTransformerBlock(
            embed_dim=K,
            hidden_dim=32,
            kappa=1.0,
            generators=generators,
            ffn_mode='learned',
        )
        assert block is not None

    def test_forward(self, block, cpu_device):
        """Test forward pass."""
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
        mu_prior = mu.clone()

        # Create causal mask
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        mu_out, sigma_out, phi_out, beta = block(
            mu, sigma, phi, mask=mask, mu_prior=mu_prior
        )

        # Check output shapes
        assert mu_out.shape == (B, N, K)
        assert beta.shape[-2:] == (N, N)

    def test_forward_output_finite(self, block, cpu_device):
        """Test forward outputs are finite."""
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
        mu_prior = mu.clone()
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        mu_out, sigma_out, phi_out, beta = block(
            mu, sigma, phi, mask=mask, mu_prior=mu_prior
        )

        assert torch.isfinite(mu_out).all()
        assert torch.isfinite(sigma_out).all()
        assert torch.isfinite(beta).all()


class TestGaugeTransformerStack:
    """Test GaugeTransformerStack class."""

    @pytest.fixture
    def stack(self, cpu_device):
        """Create a transformer stack."""
        from transformer.core.blocks import GaugeTransformerStack

        K = 16
        hidden_dim = 32
        n_layers = 2
        kappa = 1.0

        generators = torch.randn(3, K, K)
        generators = generators - generators.transpose(-1, -2)

        stack = GaugeTransformerStack(
            n_layers=n_layers,
            embed_dim=K,
            hidden_dim=hidden_dim,
            kappa=kappa,
            generators=generators,
            ffn_mode='learned',
        )
        return stack.to(cpu_device)

    def test_creation(self, cpu_device):
        """Test creating transformer stack."""
        from transformer.core.blocks import GaugeTransformerStack

        K = 16
        generators = torch.randn(3, K, K)
        generators = generators - generators.transpose(-1, -2)

        stack = GaugeTransformerStack(
            n_layers=2,
            embed_dim=K,
            hidden_dim=32,
            kappa=1.0,
            generators=generators,
            ffn_mode='learned',
        )
        assert stack is not None
        assert len(stack.blocks) == 2

    def test_forward(self, stack, cpu_device):
        """Test forward pass through stack."""
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
        mu_prior = mu.clone()
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        mu_out, sigma_out, phi_out, beta_list = stack(
            mu, sigma, phi, mask=mask, mu_prior=mu_prior
        )

        # Check output shapes
        assert mu_out.shape == (B, N, K)

        # Check we get beta from each layer
        assert len(beta_list) == 2

    def test_forward_output_finite(self, stack, cpu_device):
        """Test forward outputs are finite."""
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
        mu_prior = mu.clone()
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        mu_out, sigma_out, phi_out, beta_list = stack(
            mu, sigma, phi, mask=mask, mu_prior=mu_prior
        )

        assert torch.isfinite(mu_out).all()
        assert torch.isfinite(sigma_out).all()

    def test_multiple_layers(self, cpu_device):
        """Test stack with various layer counts."""
        from transformer.core.blocks import GaugeTransformerStack

        K = 16
        generators = torch.randn(3, K, K)
        generators = generators - generators.transpose(-1, -2)

        for n_layers in [1, 2, 4]:
            stack = GaugeTransformerStack(
                n_layers=n_layers,
                embed_dim=K,
                hidden_dim=32,
                kappa=1.0,
                generators=generators,
                ffn_mode='learned',
            ).to(cpu_device)

            assert len(stack.blocks) == n_layers

            B, N = 2, 8
            mu = torch.randn(B, N, K, device=cpu_device)
            sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
            phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
            mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

            mu_out, _, _, beta_list = stack(
                mu, sigma, phi, mask=mask, mu_prior=mu.clone()
            )

            assert len(beta_list) == n_layers
            assert torch.isfinite(mu_out).all()
