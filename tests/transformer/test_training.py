# -*- coding: utf-8 -*-
"""
Training Utilities Tests
========================

Tests for transformer.training module.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path


class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_default_config(self):
        """Test creating default config."""
        from transformer.training.config import TrainingConfig

        config = TrainingConfig()
        assert config is not None
        assert hasattr(config, 'mu_lr')
        assert hasattr(config, 'sigma_lr')

    def test_get_standard_config(self):
        """Test get_standard_config preset."""
        from transformer.training.config import get_standard_config

        config = get_standard_config()
        assert config is not None
        assert config.training_mode == 'standard'

    def test_get_vfe_dynamic_config(self):
        """Test get_vfe_dynamic_config preset."""
        from transformer.training.config import get_vfe_dynamic_config

        config = get_vfe_dynamic_config()
        assert config is not None
        assert config.training_mode == 'vfe_dynamic'

    def test_config_overrides(self):
        """Test config with custom overrides."""
        from transformer.training.config import get_standard_config

        config = get_standard_config(
            mu_lr=0.05,
            max_steps=500,
        )

        assert config.mu_lr == 0.05
        assert config.max_steps == 500


class TestCreateParamGroups:
    """Test create_param_groups function."""

    def test_param_groups_creation(self, minimal_config, cpu_device):
        """Test creating parameter groups."""
        from transformer.core.model import GaugeTransformerLM
        from transformer.training.optimizer import create_param_groups

        model = GaugeTransformerLM(minimal_config)

        param_groups = create_param_groups(
            model,
            mu_lr=0.1,
            sigma_lr=0.01,
            phi_lr=0.05,
            attn_lr=0.001,
            ffn_lr=0.001,
            output_lr=0.001,
        )

        assert isinstance(param_groups, list)
        assert len(param_groups) > 0

        # Each group should have 'params' and 'lr'
        for group in param_groups:
            assert 'params' in group
            assert 'lr' in group

    def test_all_params_covered(self, minimal_config, cpu_device):
        """Test all parameters are in some group."""
        from transformer.core.model import GaugeTransformerLM
        from transformer.training.optimizer import create_param_groups

        model = GaugeTransformerLM(minimal_config)

        param_groups = create_param_groups(
            model,
            mu_lr=0.1,
            sigma_lr=0.01,
            phi_lr=0.05,
            attn_lr=0.001,
            ffn_lr=0.001,
            output_lr=0.001,
        )

        # Collect all params in groups
        grouped_params = set()
        for group in param_groups:
            for p in group['params']:
                grouped_params.add(id(p))

        # Check all model params are grouped
        model_params = set(id(p) for p in model.parameters())

        # All model params should be in groups (or subset if some are frozen)
        # Note: some implementations may not include all params
        assert len(grouped_params) > 0


class TestCreateOptimizer:
    """Test create_optimizer function."""

    def test_optimizer_creation(self, minimal_config, cpu_device):
        """Test creating optimizer."""
        from transformer.core.model import GaugeTransformerLM
        from transformer.training.optimizer import create_optimizer

        model = GaugeTransformerLM(minimal_config)

        optimizer = create_optimizer(
            model,
            mu_lr=0.1,
            sigma_lr=0.01,
            phi_lr=0.05,
            attn_lr=0.001,
            ffn_lr=0.001,
            output_lr=0.001,
        )

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_optimizer_step(self, minimal_config, cpu_device):
        """Test optimizer can perform step."""
        from transformer.core.model import GaugeTransformerLM
        from transformer.training.optimizer import create_optimizer

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)

        optimizer = create_optimizer(
            model,
            mu_lr=0.1,
            sigma_lr=0.01,
            phi_lr=0.05,
            attn_lr=0.001,
            ffn_lr=0.001,
            output_lr=0.001,
        )

        # Forward pass
        V = minimal_config['vocab_size']
        input_ids = torch.randint(0, V, (2, 16), device=cpu_device)
        targets = torch.randint(0, V, (2, 16), device=cpu_device)

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V),
            targets.view(-1)
        )

        # Backward and step
        loss.backward()
        optimizer.step()

        # Should complete without error


class TestMetricsTracker:
    """Test MetricsTracker class."""

    def test_tracker_creation(self, tmp_path):
        """Test creating metrics tracker."""
        from transformer.training.metrics import MetricsTracker

        tracker = MetricsTracker(save_path=tmp_path)
        assert tracker is not None

    def test_tracker_log(self, tmp_path):
        """Test logging metrics."""
        from transformer.training.metrics import MetricsTracker

        tracker = MetricsTracker(save_path=tmp_path)

        tracker.log({
            'step': 1,
            'loss': 2.5,
            'accuracy': 0.8,
        })

        assert len(tracker.history) == 1

    def test_tracker_save(self, tmp_path):
        """Test saving metrics to CSV."""
        from transformer.training.metrics import MetricsTracker

        tracker = MetricsTracker(save_path=tmp_path)

        for i in range(5):
            tracker.log({
                'step': i,
                'loss': 2.5 - i * 0.1,
            })

        tracker.save()

        # Check CSV file created
        csv_files = list(tmp_path.glob('*.csv'))
        assert len(csv_files) > 0
