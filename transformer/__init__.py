# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 12:01:15 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
Gauge-Theoretic Transformer Package
====================================

Implements Hamiltonian dynamics on SPD manifolds for transformer architectures.
"""

# Suppress noisy Triton warnings about missing CUDA binaries on Windows
# These occur because Triton looks for cuobjdump.exe and nvdisasm.exe
# which are only in the CUDA Toolkit (not required for PyTorch GPU usage)
import warnings
warnings.filterwarnings(
    "ignore",
    message="Failed to find cuobjdump",
    category=UserWarning,
    module="triton"
)
warnings.filterwarnings(
    "ignore",
    message="Failed to find nvdisasm",
    category=UserWarning,
    module="triton"
)

from .model import GaugeTransformerLM
from .train import Trainer, TrainingConfig
from .data import (
    create_dataloaders,
    create_char_dataloaders,
    create_byte_dataloaders,
)

# New unified training module (Phase 3 consolidation)
from .training import (
    create_optimizer,
    create_param_groups,
    MetricsTracker,
)
# Re-export preset configs
from .training.config import (
    get_standard_config,
    get_vfe_dynamic_config,
    get_pure_fep_config,
)

__all__ = [
    # Core model
    'GaugeTransformerLM',

    # Training (legacy exports for backward compatibility)
    'Trainer',
    'TrainingConfig',

    # Training (new unified module)
    'create_optimizer',
    'create_param_groups',
    'MetricsTracker',
    'get_standard_config',
    'get_vfe_dynamic_config',
    'get_pure_fep_config',

    # Data loading
    'create_dataloaders',
    'create_char_dataloaders',
    'create_byte_dataloaders',
]