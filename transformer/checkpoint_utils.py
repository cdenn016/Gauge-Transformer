"""
Checkpoint Loading Utilities
=============================

Shared utilities for loading trained model checkpoints.
Used by visualization and analysis scripts.
"""

import torch
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from transformer.model import GaugeTransformerLM


def load_model(checkpoint_path: str) -> Tuple[GaugeTransformerLM, Dict[str, Any]]:
    """
    Load a trained GaugeTransformerLM model from checkpoint.

    Handles both:
    - experiment_config.json (preferred)
    - Config embedded in checkpoint file (fallback)

    Args:
        checkpoint_path: Path to best_model.pt or similar checkpoint

    Returns:
        model: Loaded GaugeTransformerLM in eval mode
        config: Configuration dictionary used to create the model

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If config cannot be determined
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_dir = checkpoint_path.parent
    config_json_path = checkpoint_dir / "experiment_config.json"

    # Default config (fallback values)
    config = {
        'vocab_size': 50257,
        'embed_dim': 25,
        'n_layers': 1,
        'irrep_spec': [('ℓ0', 5, 1), ('ℓ1', 3, 3), ('ℓ2', 1, 5)],
        'hidden_dim': 112,
        'max_seq_len': 128,
        'kappa_beta': 1.0,
        'dropout': 0.1,
        'pos_encoding_mode': 'learned',
        'evolve_sigma': True,
        'evolve_phi': False,
        'tie_embeddings': True,
        'use_diagonal_covariance': True,
        'ffn_mode': 'variational_gradient_engine',
    }

    # Try loading from experiment_config.json first (more reliable)
    if config_json_path.exists():
        print(f"Loading config from {config_json_path}")
        with open(config_json_path, 'r') as f:
            json_data = json.load(f)

        # Check if config is nested under a 'config' key
        if 'config' in json_data and isinstance(json_data['config'], dict):
            config.update(json_data['config'])
            print("Loaded nested config from experiment_config.json")
        else:
            config.update(json_data)
            print("Loaded config from experiment_config.json")
    else:
        # Try to extract config from checkpoint pickle
        print(f"Warning: {config_json_path} not found, trying to extract from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'config' in checkpoint:
            ckpt_config = checkpoint['config']
            if isinstance(ckpt_config, dict):
                config.update(ckpt_config)
            elif hasattr(ckpt_config, '__dict__'):
                config.update(vars(ckpt_config))
            print("Extracted config from checkpoint")
        else:
            print("Warning: No config found, using defaults")

    # Handle config key translations for backward compatibility
    if 'kappa_beta' not in config and 'kappa_beta_base' in config:
        config['kappa_beta'] = config['kappa_beta_base']
    if 'use_diagonal_covariance' not in config and 'diagonal_covariance' in config:
        config['use_diagonal_covariance'] = config['diagonal_covariance']

    print(f"Config: K={config['embed_dim']}, vocab={config['vocab_size']}, "
          f"layers={config['n_layers']}")

    # Create model
    model = GaugeTransformerLM(config)

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded checkpoint from {checkpoint_path}")
    model.eval()

    return model, config


def get_tokenizer(config: Dict[str, Any], dataset_name: Optional[str] = None):
    """
    Get tokenizer for a given config.

    Tries WikiTextDataset first, falls back to tiktoken.

    Args:
        config: Model configuration dict
        dataset_name: Dataset name override (default: from config or 'wikitext-2')

    Returns:
        tokenizer: Object with encode/decode methods
    """
    if dataset_name is None:
        dataset_name = config.get('dataset', 'wikitext-2')

    try:
        from transformer.data import WikiTextDataset
        dataset = WikiTextDataset(
            split='train',
            max_seq_len=128,
            dataset_name=dataset_name
        )
        return dataset
    except Exception as e:
        print(f"Warning: Could not load dataset tokenizer: {e}")
        try:
            import tiktoken
            return tiktoken.get_encoding("gpt2")
        except ImportError:
            print("Warning: tiktoken not available")
            return None
