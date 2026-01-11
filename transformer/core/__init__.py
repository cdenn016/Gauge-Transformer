"""
Core Model Components
=====================

This module contains the core transformer model components:
- GaugeTransformerLM: Main language model
- GaugeTransformerBlock/Stack: Transformer blocks
- Attention: KL-divergence based attention
- VariationalFFN: VFE-based feed-forward networks
- Embeddings: Token and positional embeddings
- PriorBank: Token-dependent priors
"""

from transformer.core.model import GaugeTransformerLM, create_gauge_transformer_lm
from transformer.core.blocks import GaugeTransformerBlock, GaugeTransformerStack
from transformer.core.attention import (
    compute_attention_weights,
    compute_attention_weights_local,
    compute_attention_weights_sparse,
    compute_kl_matrix,
    aggregate_messages,
    IrrepMultiHeadAttention,
    create_attention_mask,
    compute_transport_operators,
)
from transformer.core.embeddings import (
    GaugeTokenEmbedding,
    GaugePositionalEncoding,
)
from transformer.core.prior_bank import PriorBank
from transformer.core.ffn import FFNWrapper
from transformer.core.variational_ffn import VariationalFFNDynamic

__all__ = [
    # Main model
    'GaugeTransformerLM',
    'create_gauge_transformer_lm',

    # Blocks
    'GaugeTransformerBlock',
    'GaugeTransformerStack',

    # Attention
    'compute_attention_weights',
    'compute_attention_weights_local',
    'compute_attention_weights_sparse',
    'compute_kl_matrix',
    'aggregate_messages',
    'IrrepMultiHeadAttention',
    'create_attention_mask',
    'compute_transport_operators',

    # Embeddings
    'GaugeTokenEmbedding',
    'GaugePositionalEncoding',

    # Prior bank
    'PriorBank',

    # FFN
    'FFNWrapper',
    'VariationalFFNDynamic',
]
