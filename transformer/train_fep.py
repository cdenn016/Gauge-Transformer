#!/usr/bin/env python3
"""
Training script for Pure FEP Transformer.

Minimal, clean training loop for the Free Energy Principle Transformer.
No bells and whistles - just the core VFE minimization.

Usage:
    python -m transformer.train_fep --dataset wikitext-2 --epochs 10
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .fep_transformer import FEPTransformer, IrrepSpec


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Model
    'vocab_size': 50257,      # GPT-2 tokenizer
    'embed_dim': 30,          # Small for testing (3 x SO(10))
    'gauge_dim': 10,          # SO(10)
    'irrep_spec': [('fund', 3, 10)],  # 3 copies of fundamental
    'n_layers': 1,
    'n_q_iterations': 5,

    # VFE weights
    'alpha': 0.1,             # Self-coupling (entropy)
    'beta': 1.0,              # Belief alignment (attention)
    'gamma': 0.1,             # Prior coupling
    'bch_order': 2,           # BCH truncation order
    'temperature': 1.0,       # Attention temperature

    # Training
    'batch_size': 8,
    'seq_len': 128,
    'learning_rate': 1e-3,
    'weight_decay': 0.01,
    'epochs': 10,
    'grad_clip': 1.0,

    # Logging
    'log_interval': 100,
    'eval_interval': 500,
    'save_interval': 1000,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def get_tokenizer():
    """Get GPT-2 tokenizer."""
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except ImportError:
        print("transformers not installed. Using dummy tokenizer.")
        return None


def get_dataset(name: str, tokenizer, seq_len: int):
    """Load dataset."""
    try:
        from datasets import load_dataset

        if name == 'wikitext-2':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        elif name == 'wikitext-103':
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
        else:
            raise ValueError(f"Unknown dataset: {name}")

        def tokenize(examples):
            return tokenizer(examples['text'], truncation=True,
                           max_length=seq_len, padding='max_length',
                           return_tensors='pt')

        train_data = dataset['train'].map(tokenize, batched=True,
                                          remove_columns=['text'])
        val_data = dataset['validation'].map(tokenize, batched=True,
                                             remove_columns=['text'])

        train_data.set_format('torch', columns=['input_ids', 'attention_mask'])
        val_data.set_format('torch', columns=['input_ids', 'attention_mask'])

        return train_data, val_data

    except ImportError:
        print("datasets not installed. Using random data.")
        return None, None


class RandomDataset(torch.utils.data.Dataset):
    """Fallback random dataset for testing."""

    def __init__(self, vocab_size: int, seq_len: int, size: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,)),
            'attention_mask': torch.ones(self.seq_len),
        }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(model, dataloader, optimizer, config, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)

        # Shift for next-token prediction
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs, targets)

        loss = outputs['loss']
        ce_loss = outputs['ce_loss']

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config['grad_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        n_batches += 1

        # Logging
        if batch_idx % config['log_interval'] == 0:
            avg_loss = total_loss / n_batches
            avg_ce = total_ce / n_batches
            ppl = math.exp(min(avg_ce, 20))  # Cap to avoid overflow
            pbar.set_postfix({
                'VFE': f'{avg_loss:.4f}',
                'CE': f'{avg_ce:.4f}',
                'PPL': f'{ppl:.1f}'
            })

    return total_loss / n_batches, total_ce / n_batches


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_ce = 0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        outputs = model(inputs, targets)

        total_loss += outputs['loss'].item()
        total_ce += outputs['ce_loss'].item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_ce = total_ce / n_batches
    ppl = math.exp(min(avg_ce, 20))

    return avg_loss, avg_ce, ppl


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train FEP Transformer')
    parser.add_argument('--dataset', type=str, default='wikitext-2',
                        choices=['wikitext-2', 'wikitext-103', 'random'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed_dim', type=int, default=30)
    parser.add_argument('--gauge_dim', type=int, default=10)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./fep_checkpoints')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Config
    config = DEFAULT_CONFIG.copy()
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['embed_dim'] = args.embed_dim
    config['gauge_dim'] = args.gauge_dim
    config['n_layers'] = args.n_layers

    # Update irrep_spec based on embed_dim and gauge_dim
    n_copies = args.embed_dim // args.gauge_dim
    config['irrep_spec'] = [('fund', n_copies, args.gauge_dim)]

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load tokenizer and data
    tokenizer = get_tokenizer()
    if tokenizer is not None:
        config['vocab_size'] = len(tokenizer)

    if args.dataset == 'random' or tokenizer is None:
        print("Using random data for testing...")
        train_data = RandomDataset(config['vocab_size'], config['seq_len'], 10000)
        val_data = RandomDataset(config['vocab_size'], config['seq_len'], 1000)
    else:
        train_data, val_data = get_dataset(args.dataset, tokenizer, config['seq_len'])
        if train_data is None:
            train_data = RandomDataset(config['vocab_size'], config['seq_len'], 10000)
            val_data = RandomDataset(config['vocab_size'], config['seq_len'], 1000)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'],
                            shuffle=False, num_workers=0)

    # Create model
    print("\n" + "=" * 60)
    print("FEP TRANSFORMER")
    print("=" * 60)
    print(f"  Vocab size: {config['vocab_size']}")
    print(f"  Embed dim: {config['embed_dim']}")
    print(f"  Gauge dim: {config['gauge_dim']} (SO({config['gauge_dim']}))")
    print(f"  Irrep spec: {config['irrep_spec']}")
    print(f"  BCH order: {config['bch_order']}")
    print(f"  VFE weights: α={config['alpha']}, β={config['beta']}, γ={config['gamma']}")
    print("=" * 60 + "\n")

    model = FEPTransformer(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        gauge_dim=config['gauge_dim'],
        irrep_spec=config['irrep_spec'],
        n_layers=config['n_layers'],
        n_q_iterations=config['n_q_iterations'],
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        bch_order=config['bch_order'],
        temperature=config['temperature'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_ppl = float('inf')

    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()

        train_loss, train_ce = train_epoch(
            model, train_loader, optimizer, config, device, epoch
        )

        val_loss, val_ce, val_ppl = evaluate(model, val_loader, device)

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch} | Time: {elapsed:.1f}s")
        print(f"  Train: VFE={train_loss:.4f}, CE={train_ce:.4f}, PPL={math.exp(min(train_ce, 20)):.1f}")
        print(f"  Valid: VFE={val_loss:.4f}, CE={val_ce:.4f}, PPL={val_ppl:.1f}")

        # Save best model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ppl': val_ppl,
                'config': config,
            }, output_dir / 'best_model.pt')
            print(f"  Saved best model (PPL={val_ppl:.1f})")

    print(f"\nTraining complete! Best validation PPL: {best_val_ppl:.1f}")


if __name__ == '__main__':
    main()
