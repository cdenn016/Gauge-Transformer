# Archived Training Scripts

These training scripts have been archived as part of the Phase 3 refactoring
to consolidate training infrastructure.

## Why Archived?

The functionality from these scripts has been extracted into the unified
`transformer/training/` module:

- **train_fast.py** → `training/optimizer.py` (parameter grouping logic)
- **train_standard_baseline.py** → `training/metrics.py` (metrics tracking)

## What to Use Instead

- **Primary entry point**: `transformer/train_publication.py`
- **Core trainer**: `transformer/train.py` (Trainer class)
- **Configuration**: `transformer/training/config.py` (TrainingConfig)
- **Optimizer creation**: `transformer/training/optimizer.py`
- **Metrics tracking**: `transformer/training/metrics.py`

## Migration Guide

### From train_fast.py

```python
# Old:
from transformer.train_fast import FastTrainer, FastTrainingConfig

# New:
from transformer.train import Trainer
from transformer.training import TrainingConfig
config = TrainingConfig(use_param_groups=True)  # FastTrainer behavior
```

### From train_standard_baseline.py

```python
# Old:
from transformer.train_standard_baseline import train_standard_baseline

# New:
from transformer.training import TrainingConfig, MetricsTracker
config = TrainingConfig(training_mode='standard')
```

## Date Archived

January 2026 - Phase 3 of transformer module cleanup
