"""CM-native training scaffold.

Replaces YAIB's gin-based `execute_repeated_cv` with a dataclass-configured
trainer that:

* loads locked train/val/test splits from `data/processed/splits/<task>/<ds>/`
  (written by `scripts/lock_splits.py`),
* preprocesses via the CM-native pandas+sklearn preprocessor
  (`critical_mm.training.preprocess`),
* instantiates a vendored model architecture (critical_mm.models.dl_models),
* trains via PyTorch Lightning,
* saves the checkpoint + metadata to `data/checkpoints/<task>/<ds>/<model>/seed_<n>/`.

The dependency surface here is torch + pytorch-lightning + pytorch-ignite +
lightgbm + einops (no gin, no recipys). CM-native means: the entry point,
config layer, preprocessor, checkpoint format, and integration glue are all
CM-owned. Only the model architectures themselves are vendored from YAIB
(commit 7d8c591) to keep paper-architecture parity.

Public:
* config.TrainConfig -- dataclass replacing YAIB's gin config
* train.train_one -- single (task, dataset, model, seed) cell entry
* grid.run_grid   -- top-level orchestrator across cells
"""

