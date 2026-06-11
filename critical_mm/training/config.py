"""Dataclass-based training config for the CM-native trainer.

Hyperparameters flow from model_defaults_for(task)[model] (see below) plus
TrainConfig.extra_hyperparams as a dict update.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(os.environ.get("CRITICAL_MM_DATA_ROOT", Path(__file__).resolve().parents[2]))

_KNOWN_TASKS: frozenset[str] = frozenset({"sepsis", "aki", "mortality24", "los", "kidney_function"})

CLASSIFICATION_TASKS: frozenset[str] = frozenset({"sepsis", "aki", "mortality24"})
REGRESSION_TASKS: frozenset[str] = frozenset({"los", "kidney_function"})


@dataclass(frozen=True)
class TrainConfig:
    """One training cell: (task, dataset, model, seed) + paths."""

    task: str
    dataset: str
    model: str
    seed: int
    cv_repetitions: int = 5
    cv_folds: int = 5
    repetition_index: int = 0
    fold_index: int = 0
    cpu: bool = False
    debug: bool = False
    extra_hyperparams: dict[str, object] = field(default_factory=dict)
    data_root: Path = field(default_factory=lambda: REPO)

    def __post_init__(self) -> None:
        from critical_mm.registry import discover_models

        known_models = set(discover_models())
        if known_models and self.model not in known_models:
            raise ValueError(f"unknown model {self.model!r}; registered: {sorted(known_models)}")
        if self.task not in _KNOWN_TASKS:
            raise ValueError(f"unknown task {self.task!r}; valid: {sorted(_KNOWN_TASKS)}")

    @property
    def is_classification(self) -> bool:
        return self.task in CLASSIFICATION_TASKS

    @property
    def data_dir(self) -> Path:
        return self.data_root / "data" / "processed" / self.task / self.dataset

    @property
    def splits_dir(self) -> Path:
        return self.data_root / "data" / "processed" / "splits" / self.task / self.dataset

    @property
    def checkpoint_dir(self) -> Path:
        return (
            self.data_root
            / "data"
            / "checkpoints"
            / self.task
            / self.dataset
            / self.model
            / f"seed_{self.seed}"
        )


def _model_defaults(num_classes: int) -> dict[str, dict[str, object]]:
    return {
        "GRU": {"hidden_dim": 256, "layer_dim": 1, "num_classes": num_classes},
        "LSTM": {"hidden_dim": 256, "layer_dim": 1, "num_classes": num_classes},
        "TCN": {
            "num_channels": [64, 64, 64],
            "kernel_size": 7,
            "dropout": 0.0,
            "num_classes": num_classes,
        },
        "Transformer": {
            "hidden": 64,
            "heads": 2,
            "ff_hidden_mult": 4,
            "depth": 1,
            "num_classes": num_classes,
            "dropout": 0.0,
            "l1_reg": 0,
        },
    }


CLASSIFICATION_MODEL_DEFAULTS = _model_defaults(num_classes=2)
REGRESSION_MODEL_DEFAULTS = _model_defaults(num_classes=1)


def model_defaults_for(task: str) -> dict[str, dict[str, object]]:
    return (
        CLASSIFICATION_MODEL_DEFAULTS if task in CLASSIFICATION_TASKS else REGRESSION_MODEL_DEFAULTS
    )


def resolve_trainer_overrides(
    extra_hyperparams: dict[str, object], use_cuda: bool
) -> tuple[dict[str, object], int, object]:
    """Pop optional ``precision`` / ``max_epochs`` overrides from a COPY of
    ``extra_hyperparams``.

    Returns ``(cleaned_extra, max_epochs, precision)``. When neither override is
    present the defaults reproduce the historical locked-cell behaviour exactly
    (``epochs=100``; ``precision="16-mixed"`` on CUDA else ``32``), so the
    seed-42 preservation gate stays bit-exact. The popped keys do NOT reach the
    model constructor (they are not model hyperparameters).
    """
    extra = dict(extra_hyperparams)
    max_epochs = int(extra.pop("max_epochs", 100))  # type: ignore[call-overload]
    precision = extra.pop("precision", None)
    if precision is None:
        precision = "16-mixed" if use_cuda else 32
    return extra, max_epochs, precision
