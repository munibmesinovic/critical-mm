"""Worked example: a minimal registered deep-learning model.

Copy this file to ``critical_mm/contrib/my_model.py`` (i.e. OUT of
``_examples/``, which is excluded from auto-discovery), adjust it, then run::

    python scripts/train.py --models ExampleNet --tasks mortality24 \
        --datasets synthetic --seeds 42

Contract (full version in ``docs/extending/models.md``). The trainer
instantiates the model as::

    ExampleNet(input_size=(B, T, F), optimizer=Adam, epochs=...,
               run_mode=..., **hparams)

where ``hparams`` come from
``critical_mm.training.config.model_defaults_for(task)["ExampleNet"]``. So
``__init__`` must forward ``input_size`` (and the rest) to ``super().__init__``
and read the feature dimension from ``input_size[2]``. ``forward`` returns
PER-TIMESTEP logits of shape ``(B, T, num_classes)``; the wrapper reduces them
and computes the masked loss. Register hyperparameter defaults for the model
name in ``critical_mm/training/config.py::_model_defaults``.
"""

from __future__ import annotations

from typing import ClassVar

import torch.nn as nn

from critical_mm.api import register_model
from critical_mm.models._runmode import RunMode
from critical_mm.models.wrappers import DLPredictionWrapper

@register_model("ExampleNet")
class ExampleNet(DLPredictionWrapper):
    """A single-hidden-layer MLP applied independently at every timestep."""

    _supported_run_modes: ClassVar[list[RunMode]] = [
        RunMode.classification,
        RunMode.regression,
    ]

    def __init__(self, input_size, hidden_dim, num_classes, *args, **kwargs):
        super().__init__(
            *args,
            input_size=input_size,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            **kwargs,
        )
        self.proj = nn.Linear(input_size[2], hidden_dim)
        self.act = nn.ReLU()
        self.logit = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.act(self.proj(x))
        return self.logit(h)
