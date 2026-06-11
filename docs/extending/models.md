# Extending CRITICAL-MM — Models

CRITICAL-MM ships 4 deep-learning models (GRU, LSTM, TCN, Transformer) and a set
of machine-learning models (LightGBM plus scikit-learn estimators), all wired
into the training pipeline. Pick a model name from `discover_models()` and
`python scripts/train.py --models <name>` runs it. This document covers the
wrapper contracts for adding new models.

## Two wrapper hierarchies

```
DLPredictionWrapper (in critical_mm/models/wrappers.py)
  ├── GRUNet            — @register_model("GRU")
  ├── LSTMNet           — @register_model("LSTM")
  ├── TemporalConvNet   — @register_model("TCN")
  └── Transformer       — @register_model("Transformer")

MLWrapper (in critical_mm/models/wrappers.py)
  ├── LGBMClassifier    — @register_model("LGBMClassifier")
  ├── LGBMRegressor     — @register_model("LGBMRegressor")
  ├── LogisticRegression— @register_model("LogisticRegression")
  ├── ElasticNet        — @register_model("ElasticNet")
  ├── RFClassifier      — @register_model("RFClassifier")
  └── ...               — see critical_mm/models/ml_models.py
```

## DL contract — `DLPredictionWrapper`

A DL model is a `torch.nn.Module` subclass. The trainer instantiates it as:

```python
model = ExampleNet(
    input_size=(B, T, F),   # the (batch, time, features) shape of one batch
    optimizer=Adam,
    epochs=...,
    run_mode=...,           # RunMode.classification or RunMode.regression
    **hparams,              # from model_defaults_for(task)["ExampleNet"]
)
```

So `__init__` must accept `input_size` first, forward it (and the rest) to
`super().__init__`, and read the feature dimension from `input_size[2]`.
`forward` takes a 3D tensor `(batch, time, features)` and returns **per-timestep**
logits of shape `(batch, time, num_classes)`. The wrapper reduces over time,
applies the masked loss, and handles batching, optimizer, and Lightning
integration.

Minimal example (copy the scaffold at
`critical_mm/contrib/_examples/example_model.py`):

```python
import torch.nn as nn

from critical_mm.api import register_model
from critical_mm.models._runmode import RunMode
from critical_mm.models.wrappers import DLPredictionWrapper


@register_model("ExampleNet")
class ExampleNet(DLPredictionWrapper):
    _supported_run_modes = [RunMode.classification, RunMode.regression]

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
        return self.logit(h)  # (batch, time, num_classes)
```

Hyperparameters land via
`critical_mm/training/config.py::model_defaults_for(task)[model_name]`. Add an
entry there for your model — the trainer reads `hidden_dim` / `num_classes` /
etc. from that map. For classification `num_classes=2` (binary logits over
`[neg, pos]`); for regression `num_classes=1`.

## ML contract — `MLWrapper`

ML wrappers wrap a scikit-learn-style estimator with a `fit` / `predict_proba`
/ `predict` interface. The wrapper flattens the time series to a tabular matrix
and computes metrics. See `critical_mm/models/ml_models.py` for the built-in
examples.

`train_one()` dispatches on the wrapper's class-level flags:

- `needs_training = True, needs_fit = False` → DL path (Lightning Trainer)
- `needs_training = False, needs_fit = True` → ML path (sklearn fit + metrics)

Inherit from `DLPredictionWrapper` or `MLWrapper` and the flags are set
correctly by default. The ML path computes test metrics with
`roc_auc_score` (classification) or `mean_absolute_error` (regression) and
writes them to `metadata.json` under the same `test/AUC` / `test/MAE` keys the
DL path uses.

## Activation checklist

1. Copy the scaffold to `critical_mm/contrib/<your_model>.py`.
2. Keep (or add) `@register_model("YourName")` above the class.
3. Choose a name that does not collide (the decorator raises `ValueError`).
4. Add hyperparameter defaults for your model name to
   `critical_mm/training/config.py::_model_defaults`.
5. Smoke-test on the bundled synthetic dataset:
   `python scripts/train.py --tasks mortality24 --datasets synthetic --models YourName --seeds 42`.

## Comparison against the reference

Adding a new model produces *new* cells — they do not exist until you run the
grid with your model. Compare your numbers against the shipped
`baseline_metrics.json`, which carries the across-seed mean for every baseline
and fusion cell in the reference grid.

## See also

- `critical_mm/models/wrappers.py` — `DLPredictionWrapper` and `MLWrapper`
- `critical_mm/models/dl_models.py` — the 4 wired DL models
- `critical_mm/models/ml_models.py` — the wired ML models
- `critical_mm/training/train.py::train_one` — the dispatcher
- `critical_mm/training/config.py::_model_defaults` — DL hyperparameter defaults
- `critical_mm/contrib/_examples/example_model.py` — copy-paste template
