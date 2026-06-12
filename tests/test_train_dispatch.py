"""Tests for the train_one dispatcher (DL vs ML routing).

End-to-end training tests require torch + a populated data/ directory; those
run in critical-mm-train, not here. These tests verify the dispatcher logic
in isolation by mocking the model lookup + preamble.
"""

from __future__ import annotations

from typing import Any

import pytest

import critical_mm.registry as _reg
from critical_mm.training import train as train_mod
from critical_mm.training.config import TrainConfig

def _cfg() -> TrainConfig:
    """A minimal valid TrainConfig (paths are sentinel; not opened by dispatch tests)."""
    return TrainConfig(task="mortality24", dataset="synthetic", model="GRU", seed=42, cpu=True)

class _DLOnlyModel:
    needs_training = True
    needs_fit = False

class _MLOnlyModel:
    needs_training = False
    needs_fit = True

class _AmbiguousModel:
    needs_training = True
    needs_fit = True

class _NeitherModel:
    needs_training = False
    needs_fit = False

def test_dispatch_routes_dl_to_train_one_dl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_reg, "discover_models", lambda: {"GRU": _DLOnlyModel})
    monkeypatch.setattr(
        train_mod, "_train_one_preamble", lambda cfg, generate_features=False: "preamble-ctx"
    )

    called: dict[str, Any] = {}

    def _fake_dl(cfg: TrainConfig, ctx: Any, model_class: type) -> dict[str, Any]:
        called["which"] = "dl"
        called["ctx"] = ctx
        called["model_class"] = model_class
        return {"status": "ok", "metadata": {}}

    def _fake_ml(cfg: TrainConfig, ctx: Any, model_class: type) -> dict[str, Any]:
        called["which"] = "ml"
        return {"status": "ok", "metadata": {}}

    monkeypatch.setattr(train_mod, "_train_one_dl", _fake_dl)
    monkeypatch.setattr(train_mod, "_train_one_ml", _fake_ml)

    result = train_mod.train_one(_cfg())
    assert result["status"] == "ok"
    assert called["which"] == "dl"
    assert called["ctx"] == "preamble-ctx"
    assert called["model_class"] is _DLOnlyModel

def test_dispatch_routes_ml_to_train_one_ml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_reg, "discover_models", lambda: {"GRU": _MLOnlyModel})

    called: dict[str, Any] = {}

    def _fake_dl(cfg: TrainConfig, ctx: Any, model_class: type) -> dict[str, Any]:
        called["which"] = "dl"
        return {"status": "ok", "metadata": {}}

    def _fake_ml(cfg: TrainConfig, model_class: type) -> dict[str, Any]:
        called["which"] = "ml"
        called["model_class"] = model_class
        return {"status": "ok", "metadata": {}}

    monkeypatch.setattr(train_mod, "_train_one_dl", _fake_dl)
    monkeypatch.setattr(train_mod, "_train_one_ml", _fake_ml)

    result = train_mod.train_one(_cfg())
    assert result["status"] == "ok"
    assert called["which"] == "ml"
    assert called["model_class"] is _MLOnlyModel

def test_dispatch_unknown_model_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_reg, "discover_models", lambda: {"OtherModel": _DLOnlyModel})
    with pytest.raises(ValueError, match="unknown model 'GRU'"):
        train_mod.train_one(_cfg())

def test_dispatch_ambiguous_wrapper_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_reg, "discover_models", lambda: {"GRU": _AmbiguousModel})
    monkeypatch.setattr(
        train_mod, "_train_one_preamble", lambda cfg, generate_features=False: "preamble-ctx"
    )
    with pytest.raises(ValueError, match=r"neither pure-ML.*nor pure-DL"):
        train_mod.train_one(_cfg())

def test_dispatch_neither_wrapper_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_reg, "discover_models", lambda: {"GRU": _NeitherModel})
    monkeypatch.setattr(
        train_mod, "_train_one_preamble", lambda cfg, generate_features=False: "preamble-ctx"
    )
    with pytest.raises(ValueError, match=r"neither pure-ML.*nor pure-DL"):
        train_mod.train_one(_cfg())

class _MicrobioReqTask:
    """Stub task class returning a microbio requirement for every dataset."""

    @staticmethod
    def __call__(*args: Any, **kwargs: Any) -> Any:
        return _MicrobioReqTask

    def required_dataset_capabilities(self, dataset: str) -> frozenset[str]:
        return frozenset({"microbio"})

class _NwicuLikeDataset:
    """Stub dataset class with no microbio capability."""

    CAPABILITIES = frozenset({"abx_duration"})

class _MiivLikeDataset:
    """Stub dataset class with the full capability set including microbio."""

    CAPABILITIES = frozenset({"microbio", "urine", "abx_duration"})

def test_dispatch_capability_check_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """train_one raises with a precise message when (task, dataset) is incompatible."""
    monkeypatch.setattr(_reg, "discover_models", lambda: {"GRU": _DLOnlyModel})
    monkeypatch.setattr(_reg, "discover_tasks", lambda: {"mortality24": _MicrobioReqTask})
    monkeypatch.setattr(_reg, "discover_datasets", lambda: {"synthetic": _NwicuLikeDataset})
    with pytest.raises(ValueError, match=r"requires dataset capabilities \['microbio'\]"):
        train_mod.train_one(_cfg())

def test_dispatch_capability_check_passes_when_satisfied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_one proceeds to dispatch when the dataset declares the required capability."""
    monkeypatch.setattr(_reg, "discover_models", lambda: {"GRU": _DLOnlyModel})
    monkeypatch.setattr(_reg, "discover_tasks", lambda: {"mortality24": _MicrobioReqTask})
    monkeypatch.setattr(_reg, "discover_datasets", lambda: {"synthetic": _MiivLikeDataset})
    monkeypatch.setattr(
        train_mod, "_train_one_preamble", lambda cfg, generate_features=False: "preamble-ctx"
    )
    monkeypatch.setattr(
        train_mod,
        "_train_one_dl",
        lambda cfg, ctx, mc: {"status": "ok", "metadata": {}},
    )
    result = train_mod.train_one(_cfg())
    assert result["status"] == "ok"

def test_dispatch_capability_check_skipped_when_task_or_dataset_unregistered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the task or dataset isn't in the registry, the capability check is
    skipped (silently). Avoids spurious errors during partial-registry test setups."""
    monkeypatch.setattr(_reg, "discover_models", lambda: {"GRU": _DLOnlyModel})
    monkeypatch.setattr(_reg, "discover_tasks", lambda: {})
    monkeypatch.setattr(_reg, "discover_datasets", lambda: {})
    monkeypatch.setattr(
        train_mod, "_train_one_preamble", lambda cfg, generate_features=False: "preamble-ctx"
    )
    monkeypatch.setattr(
        train_mod,
        "_train_one_dl",
        lambda cfg, ctx, mc: {"status": "ok", "metadata": {}},
    )
    result = train_mod.train_one(_cfg())
    assert result["status"] == "ok"

def test_real_dl_models_have_needs_training_true() -> None:
    """Smoke test: the 4 built-in DL models have needs_training=True."""
    pytest.importorskip("torch")
    from critical_mm.models.dl_models import GRUNet, LSTMNet, TemporalConvNet, Transformer

    for cls in (GRUNet, LSTMNet, TemporalConvNet, Transformer):
        assert cls.needs_training is True, f"{cls.__name__} should have needs_training=True"
        assert cls.needs_fit is False, f"{cls.__name__} should have needs_fit=False"

def test_real_ml_models_have_needs_fit_true() -> None:
    """Smoke test: the 11 built-in ML models have needs_fit=True."""
    pytest.importorskip("sklearn")
    pytest.importorskip("lightgbm")
    from critical_mm.models.ml_models import (
        ElasticNet,
        LGBMClassifier,
        LGBMRegressor,
        LinearRegression,
        LogisticRegression,
        MLPClassifier,
        MLPRegressor,
        PerceptronClassifier,
        RFClassifier,
        SVMClassifier,
        SVMRegressor,
    )

    for cls in (
        LGBMClassifier,
        LGBMRegressor,
        LogisticRegression,
        LinearRegression,
        ElasticNet,
        RFClassifier,
        SVMClassifier,
        SVMRegressor,
        PerceptronClassifier,
        MLPClassifier,
        MLPRegressor,
    ):
        assert cls.needs_fit is True, f"{cls.__name__} should have needs_fit=True"
        assert cls.needs_training is False, f"{cls.__name__} should have needs_training=False"
