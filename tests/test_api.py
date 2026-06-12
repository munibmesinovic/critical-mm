"""Tests for the public critical_mm.api surface."""

from __future__ import annotations

import importlib

import pytest

import critical_mm.api as api

def test_all_names_in_api_importable() -> None:
    """Every name in __all__ must be a real attribute on the api module."""
    for name in api.__all__:
        assert hasattr(api, name), f"critical_mm.api missing exported name {name!r}"

def test_api_task_is_same_object_as_base() -> None:
    """Re-exports must point at the same class object as the underlying module."""
    from critical_mm.tasks.base import Task as _BaseTask

    assert api.Task is _BaseTask

def test_api_dataset_reader_is_same_object_as_base() -> None:
    from critical_mm.datasets.base import DatasetReader as _BaseDS

    assert api.DatasetReader is _BaseDS

def test_discover_returns_five_builtin_tasks() -> None:
    """After importing the concrete task modules, all 5 land in discover_tasks()."""
    importlib.import_module("critical_mm.tasks.mortality24")
    importlib.import_module("critical_mm.tasks.aki")
    importlib.import_module("critical_mm.tasks.sepsis")
    importlib.import_module("critical_mm.tasks.los")
    importlib.import_module("critical_mm.tasks.kf")

    tasks = api.discover_tasks()
    expected = {"mortality24", "aki", "sepsis", "los", "kidney_function"}
    assert expected <= set(tasks.keys()), f"missing tasks: {expected - set(tasks)}"

def test_discover_returns_five_builtin_datasets() -> None:
    for mod in ("eicu", "hirid", "mimic_iv", "nwicu", "synthetic"):
        importlib.import_module(f"critical_mm.datasets.{mod}")

    datasets = api.discover_datasets()
    expected = {"eicu", "hirid", "miiv", "nwicu", "synthetic"}
    assert expected <= set(datasets.keys()), f"missing datasets: {expected - set(datasets)}"

def test_discover_returns_dl_and_ml_models() -> None:
    """Four canonical DL models + the 11 vendored ML models are registered."""
    pytest.importorskip(
        "torch", reason="DL model registration requires torch (critical-mm-train env)"
    )
    pytest.importorskip("sklearn", reason="ML model registration requires sklearn")
    pytest.importorskip("lightgbm", reason="LGBM registration requires lightgbm")
    importlib.import_module("critical_mm.models.dl_models")
    importlib.import_module("critical_mm.models.ml_models")

    models = api.discover_models()
    expected_dl = {"GRU", "LSTM", "TCN", "Transformer"}
    expected_ml = {
        "LGBMClassifier",
        "LGBMRegressor",
        "LogisticRegression",
        "LinearRegression",
        "ElasticNet",
        "RFClassifier",
        "SVMClassifier",
        "SVMRegressor",
        "PerceptronClassifier",
        "MLPClassifier",
        "MLPRegressor",
    }
    assert expected_dl <= set(models.keys()), f"missing DL: {expected_dl - set(models)}"
    assert expected_ml <= set(models.keys()), f"missing ML: {expected_ml - set(models)}"

def test_decorator_is_passthrough() -> None:
    """register_task returns the class unchanged -- builtins keep working."""
    from critical_mm.tasks.aki import AKI

    assert api.register_task(AKI) is AKI

def test_discover_tasks_returns_built_in_class_identities() -> None:
    """discover_tasks()['aki'] must BE the AKI class, not a re-import twin."""
    from critical_mm.tasks.aki import AKI

    assert api.discover_tasks()["aki"] is AKI
