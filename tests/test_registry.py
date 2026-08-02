"""Tests for critical_mm.registry — decorators + discovery."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import polars as pl
import pytest

from critical_mm import registry as _registry_mod
from critical_mm.datasets.base import DatasetReader
from critical_mm.registry import (
    _ensure_contrib_loaded,
    _peek_dataset_name,
    discover_datasets,
    discover_models,
    discover_tasks,
    register_dataset,
    register_model,
    register_task,
)
from critical_mm.tasks.base import Task

@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot + restore the global registries around every test in this file.

    The registries are module-global dicts; registering throw-away test models
    (``rm_*_unique_a`` etc.) here would otherwise leak into the rest of the
    suite. That bites in the torch-less ``critical-mm`` test env, where the
    built-in DL models never register (suppressed import) so the registry is
    normally empty — a leaked non-empty-but-incomplete registry then makes
    ``TrainConfig.__post_init__`` reject valid names like ``'GRU'`` in
    downstream tests (test_train_grid). Restoring here keeps tests isolated.
    """
    saved = (
        dict(_registry_mod._TASK_REGISTRY),
        dict(_registry_mod._DATASET_REGISTRY),
        dict(_registry_mod._MODEL_REGISTRY),
    )
    yield
    for live, snap in zip(
        (
            _registry_mod._TASK_REGISTRY,
            _registry_mod._DATASET_REGISTRY,
            _registry_mod._MODEL_REGISTRY,
        ),
        saved,
    ):
        live.clear()
        live.update(snap)

def _make_fake_task(name: str) -> type[Task]:
    """Build a minimal Task subclass with a unique name."""

    class _FakeTask(Task):
        task_name: ClassVar[str] = name
        task_type: ClassVar[str] = "classification"
        prediction_horizon_hours: ClassVar[int] = 24

        def build_labels(
            self,
            base_cohort,
            events_long,
            meds,
            dataset,
            microbio=None,
            interventions=None,
            abx_duration=None,
        ):
            return pl.DataFrame()

    _FakeTask.__name__ = f"FakeTask_{name}"
    return _FakeTask

def _make_fake_dataset(name: str) -> type[DatasetReader]:
    """Build a minimal DatasetReader subclass with a unique name."""

    class _FakeDataset(DatasetReader):
        DATASET_NAME: ClassVar[str] = name

        @property
        def dataset_name(self) -> str:
            return name

        def read_stays(self):
            return pl.LazyFrame()

        def read_events_long(self, concepts):
            return pl.LazyFrame()

        def read_meds(self):
            return pl.LazyFrame()

        def read_interventions(self):
            return pl.LazyFrame()

        def read_notes(self):
            return pl.LazyFrame()

        def read_diagnoses(self):
            return pl.LazyFrame()

        def read_microbio(self):
            return pl.LazyFrame()

    _FakeDataset.__name__ = f"FakeDataset_{name}"
    return _FakeDataset

def test_register_task_returns_class_unchanged() -> None:
    cls = _make_fake_task("rt_passthrough_unique_a")
    assert register_task(cls) is cls

def test_register_task_idempotent_same_class() -> None:
    cls = _make_fake_task("rt_idempotent_unique_a")
    register_task(cls)
    register_task(cls)
    assert discover_tasks()["rt_idempotent_unique_a"] is cls

def test_register_task_rejects_name_collision() -> None:
    cls_a = _make_fake_task("rt_collision_unique_a")
    cls_b = _make_fake_task("rt_collision_unique_a")
    register_task(cls_a)
    with pytest.raises(ValueError, match="already registered"):
        register_task(cls_b)

def test_register_dataset_returns_class_unchanged() -> None:
    cls = _make_fake_dataset("rd_passthrough_unique_a")
    assert register_dataset(cls) is cls

def test_register_dataset_idempotent_same_class() -> None:
    cls = _make_fake_dataset("rd_idempotent_unique_a")
    register_dataset(cls)
    register_dataset(cls)
    assert discover_datasets()["rd_idempotent_unique_a"] is cls

def test_register_dataset_rejects_name_collision() -> None:
    cls_a = _make_fake_dataset("rd_collision_unique_a")
    cls_b = _make_fake_dataset("rd_collision_unique_a")
    register_dataset(cls_a)
    with pytest.raises(ValueError, match="already registered"):
        register_dataset(cls_b)

def test_peek_dataset_name_uses_class_var() -> None:
    cls = _make_fake_dataset("peek_via_classvar_unique")
    assert _peek_dataset_name(cls) == "peek_via_classvar_unique"

def test_peek_dataset_name_falls_back_to_instance() -> None:
    """Subclass without DATASET_NAME ClassVar — peek must still work via instance."""

    class _NoClassVarDS(DatasetReader):
        @property
        def dataset_name(self) -> str:
            return "peek_via_instance_unique"

        def read_stays(self):
            return pl.LazyFrame()

        def read_events_long(self, concepts):
            return pl.LazyFrame()

        def read_meds(self):
            return pl.LazyFrame()

        def read_interventions(self):
            return pl.LazyFrame()

        def read_notes(self):
            return pl.LazyFrame()

        def read_diagnoses(self):
            return pl.LazyFrame()

        def read_microbio(self):
            return pl.LazyFrame()

    assert _peek_dataset_name(_NoClassVarDS) == "peek_via_instance_unique"

def test_register_model_parametrised() -> None:
    @register_model("rm_param_unique_a")
    class _M:
        pass

    assert discover_models()["rm_param_unique_a"] is _M

def test_register_model_idempotent_same_class() -> None:
    class _M:
        pass

    register_model("rm_idempotent_unique_a")(_M)
    register_model("rm_idempotent_unique_a")(_M)
    assert discover_models()["rm_idempotent_unique_a"] is _M

def test_register_model_rejects_name_collision() -> None:
    class _M1:
        pass

    class _M2:
        pass

    register_model("rm_collision_unique_a")(_M1)
    with pytest.raises(ValueError, match="already registered"):
        register_model("rm_collision_unique_a")(_M2)

def test_discover_returns_copy_not_live_dict() -> None:
    cls = _make_fake_task("disc_copy_unique_a")
    register_task(cls)
    snapshot = discover_tasks()
    snapshot.pop("disc_copy_unique_a", None)
    assert "disc_copy_unique_a" in discover_tasks()

def test_ensure_contrib_loaded_idempotent() -> None:
    """Calling twice must not double-import; tested by hooking module count."""
    _ensure_contrib_loaded()
    n1 = sum(1 for m in sys.modules if m.startswith("critical_mm.contrib"))
    _ensure_contrib_loaded()
    n2 = sum(1 for m in sys.modules if m.startswith("critical_mm.contrib"))
    assert n1 == n2

def test_ensure_contrib_loaded_surfaces_import_error(tmp_path: Path) -> None:
    """A syntactically broken contrib module must surface its ImportError."""
    import critical_mm.contrib as contrib_pkg

    contrib_dir = Path(contrib_pkg.__file__).parent
    broken_path = contrib_dir / "_broken_test_module.py"
    broken_path.write_text("this is not valid python !!!\n")
    try:
        from critical_mm import registry as reg

        reg._CONTRIB_LOADED = False
        for mod in list(sys.modules):
            if mod.startswith("critical_mm.contrib."):
                del sys.modules[mod]
        with pytest.raises(SyntaxError):
            reg._ensure_contrib_loaded()
    finally:
        broken_path.unlink(missing_ok=True)
        from critical_mm import registry as reg

        reg._CONTRIB_LOADED = False
        for mod in list(sys.modules):
            if mod.startswith("critical_mm.contrib."):
                del sys.modules[mod]

def test_examples_dir_skipped(tmp_path: Path) -> None:
    """A registered class inside contrib/_examples/ must NOT appear in discover_*."""
    import critical_mm.contrib as contrib_pkg

    examples_dir = Path(contrib_pkg.__file__).parent / "_examples"
    if not examples_dir.exists():
        pytest.skip("_examples dir not yet created (created in .D)")
    probe_path = examples_dir / "_probe_test.py"
    probe_path.write_text(
        "from critical_mm.api import register_task, Task\n"
        "from typing import ClassVar\n"
        "import polars as pl\n"
        "@register_task\n"
        "class _Probe(Task):\n"
        ' task_name: ClassVar[str] = "_probe_should_not_register"\n'
        ' task_type: ClassVar[str] = "classification"\n'
        " prediction_horizon_hours: ClassVar[int] = 1\n"
        " def build_labels(self, *a, **kw): return pl.DataFrame()\n"
    )
    try:
        from critical_mm import registry as reg

        reg._CONTRIB_LOADED = False
        for mod in list(sys.modules):
            if mod.startswith("critical_mm.contrib."):
                del sys.modules[mod]
        tasks = discover_tasks()
        assert "_probe_should_not_register" not in tasks
    finally:
        probe_path.unlink(missing_ok=True)
        from critical_mm import registry as reg

        reg._CONTRIB_LOADED = False

def test_discover_tasks_includes_all_five_builtins() -> None:
    """discover_tasks() returns the 5 canonical built-in tasks without an explicit import."""
    tasks = discover_tasks()
    expected = {"mortality24", "aki", "sepsis", "los", "kidney_function"}
    assert expected <= set(tasks), f"missing builtin tasks: {expected - set(tasks)}"

def test_discover_datasets_includes_all_five_builtins() -> None:
    datasets = discover_datasets()
    expected = {"eicu", "hirid", "miiv", "nwicu", "synthetic"}
    assert expected <= set(datasets), f"missing builtin datasets: {expected - set(datasets)}"

def test_discover_tasks_returns_same_class_as_direct_import() -> None:
    """No twin classes: registry value IS the imported class."""
    from critical_mm.tasks.aki import AKI

    assert discover_tasks()["aki"] is AKI
