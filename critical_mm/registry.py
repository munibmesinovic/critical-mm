"""Discoverable registries for tasks, datasets, and models.

Built-in implementations auto-register at module import via decorators.
External contributors register by either:

  (a) dropping a Python module under ``critical_mm/contrib/<surface>/`` that
      decorates a ``Task`` / ``DatasetReader`` / model class with the matching
      ``@register_*``; the module is auto-imported on first call to a
      ``discover_*`` helper.

  (b) decorating a class in their own pip package and importing it explicitly
      before invoking ``scripts/train.py``.

The registries are simple dict-of-classes; the decorators are passthroughs
that record the class and return it unchanged. Idempotent for the same class
(re-import does not error); re-registering a *different* class for the same
name raises ``ValueError`` so collisions surface immediately.

The ``_examples/`` namespace under ``critical_mm/contrib/`` is explicitly
skipped by auto-discovery -- those files are copy-paste templates, not live
registrations.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from critical_mm.datasets.base import DatasetReader
    from critical_mm.tasks.base import Task

_TASK_REGISTRY: dict[str, type] = {}
_DATASET_REGISTRY: dict[str, type] = {}
_MODEL_REGISTRY: dict[str, type] = {}
_MODALITY_REGISTRY: dict[str, type] = {}
_NOTE_READER_REGISTRY: dict[str, type] = {}
_NOTE_ENCODER_REGISTRY: dict[str, type] = {}
_FUSION_REGISTRY: dict[str, type] = {}

T = TypeVar("T", bound=type)

def register_task(cls: type[Task]) -> type[Task]:
    """Class decorator: register a ``Task`` subclass by its ``task_name`` ClassVar.

    Raises ``ValueError`` if a *different* class is already registered for the
    same name. Re-registering the *same* class is a no-op (safe under re-import).
    """
    name = cls.task_name
    existing = _TASK_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"task name {name!r} already registered by "
            f"{existing.__module__}.{existing.__name__}; refusing to overwrite "
            f"with {cls.__module__}.{cls.__name__}"
        )
    _TASK_REGISTRY[name] = cls
    return cls

def register_dataset(cls: type[DatasetReader]) -> type[DatasetReader]:
    """Class decorator: register a ``DatasetReader`` subclass.

    The dataset name is read via ``_peek_dataset_name`` -- either a class-level
    ``DATASET_NAME`` ClassVar (preferred when the subclass exposes it) or by
    instantiating with sentinel paths and reading the ``dataset_name`` property.
    """
    name = _peek_dataset_name(cls)
    existing = _DATASET_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"dataset name {name!r} already registered by "
            f"{existing.__module__}.{existing.__name__}; refusing to overwrite "
            f"with {cls.__module__}.{cls.__name__}"
        )
    _DATASET_REGISTRY[name] = cls
    return cls

def register_model(name: str) -> Callable[[T], T]:
    """Parametrised class decorator: ``@register_model("GRU")``.

    Model classes don't carry a self-describing name ClassVar today (the name
    is a CLI flag chosen by the wrapper hierarchy), so the name is passed
    explicitly. may consolidate.
    """

    def _decorate(cls: T) -> T:
        existing = _MODEL_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"model name {name!r} already registered by "
                f"{existing.__module__}.{existing.__name__}; refusing to "
                f"overwrite with {cls.__module__}.{cls.__name__}"
            )
        _MODEL_REGISTRY[name] = cls
        return cls

    return _decorate

def register_modality(name: str) -> Callable[[T], T]:
    """Parametrised class decorator: ``@register_modality("diagnoses")``.

    Mirrors ``register_model``: the modality name is passed explicitly.
    Re-registering the same class is a no-op; a different class for the same
    name raises ValueError.
    """

    def _decorate(cls: T) -> T:
        existing = _MODALITY_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"modality name {name!r} already registered by "
                f"{existing.__module__}.{existing.__name__}; refusing to "
                f"overwrite with {cls.__module__}.{cls.__name__}"
            )
        _MODALITY_REGISTRY[name] = cls
        return cls

    return _decorate

def get_modality(name: str) -> type:
    """Return the registered modality class for ``name`` (KeyError if absent)."""
    _ensure_builtin_modalities_loaded()
    _ensure_contrib_loaded()
    return _MODALITY_REGISTRY[name]

def _ensure_builtin_modalities_loaded() -> None:
    """Import builtin modality modules so their decorators register."""
    import contextlib

    global _BUILTIN_MODALITIES_LOADED
    if _BUILTIN_MODALITIES_LOADED:
        return
    with contextlib.suppress(ImportError):
        import critical_mm.modalities
    _BUILTIN_MODALITIES_LOADED = True

def register_note_reader(name: str) -> Callable[[T], T]:
    """Parametrised class decorator: ``@register_note_reader("notes_miiv")``.

    Mirrors ``register_modality``. Re-registering the same class is a no-op;
    a different class for the same name raises ValueError.
    """

    def _decorate(cls: T) -> T:
        existing = _NOTE_READER_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"note reader {name!r} already registered by "
                f"{existing.__module__}.{existing.__name__}; refusing to "
                f"overwrite with {cls.__module__}.{cls.__name__}"
            )
        _NOTE_READER_REGISTRY[name] = cls
        return cls

    return _decorate

def register_note_encoder(name: str) -> Callable[[T], T]:
    """Parametrised class decorator: ``@register_note_encoder("bge_large_zh")`` (Plan 2)."""

    def _decorate(cls: T) -> T:
        existing = _NOTE_ENCODER_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"note encoder {name!r} already registered by "
                f"{existing.__module__}.{existing.__name__}; refusing to "
                f"overwrite with {cls.__module__}.{cls.__name__}"
            )
        _NOTE_ENCODER_REGISTRY[name] = cls
        return cls

    return _decorate

def get_note_reader(name: str) -> type:
    """Return the registered note-reader class for ``name`` (KeyError if absent)."""
    _ensure_builtin_modalities_loaded()
    _ensure_contrib_loaded()
    return _NOTE_READER_REGISTRY[name]

def get_note_encoder(name: str) -> type:
    """Return the registered note-encoder class for ``name`` (KeyError if absent)."""
    _ensure_builtin_modalities_loaded()
    _ensure_contrib_loaded()
    return _NOTE_ENCODER_REGISTRY[name]

def register_fusion(name: str) -> Callable[[T], T]:
    """Parametrised class decorator: ``@register_fusion("feature_augmentation")``.

    Mirrors ``register_modality``. Re-registering the same class is a no-op;
    a different class for the same name raises ValueError.
    """

    def _decorate(cls: T) -> T:
        existing = _FUSION_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"fusion strategy {name!r} already registered by "
                f"{existing.__module__}.{existing.__name__}; refusing to "
                f"overwrite with {cls.__module__}.{cls.__name__}"
            )
        _FUSION_REGISTRY[name] = cls
        return cls

    return _decorate

def get_fusion(name: str) -> type:
    """Return the registered fusion-strategy class for ``name`` (KeyError if absent).

    Fusion strategies are builtin-only (under ``critical_mm.fusion``); contrib
    auto-discovery is intentionally not run here.
    """
    _ensure_builtin_fusion_loaded()
    return _FUSION_REGISTRY[name]

_BUILTIN_FUSION_LOADED = False

def _ensure_builtin_fusion_loaded() -> None:
    """Import the builtin fusion package so its decorators register."""
    import contextlib

    global _BUILTIN_FUSION_LOADED
    if _BUILTIN_FUSION_LOADED:
        return
    with contextlib.suppress(ImportError):
        import critical_mm.fusion
    _BUILTIN_FUSION_LOADED = True

def discover_tasks() -> dict[str, type[Task]]:
    """Return the task registry. Triggers built-in + contrib auto-discovery on first call."""
    _ensure_builtins_loaded()
    _ensure_contrib_loaded()
    return dict(_TASK_REGISTRY)

def discover_datasets() -> dict[str, type[DatasetReader]]:
    """Return the dataset registry. Triggers built-in + contrib auto-discovery on first call."""
    _ensure_builtins_loaded()
    _ensure_contrib_loaded()
    return dict(_DATASET_REGISTRY)

def discover_models() -> dict[str, type]:
    """Return the model registry. Triggers built-in + contrib auto-discovery on first call.

    Model registrations require torch / sklearn / lightgbm (loaded lazily). In a
    train-light env without those, the model registry stays empty (no error).
    """
    _ensure_builtins_loaded()
    _ensure_contrib_loaded()
    return dict(_MODEL_REGISTRY)

def _peek_dataset_name(cls: type[DatasetReader]) -> str:
    """Read a dataset class's name without doing real I/O.

    ``DatasetReader.__init__`` only stores the three path args -- no filesystem
    access -- so a sentinel-path instance is cheap. Subclasses that need to
    avoid construction can expose ``DATASET_NAME`` as a ClassVar; this helper
    checks for it first.
    """
    name_attr = getattr(cls, "DATASET_NAME", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    sentinel = Path("/__cmm_registry_probe__")
    instance = cls(raw_root=sentinel, interim_root=sentinel, repo_root=sentinel)
    return instance.dataset_name

_BUILTINS_LOADED = False
_CONTRIB_LOADED = False
_BUILTIN_MODALITIES_LOADED = False

def _ensure_builtins_loaded() -> None:
    """Import every built-in task / dataset / model module on first call.

    Idempotent. Errors importing optional model modules (require torch /
    sklearn / lightgbm) are swallowed -- in a train-light env without those,
    the model registry stays empty rather than raising. Task and dataset
    imports must succeed; any error there surfaces.
    """
    import contextlib

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    importlib.import_module("critical_mm.tasks")
    importlib.import_module("critical_mm.datasets")
    for mod_name in (
        "critical_mm.models.dl_models",
        "critical_mm.models.ml_models",
        "critical_mm.models.fm_models",
    ):
        with contextlib.suppress(ImportError):
            importlib.import_module(mod_name)
    _BUILTINS_LOADED = True

def _ensure_contrib_loaded() -> None:
    """Import every module under ``critical_mm.contrib.*`` on first call.

    Idempotent. Skips ``_examples`` and any submodule whose path contains
    ``._examples.`` -- those are copy-paste templates. Errors in contrib
    modules are surfaced (not swallowed) so contributors see broken-import
    diagnostics.
    """
    global _CONTRIB_LOADED
    if _CONTRIB_LOADED:
        return
    try:
        import critical_mm.contrib as contrib_pkg
    except ImportError:
        _CONTRIB_LOADED = True
        return
    for _, mod_name, _ in pkgutil.iter_modules(contrib_pkg.__path__, prefix="critical_mm.contrib."):
        if mod_name.endswith("._examples") or "._examples." in mod_name:
            continue
        importlib.import_module(mod_name)
    _CONTRIB_LOADED = True

def _reset_contrib_for_tests() -> None:
    """Test-only: clear the contrib-loaded flag so a re-scan happens on next discover.

    Does NOT clear the registries -- built-in decorators only fire when their
    modules first execute, and Python caches modules in sys.modules; clearing
    the registry without re-execution leaves it permanently empty.
    """
    global _CONTRIB_LOADED
    _CONTRIB_LOADED = False
