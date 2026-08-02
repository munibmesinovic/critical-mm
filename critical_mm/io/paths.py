"""Resolve where CRITICAL-MM reads from and writes to.

Pure string/Path manipulation plus a single env lookup. No filesystem I/O
beyond the `.exists()` check in `raw_path`'s error branch.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA = _REPO_ROOT / "data"

_DATASET_DIRS: dict[str, str] = {
    "mimic_iv": "mimic-iv-3.1",
    "eicu": "eicu-crd-2.0",
    "hirid": "hirid-1.1.1",
    "nwicu": "nwicu-0.1.0",
    "omix": "OMIX005817",
    "sicdb": (
        "sicdb/salzburg-intensive-care-database-sicdb-a-freely-accessible"
        "-intensive-care-database-1.0.8"
    ),
    "zigong": "zigong/DataTables",
}

def data_root() -> Path:
    """Return the absolute data root, honouring `CRITICAL_MM_DATA_ROOT` if set."""
    env = os.environ.get("CRITICAL_MM_DATA_ROOT")
    if env:
        return Path(env).resolve()
    return _DEFAULT_DATA.resolve()

def raw_path(dataset: str) -> Path:
    """Return the raw-data directory for `dataset`.

    Raises `ValueError` if `dataset` is not one of the four canonical names,
    and `FileNotFoundError` (at call time, not import time) if the resolved
    directory does not exist on disk.
    """
    if dataset not in _DATASET_DIRS:
        raise ValueError(f"unknown dataset {dataset!r}; valid: {sorted(_DATASET_DIRS)}")
    path = data_root() / "raw" / _DATASET_DIRS[dataset]
    if not path.exists():
        raise FileNotFoundError(
            f"raw data for {dataset} not found at {path}; "
            f"set CRITICAL_MM_DATA_ROOT or run scripts/download_datasets.sh"
        )
    return path

def interim_path(dataset: str, table: str) -> Path:
    """Return `data_root()/interim/<dataset>/<table>.parquet`.

    Does NOT verify existence; that's the caller's responsibility.
    """
    return data_root() / "interim" / dataset / f"{table}.parquet"

def processed_path(task: str, dataset: str, filename: str) -> Path:
    """Return `data_root()/processed/<task>/<dataset>/<filename>`."""
    return data_root() / "processed" / task / dataset / filename

