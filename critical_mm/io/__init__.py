"""Stateless IO helpers (paths, lazy parquet, polars thread-pool calibration).

The submodules are pure infrastructure: `paths` resolves where data lives,
`parquet` wraps `pl.scan_parquet` / `pl.LazyFrame.sink_parquet` with safe
defaults and an atomic schema-validated write, `parallel` calibrates polars's
thread pool against the worker count.
"""

from __future__ import annotations

from critical_mm.io.parallel import calibrate_thread_pool
from critical_mm.io.parquet import scan, sink, write_with_schema
from critical_mm.io.paths import data_root, interim_path, processed_path, raw_path

__all__ = [
    "calibrate_thread_pool",
    "data_root",
    "interim_path",
    "processed_path",
    "raw_path",
    "scan",
    "sink",
    "write_with_schema",
]
