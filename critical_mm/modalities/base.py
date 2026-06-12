"""Base contract for modality readers + the timed/aligned frame schemas.

A modality reader owns the *alignment* of a non-tabular (or
cross-admission) data stream to the locked task cohorts. This module defines
the two canonical frame shapes:

- TIMED_SCHEMA: per-dataset, task-independent. One row per (code, source
  admission), carrying the earliest clinically-knowable timestamp.
- ALIGNED_SCHEMA: the per-stay leakage-safe stream after applying the
  ``knowable_time <= intime`` invariant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import polars as pl

from critical_mm.schema import Schema

_DT_UTC = pl.Datetime("us", "UTC")

TIMED_SCHEMA: Schema = {
    "patient_id": pl.Utf8(),
    "bound_stay_id": pl.Utf8(),
    "source_admission_id": pl.Utf8(),
    "code": pl.Utf8(),
    "code_system": pl.Utf8(),
    "knowable_time": _DT_UTC,
    "origin": pl.Utf8(),
}

ALIGNED_SCHEMA: Schema = {
    "stay_id": pl.Utf8(),
    "code": pl.Utf8(),
    "code_system": pl.Utf8(),
    "origin": pl.Utf8(),
    "delta_h": pl.Float64(),
    "prior_visit_idx": pl.Int32(),
}

def empty_timed() -> pl.LazyFrame:
    """Zero-row LazyFrame with the canonical timed schema."""
    return pl.LazyFrame(schema=TIMED_SCHEMA)

class ModalityReader(ABC):
    """Contract: produce a task-independent timed frame for one modality."""

    MODALITY_NAME: ClassVar[str]

    @abstractmethod
    def read_timed(self, dataset: str) -> pl.LazyFrame:
        """Return the TIMED_SCHEMA frame for ``dataset`` (or ``empty_timed()``)."""
