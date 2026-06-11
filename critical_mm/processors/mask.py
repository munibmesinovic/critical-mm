"""MissingIndicator — emit a parallel boolean `_is_null` column per feature.

Matches YAIB-models' `use_missingness_mask=True` recipe. Apply BEFORE
ImputeFill so the indicator columns record pre-impute nullity; the
sequence (mask → impute_fill) is the canonical pipeline order.
"""

from __future__ import annotations

import json
from typing import Self

import polars as pl

from critical_mm.processors.base import Processor


class MissingIndicator(Processor):
    """Add `<col>_is_null` Boolean columns for each tracked feature."""

    def __init__(self) -> None:
        self._columns: list[str] = []
        self._fitted: bool = False

    def fit(self, X: pl.LazyFrame, *, columns: list[str]) -> Self:
        del X
        if not columns:
            raise ValueError("MissingIndicator.fit: `columns` cannot be empty")
        self._columns = list(columns)
        self._fitted = True
        return self

    def apply(self, X: pl.LazyFrame) -> pl.LazyFrame:
        if not self._fitted:
            raise RuntimeError("MissingIndicator.apply called before fit; call .fit(...) first")
        present_cols = X.collect_schema().names()
        new_cols: list[pl.Expr] = [
            pl.col(col).is_null().alias(f"{col}_is_null")
            for col in self._columns
            if col in present_cols
        ]
        if not new_cols:
            return X
        return X.with_columns(new_cols)

    @property
    def fit_string(self) -> str:
        """Sorted JSON list of tracked columns."""
        return json.dumps({"columns": sorted(self._columns)}, separators=(",", ":"))
