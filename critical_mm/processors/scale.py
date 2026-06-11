"""StandardScaler — per-column z-score over a Polars LazyFrame.

Equivalent to `sklearn.preprocessing.StandardScaler` on the columns
specified in `fit(..., columns=[...])`, but the entire fit is one
`pl.LazyFrame.select(...).collect()` aggregate — no full materialisation
of the input frame.
"""

from __future__ import annotations

import json
from typing import Self

import polars as pl

from critical_mm.processors.base import Processor


class StandardScaler(Processor):
    """Per-column (x − mean) / std standardisation, fitted lazily."""

    def __init__(self) -> None:
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._fitted: bool = False

    def fit(self, X: pl.LazyFrame, *, columns: list[str]) -> Self:
        if not columns:
            raise ValueError("StandardScaler.fit: `columns` cannot be empty")
        agg_exprs: list[pl.Expr] = [pl.len().alias("__n__")]
        for col in columns:
            agg_exprs.append(pl.col(col).cast(pl.Float64).mean().alias(f"__mean_{col}__"))
            agg_exprs.append(pl.col(col).cast(pl.Float64).std(ddof=0).alias(f"__std_{col}__"))
        stats = X.select(agg_exprs).collect()
        n_rows = int(stats["__n__"][0])
        if n_rows == 0:
            raise ValueError("StandardScaler.fit: input LazyFrame has zero rows; refusing to fit")
        all_null_cols = [col for col in columns if stats[f"__mean_{col}__"][0] is None]
        if all_null_cols:
            raise ValueError(
                "StandardScaler.fit: cannot fit on columns whose values are "
                f"entirely null: {all_null_cols!r}. Drop these columns or "
                "filter the fit cohort before calling .fit()."
            )
        self._means = {col: float(stats[f"__mean_{col}__"][0]) for col in columns}
        self._stds = {col: float(stats[f"__std_{col}__"][0]) for col in columns}
        self._fitted = True
        return self

    def apply(self, X: pl.LazyFrame) -> pl.LazyFrame:
        if not self._fitted:
            raise RuntimeError(
                "StandardScaler.apply called before fit; call .fit(...) on the training frame first"
            )
        new_cols: list[pl.Expr] = []
        for col in X.collect_schema().names():
            if col in self._means:
                mean = self._means[col]
                std = self._stds[col] or 1.0
                new_cols.append(
                    ((pl.col(col).cast(pl.Float64) - mean) / std).cast(pl.Float32).alias(col)
                )
            else:
                new_cols.append(pl.col(col))
        return X.with_columns(new_cols)

    @property
    def fit_string(self) -> str:
        """JSON of (means, stds), sorted keys, 12 significant digits per float.

        12 digits gets us bit-identical strings across platforms for the
        same input within Polars's Float64 precision, while keeping the
        cache key short.
        """
        payload = {
            "means": {k: f"{v:.12g}" for k, v in sorted(self._means.items())},
            "stds": {k: f"{v:.12g}" for k, v in sorted(self._stds.items())},
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
