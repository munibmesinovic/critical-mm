"""ImputeFill — per-group forward-fill with training-set median fallback.

For each `(patient_id, stay_id)` group, forward-fill missing values along
`hour`. Any leading null (where forward-fill has nothing to copy from)
gets the per-column training-set median computed in `fit()`.

Matches the YAIB-models impute step: fill in-stay first, fall back to
the cohort median only where in-stay context is missing.
"""

from __future__ import annotations

import json
from typing import Self

import polars as pl

from critical_mm.processors.base import Processor

_GROUP_COLS: tuple[str, ...] = ("patient_id", "stay_id")
_ORDER_COL: str = "hour"


class ImputeFill(Processor):
    """Forward-fill missing values within each stay; fallback to median."""

    def __init__(self) -> None:
        self._medians: dict[str, float] = {}
        self._columns: list[str] = []
        self._fitted: bool = False

    def fit(self, X: pl.LazyFrame, *, columns: list[str]) -> Self:
        if not columns:
            raise ValueError("ImputeFill.fit: `columns` cannot be empty")
        agg_exprs: list[pl.Expr] = [pl.len().alias("__n__")]
        for col in columns:
            agg_exprs.append(pl.col(col).cast(pl.Float64).median().alias(f"__median_{col}__"))
        stats = X.select(agg_exprs).collect()
        if int(stats["__n__"][0]) == 0:
            raise ValueError("ImputeFill.fit: input LazyFrame has zero rows; refusing to fit")
        self._columns = list(columns)
        self._medians = {}
        for col in columns:
            value = stats[f"__median_{col}__"][0]
            self._medians[col] = float(value) if value is not None else float("nan")
        self._fitted = True
        return self

    def apply(self, X: pl.LazyFrame) -> pl.LazyFrame:
        if not self._fitted:
            raise RuntimeError(
                "ImputeFill.apply called before fit; call .fit(...) on the training frame first"
            )
        present_cols = X.collect_schema().names()
        group_cols = [c for c in _GROUP_COLS if c in present_cols]
        order_col = _ORDER_COL if _ORDER_COL in present_cols else None

        out = X
        if group_cols:
            fill_exprs: list[pl.Expr] = []
            for col in self._columns:
                if col not in present_cols:
                    continue
                if order_col is not None:
                    fwd = pl.col(col).forward_fill().over(group_cols, order_by=order_col)
                else:
                    fwd = pl.col(col).forward_fill().over(group_cols)
                fill_exprs.append(fwd.alias(col))
            if fill_exprs:
                out = out.with_columns(fill_exprs)

        median_exprs: list[pl.Expr] = []
        for col in self._columns:
            if col not in present_cols:
                continue
            median = self._medians[col]
            if median != median:
                continue
            median_exprs.append(pl.col(col).fill_null(median).alias(col))
        if median_exprs:
            out = out.with_columns(median_exprs)
        return out

    @property
    def fit_string(self) -> str:
        """JSON of `medians` only — `columns` is implied by the keys."""
        payload = {
            "medians": {
                k: ("NaN" if v != v else f"{v:.12g}") for k, v in sorted(self._medians.items())
            }
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
