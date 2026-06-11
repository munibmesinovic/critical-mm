"""HistoricalAggregator — hourly resampling of events_long, ricu-compatible.

Ports `aggregate_hourly` from `critical_mm_legacy/processors/historical.py`
into the Processor ABC. Default aggregator is **median** (matches ricu's
`aggregate.id_tbl` default for numeric variables). Per-concept overrides
ship for the two canonical exceptions:

- `urine` → `sum` (volume accumulates over the window; median rate would
  systematically under-report).
- `abx` → `any` (was the drug administered in this hour at all?).

Time-bucket boundaries are clock-aligned (e.g. 19:20 floors to 19:00).
This matches ricu's convention exactly — see
`reports/ricu_aggregator_diagnosis.md` for the evidence chain. Using
elapsed-from-admission instead would shift every observation by up to
one hour and disagree with ricu by a few percent silently.

Input shape: long-form events_long (`patient_id`, `stay_id`, `charttime`,
`concept`, `value`). Output shape: wide-form one row per
`(patient_id, stay_id, hour_bucket)` with one column per fitted concept.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import reduce
from typing import Literal, Self

import polars as pl

from critical_mm.processors.base import Processor

Aggregator = Literal[
    "median",
    "mean",
    "sum",
    "any",
    "first",
    "last",
    "max",
    "min",
    "count",
    "mode",
]

PER_CONCEPT_OVERRIDES: dict[str, Aggregator] = {
    "urine": "sum",
    "abx": "any",
}


def _agg_expr(value_col: str, agg: Aggregator) -> pl.Expr:
    """Map an aggregator name to a polars aggregation expression.

    `mode` matches the legacy module's `_mode_agg` callable: most-common
    value with a deterministic tie-break to the smallest sort order.
    """
    col = pl.col(value_col)
    table: dict[str, pl.Expr] = {
        "median": col.median(),
        "mean": col.mean(),
        "sum": col.sum(),
        "any": col.cast(pl.Boolean).any(),
        "first": col.first(),
        "last": col.last(),
        "max": col.max(),
        "min": col.min(),
        "count": col.count(),
        "mode": col.drop_nulls().mode().sort().first(),
    }
    if agg not in table:
        raise ValueError(f"unknown aggregator {agg!r}; valid: {sorted(table)}")
    return table[agg]


def expected_aggregator_for(concept: str, default: Aggregator = "median") -> Aggregator:
    """Per-concept aggregator lookup; returns `default` for unknown concepts."""
    return PER_CONCEPT_OVERRIDES.get(concept, default)


class HistoricalAggregator(Processor):
    """Hourly per-(stay, concept) aggregation with ricu-compatible defaults."""

    def __init__(
        self,
        *,
        default: Aggregator = "median",
        per_concept: Mapping[str, Aggregator] | None = None,
        freq: str = "1h",
    ) -> None:
        self._default: Aggregator = default
        self._per_concept: dict[str, Aggregator] = {
            **PER_CONCEPT_OVERRIDES,
            **(per_concept or {}),
        }
        self._freq: str = freq
        self._concepts: list[str] = []
        self._fitted: bool = False

    def fit(self, X: pl.LazyFrame, *, columns: list[str]) -> Self:
        del X
        if not columns:
            raise ValueError("HistoricalAggregator.fit: `columns` cannot be empty")
        self._concepts = list(columns)
        self._fitted = True
        return self

    def apply(self, X: pl.LazyFrame) -> pl.LazyFrame:
        if not self._fitted:
            raise RuntimeError(
                "HistoricalAggregator.apply called before fit; call .fit(columns=[...]) first"
            )
        floored = X.with_columns(pl.col("charttime").dt.truncate(self._freq).alias("hour_bucket"))
        keys = ["patient_id", "stay_id", "hour_bucket"]

        frames: list[pl.LazyFrame] = []
        for concept in self._concepts:
            agg = self._per_concept.get(concept, self._default)
            subset = (
                floored.filter(pl.col("concept") == concept)
                .group_by(keys)
                .agg(_agg_expr("value", agg).alias(concept))
            )
            frames.append(subset)
        if not frames:
            return _empty_output(self._concepts)

        return reduce(
            lambda a, b: a.join(b, on=keys, how="full", coalesce=True),
            frames,
        ).sort(keys)

    @property
    def fit_string(self) -> str:
        """Sorted JSON of (default, freq, per-concept overrides, fitted columns)."""
        payload = {
            "default": self._default,
            "freq": self._freq,
            "per_concept": dict(sorted(self._per_concept.items())),
            "concepts": sorted(self._concepts),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _empty_output(concepts: list[str]) -> pl.LazyFrame:
    schema: dict[str, pl.DataType] = {
        "patient_id": pl.Utf8(),
        "stay_id": pl.Utf8(),
        "hour_bucket": pl.Datetime("us", "UTC"),
    }
    for c in concepts:
        schema[c] = pl.Float64()
    return pl.LazyFrame(schema=schema)
