"""Processor ABC — fit/apply contract for cache-aware feature transforms.

Concrete processors:
- `StandardScaler` (): per-column z-score.
- Later: `impute_fill`, `mask` (); `historical` ported from the
  legacy scaffold ().

The `fit_string` property is the canonical encoding of a fitted state.
's content-hash cache mixes it into the cache key's `extra`
component so any change to fitted parameters (different training data,
new column list) invalidates every cached downstream output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import polars as pl


class Processor(ABC):
    """Stateful fit/apply pattern over Polars LazyFrames."""

    @abstractmethod
    def fit(self, X: pl.LazyFrame, *, columns: list[str]) -> Self:
        """Capture summary stats from `X` over the named `columns`.

        Returning `self` lets callers chain `Scaler().fit(X, columns=[...]).apply(Y)`.
        Implementations should fail loudly on empty input — silently
        producing degenerate stats (zero variance, missing keys) leads
        to invisible bugs downstream.
        """

    @abstractmethod
    def apply(self, X: pl.LazyFrame) -> pl.LazyFrame:
        """Transform `X` using the fitted state.

        Columns not in the fitted set pass through unchanged. Calling
        `apply` before `fit` must raise a clear error (no silent
        passthrough of un-transformed data).
        """

    @property
    @abstractmethod
    def fit_string(self) -> str:
        """Deterministic encoding of the fitted state.

        Same fitted params (same data, same column order) MUST produce
        byte-identical strings. mixes this into cache keys.
        """

    @property
    def is_fitted(self) -> bool:
        """True once `fit` has populated state; False before."""
        return getattr(self, "_fitted", False)
