"""Stateful fit/apply processors — feed the per-task feature pipeline.

Each `Processor` subclass holds fitted state derived from training data
(means/stds for `StandardScaler`, fill values for impute_fill, etc.) and
exposes a deterministic `fit_string` that downstream caches mix into
their keys so refits invalidate dependents automatically.
"""

from __future__ import annotations

from critical_mm.processors.base import Processor
from critical_mm.processors.historical import HistoricalAggregator
from critical_mm.processors.impute_fill import ImputeFill
from critical_mm.processors.mask import MissingIndicator
from critical_mm.processors.scale import StandardScaler

__all__ = [
    "HistoricalAggregator",
    "ImputeFill",
    "MissingIndicator",
    "Processor",
    "StandardScaler",
]
