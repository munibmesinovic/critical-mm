"""Task builders — produce YAIB-shaped sta/dyn/outc parquets per (task, dataset).

Each Task subclass declares `task_name`, `task_type`, optional outcome
bounds, and `prediction_horizon_hours`. `build()` reads the per-dataset
base cohort, runs `build_labels()` to emit outc, then layers static and
dynamic feature parquets to match the YAIB exporter contract that
and the validation oracle consume.
"""

from __future__ import annotations

from critical_mm.tasks.aki import AKI
from critical_mm.tasks.base import Task
from critical_mm.tasks.kf import KidneyFunction
from critical_mm.tasks.los import LengthOfStay
from critical_mm.tasks.mortality24 import Mortality24
from critical_mm.tasks.sepsis import Sepsis

__all__ = ["AKI", "KidneyFunction", "LengthOfStay", "Mortality24", "Sepsis", "Task"]
