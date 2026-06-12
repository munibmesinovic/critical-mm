"""SI unit converters for the v1 concept set.

Twelve pure functions covering the six conversion pairs in H1.7
(temperature, height, weight × 2, glucose, creatinine), wrapped by a
single dispatch `convert()` that honours identity passthrough.

The mmHg → kPa pair is deliberately NOT in the registry (H1.7 §"Decisions
about what is NOT in the registry"): every v1 pressure concept exports in
mmHg, so adding a kPa converter would be dead code with no test surface
and would invite future ambiguity about which scale a pressure column is on.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import polars as pl

Numeric = TypeVar("Numeric", float, "pl.Series")

_OZ_TO_G = 28.3495
_LB_TO_KG = 0.4535924
_IN_TO_CM = 2.54
_GLU_DIVISOR = 18.0182
_CREA_FACTOR = 88.4

def fahrenheit_to_celsius(value: Numeric) -> Numeric:
    """°F → °C using `(F − 32) × 5/9`."""
    return (value - 32.0) * (5.0 / 9.0)

def celsius_to_fahrenheit(value: Numeric) -> Numeric:
    """°C → °F using `C × 9/5 + 32`."""
    return value * (9.0 / 5.0) + 32.0

def inches_to_cm(value: Numeric) -> Numeric:
    """in → cm using `× 2.54`."""
    return value * _IN_TO_CM

def cm_to_inches(value: Numeric) -> Numeric:
    """cm → in using `÷ 2.54`."""
    return value / _IN_TO_CM

def ounces_to_grams(value: Numeric) -> Numeric:
    """oz → g using `× 28.3495`."""
    return value * _OZ_TO_G

def grams_to_ounces(value: Numeric) -> Numeric:
    """g → oz using `÷ 28.3495`."""
    return value / _OZ_TO_G

def pounds_to_kg(value: Numeric) -> Numeric:
    """lb → kg using `× 0.4535924`."""
    return value * _LB_TO_KG

def kg_to_pounds(value: Numeric) -> Numeric:
    """kg → lb using `÷ 0.4535924`."""
    return value / _LB_TO_KG

def mg_per_dL_to_mmol_per_L_glucose(value: Numeric) -> Numeric:
    """Glucose: mg/dL → mmol/L using `÷ 18.0182`."""
    return value / _GLU_DIVISOR

def mmol_per_L_to_mg_per_dL_glucose(value: Numeric) -> Numeric:
    """Glucose: mmol/L → mg/dL using `× 18.0182`."""
    return value * _GLU_DIVISOR

def mg_per_dL_to_micromol_per_L_creatinine(value: Numeric) -> Numeric:
    """Creatinine: mg/dL → μmol/L using `× 88.4`."""
    return value * _CREA_FACTOR

def micromol_per_L_to_mg_per_dL_creatinine(value: Numeric) -> Numeric:
    """Creatinine: μmol/L → mg/dL using `÷ 88.4`."""
    return value / _CREA_FACTOR

UNIT_CONVERSIONS: dict[tuple[str, str], Callable[[Numeric], Numeric]] = {
    ("F", "C"): fahrenheit_to_celsius,
    ("C", "F"): celsius_to_fahrenheit,
    ("in", "cm"): inches_to_cm,
    ("cm", "in"): cm_to_inches,
    ("oz", "g"): ounces_to_grams,
    ("g", "oz"): grams_to_ounces,
    ("lb", "kg"): pounds_to_kg,
    ("kg", "lb"): kg_to_pounds,
    ("mg/dL_glu", "mmol/L_glu"): mg_per_dL_to_mmol_per_L_glucose,
    ("mmol/L_glu", "mg/dL_glu"): mmol_per_L_to_mg_per_dL_glucose,
    ("mg/dL_crea", "μmol/L_crea"): mg_per_dL_to_micromol_per_L_creatinine,
    ("μmol/L_crea", "mg/dL_crea"): micromol_per_L_to_mg_per_dL_creatinine,
}

def convert(value: Numeric, source_unit: str, target_unit: str) -> Numeric:
    """Convert `value` from `source_unit` to `target_unit`.

    Identity passthrough (`source_unit == target_unit`) returns `value`
    unchanged with no float ops applied — the same object is returned,
    not a copy, so callers can detect "no conversion needed" via `is`.

    Unknown `(source_unit, target_unit)` pairs raise `ValueError` rather
    than silently returning `value`. The registry is exhaustive for the
    v1 concept set; any unknown pair is a harmoniser bug.
    """
    if source_unit == target_unit:
        return value
    key = (source_unit, target_unit)
    if key not in UNIT_CONVERSIONS:
        raise ValueError(
            f"unknown unit conversion: {source_unit!r} → {target_unit!r}; "
            f"valid pairs: {sorted(UNIT_CONVERSIONS.keys())}"
        )
    return UNIT_CONVERSIONS[key](value)
