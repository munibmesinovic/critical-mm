"""Concept registry — load configs/concepts_loinc.csv into typed Pydantic models.

The CSV is the single source of truth for the 56-concept v1 set. Module-level
validation runs once at import time; the resulting registry is frozen.

P21 (2026-05-16): `gcs` added under include_with_nwicu_null for B8 Sepsis-3
SOFA CNS component (MIMIC-IV + eICU wired; HiRID component-summing deferred
to Phase 2; NWICU absent in d_items).

An earlier review (2026-05-21): `urine_rate` added under include_with_nwicu_null
for the AKI urine arm. ricu's YAIB has a separate urine_rate concept (hirid
variableid 10020000, already mL/h) that the KDIGO urine arm consumes
directly. Other datasets (eicu/miiv/nwicu) emit no urine_rate events and
the AKI urine arm derives rate from the urine concept via /_tm_h. Not in
dynamic_vars and not exported in dyn.parquet -- internal AKI scoring only.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONCEPTS_CSV = _REPO_ROOT / "configs" / "concepts_loinc.csv"

Category = Literal["demographic", "vital", "lab", "medication", "intervention", "outcome"]
V1Status = Literal["include", "include_with_nwicu_null", "exclude_manual", "exclude_low_prevalence"]
NWICUStatus = Literal["mapped", "null", "low_prevalence"]

_CATEGORY_PREFIX: dict[Category, str] = {
    "demographic": "DEMOGRAPHIC",
    "vital": "VITAL",
    "lab": "LAB",
    "medication": "MED",
    "intervention": "PROC",
    "outcome": "OUTCOME",
}

class NWICUMapping(BaseModel):
    """Per-row NWICU table/itemid mapping; both fields may be None."""

    model_config = ConfigDict(frozen=True)

    table: str | None
    itemid: str | None

class Concept(BaseModel):
    """One row of the v1 concept registry, as parsed from concepts_loinc.csv."""

    model_config = ConfigDict(frozen=True)

    name: str
    category: Category
    description: str
    v1_status: V1Status
    miiv_prevalence: float | None
    eicu_prevalence: float | None
    hirid_prevalence: float | None
    nwicu_status: NWICUStatus
    nwicu_mapping: NWICUMapping | None
    canonical_unit: str | None
    valid_range: tuple[float, float] | None
    loinc_code: str | None
    notes: str

    @property
    def meds_code(self) -> str:
        """MEDS-style stream code: `{CATEGORY_PREFIX}//{name}`."""
        return f"{_CATEGORY_PREFIX[self.category]}//{self.name}"

def _opt_str(value: str) -> str | None:
    return value if value else None

def _opt_float(value: str) -> float | None:
    if not value or value == "NA":
        return None
    return float(value)

def _row_to_concept(row: dict[str, str]) -> Concept:
    valid_range: tuple[float, float] | None = None
    low, high = row["valid_range_low"], row["valid_range_high"]
    if low and high:
        valid_range = (float(low), float(high))

    nwicu_mapping: NWICUMapping | None = None
    if row["nwicu_itemid"] or row["nwicu_table"]:
        nwicu_mapping = NWICUMapping(
            table=_opt_str(row["nwicu_table"]),
            itemid=_opt_str(row["nwicu_itemid"]),
        )

    return Concept(
        name=row["name"],
        category=row["category"],
        description=row["description"],
        v1_status=row["v1_status"],
        miiv_prevalence=_opt_float(row["miiv_prevalence"]),
        eicu_prevalence=_opt_float(row["eicu_prevalence"]),
        hirid_prevalence=_opt_float(row["hirid_prevalence"]),
        nwicu_status=row["nwicu_status"],
        nwicu_mapping=nwicu_mapping,
        canonical_unit=_opt_str(row["canonical_unit"]),
        valid_range=valid_range,
        loinc_code=_opt_str(row["loinc_code"]),
        notes=row["notes"],
    )

def _load_concepts() -> list[Concept]:
    """Parse configs/concepts_loinc.csv into Concept objects, skipping comments."""
    with _CONCEPTS_CSV.open() as fp:
        data_lines = [line for line in fp if not line.lstrip().startswith("#")]
    reader = csv.DictReader(data_lines)
    concepts = [_row_to_concept(row) for row in reader]
    if len(concepts) != 56:
        raise ValueError(f"expected 56 concepts in {_CONCEPTS_CSV.name}, got {len(concepts)}")
    return concepts

CONCEPTS: list[Concept] = _load_concepts()

CONCEPTS_BY_NAME: dict[str, Concept] = {c.name: c for c in CONCEPTS}

CONCEPTS_BY_CATEGORY: dict[str, list[Concept]] = {}
for _c in CONCEPTS:
    CONCEPTS_BY_CATEGORY.setdefault(_c.category, []).append(_c)

ACTIVE_V1_CONCEPTS: list[Concept] = [
    c for c in CONCEPTS if c.v1_status in ("include", "include_with_nwicu_null")
]

NWICU_NULL_CONCEPTS: list[str] = sorted(
    c.name for c in CONCEPTS if c.v1_status == "include_with_nwicu_null"
)

EXCLUDE_MANUAL_CONCEPTS: list[str] = sorted(
    c.name for c in CONCEPTS if c.v1_status == "exclude_manual"
)

