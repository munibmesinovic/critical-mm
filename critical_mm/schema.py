"""Canonical interim schemas — single source of truth for all DatasetReaders."""

from __future__ import annotations

import warnings
from typing import TypeAlias

import polars as pl

Schema: TypeAlias = dict[str, pl.DataType]

_DT_UTC = pl.Datetime("us", "UTC")

LOS_CAP_HOURS: int = 168

SCHEMA_STAYS: Schema = {
    "patient_id": pl.Utf8(),
    "subject_id": pl.Utf8(),
    "stay_id": pl.Utf8(),
    "dataset": pl.Utf8(),
    "hospital_id": pl.Utf8(),
    "age": pl.Float32(),
    "sex": pl.Utf8(),
    "ethnicity": pl.Utf8(),
    "weight": pl.Float32(),
    "height": pl.Float32(),
    "admit_time": _DT_UTC,
    "discharge_time": _DT_UTC,
    "los_hours": pl.Float32(),
    "mortality_in_icu": pl.Boolean(),
    "mortality_in_hospital": pl.Boolean(),
    "mortality_30day": pl.Boolean(),
    "admission_diagnosis": pl.Utf8(),
}

NULLABLE_STAYS: frozenset[str] = frozenset(
    {
        "hospital_id",
        "ethnicity",
        "weight",
        "height",
        "mortality_in_hospital",
        "mortality_30day",
        "admission_diagnosis",
    }
)

SCHEMA_EVENTS_LONG: Schema = {
    "patient_id": pl.Utf8(),
    "stay_id": pl.Utf8(),
    "charttime": _DT_UTC,
    "concept": pl.Utf8(),
    "value": pl.Float32(),
    "unit": pl.Utf8(),
    "unit_source": pl.Utf8(),
}

NULLABLE_EVENTS_LONG: frozenset[str] = frozenset({"unit_source"})

SCHEMA_MEDS: Schema = {
    "patient_id": pl.Utf8(),
    "stay_id": pl.Utf8(),
    "starttime": _DT_UTC,
    "endtime": _DT_UTC,
    "drug": pl.Utf8(),
    "dose": pl.Float32(),
    "dose_unit": pl.Utf8(),
    "route": pl.Utf8(),
    "drug_class": pl.Utf8(),
}

NULLABLE_MEDS: frozenset[str] = frozenset({"endtime", "dose", "dose_unit", "route"})

SCHEMA_INTERVENTIONS: Schema = {
    "patient_id": pl.Utf8(),
    "stay_id": pl.Utf8(),
    "starttime": _DT_UTC,
    "endtime": _DT_UTC,
    "intervention": pl.Utf8(),
}

NULLABLE_INTERVENTIONS: frozenset[str] = frozenset({"endtime"})

SCHEMA_NOTES: Schema = {
    "patient_id": pl.Utf8(),
    "stay_id": pl.Utf8(),
    "charttime": _DT_UTC,
    "note_type": pl.Utf8(),
    "text": pl.Utf8(),
}

NULLABLE_NOTES: frozenset[str] = frozenset({"stay_id"})

SCHEMA_DIAGNOSES: Schema = {
    "patient_id": pl.Utf8(),
    "stay_id": pl.Utf8(),
    "icd_code": pl.Utf8(),
    "icd_version": pl.Utf8(),
    "diagnosis_position": pl.Int32(),
}

NULLABLE_DIAGNOSES: frozenset[str] = frozenset({"stay_id"})

SCHEMA_MICROBIO: Schema = {
    "patient_id": pl.Utf8(),
    "stay_id": pl.Utf8(),
    "charttime": _DT_UTC,
    "specimen_type": pl.Utf8(),
    "organism": pl.Utf8(),
}

NULLABLE_MICROBIO: frozenset[str] = frozenset({"organism"})

SCHEMA_ABX_DURATION: Schema = {
    "stay_id": pl.Utf8(),
    "starttime": _DT_UTC,
    "endtime": _DT_UTC,
}

NULLABLE_ABX_DURATION: frozenset[str] = frozenset({"endtime"})

TABLES: dict[str, tuple[Schema, frozenset[str]]] = {
    "stays": (SCHEMA_STAYS, NULLABLE_STAYS),
    "events_long": (SCHEMA_EVENTS_LONG, NULLABLE_EVENTS_LONG),
    "meds": (SCHEMA_MEDS, NULLABLE_MEDS),
    "interventions": (SCHEMA_INTERVENTIONS, NULLABLE_INTERVENTIONS),
    "notes": (SCHEMA_NOTES, NULLABLE_NOTES),
    "diagnoses": (SCHEMA_DIAGNOSES, NULLABLE_DIAGNOSES),
    "microbio": (SCHEMA_MICROBIO, NULLABLE_MICROBIO),
    "abx_duration": (SCHEMA_ABX_DURATION, NULLABLE_ABX_DURATION),
}

def empty_frame(table_name: str) -> pl.LazyFrame:
    """Return a zero-row LazyFrame with the canonical schema."""
    if table_name not in TABLES:
        raise ValueError(f"unknown table {table_name!r}; valid: {sorted(TABLES)}")
    schema, _ = TABLES[table_name]
    return pl.LazyFrame(schema=schema)

def validate_frame(df: pl.LazyFrame, table_name: str) -> None:
    """Raise ValueError on schema violations; warn on extra columns.

    Failure modes (each raised with a clear, machine-greppable marker):
    - "missing column": expected columns absent from df
    - "dtype mismatch": column exists but its dtype differs from canonical
    - "null in non-nullable": column declared non-nullable contains null rows

    Extra columns are not errors; they emit a UserWarning so the harmoniser
    can layer dataset-specific columns without breaking the contract.
    """
    if table_name not in TABLES:
        raise ValueError(f"unknown table {table_name!r}; valid: {sorted(TABLES)}")
    expected_schema, nullable = TABLES[table_name]
    actual_schema = df.collect_schema()

    missing = [c for c in expected_schema if c not in actual_schema]
    if missing:
        raise ValueError(f"missing column(s) in {table_name}: {missing}")

    for col, expected_dtype in expected_schema.items():
        actual_dtype = actual_schema[col]
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"dtype mismatch in {table_name}.{col}: "
                f"expected {expected_dtype}, got {actual_dtype}"
            )

    non_nullable = [c for c in expected_schema if c not in nullable]
    if non_nullable:
        null_counts = df.select(
            [pl.col(c).is_null().sum().alias(c) for c in non_nullable]
        ).collect()
        for col in non_nullable:
            n = null_counts[col][0]
            if n is not None and int(n) > 0:
                raise ValueError(f"null in non-nullable column {table_name}.{col}: {int(n)} rows")

    extra = [c for c in actual_schema if c not in expected_schema]
    if extra:
        warnings.warn(
            f"extra columns in {table_name} (not in canonical schema): {extra}",
            UserWarning,
            stacklevel=2,
        )
