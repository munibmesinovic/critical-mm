"""Treatment-concept registry + the TreatmentModalityReader (Plan B).

Plan B builds a leakage-safe treatment-variables DATA layer. This module holds
two things:

1. The canonical treatment-concept registry: the CSV (configs/treatments.csv)
   is the single source of truth for the 10-concept v1 treatment set (6 core
   binary indicators + 4 enrichment dose channels). Module-level validation
   runs once at import time; the resulting registry is frozen. This mirrors
   `critical_mm/concepts.py`: strip `#`-comment lines, parse with
   `csv.DictReader`, build frozen Pydantic models, hard-assert the exact row
   count, and expose `TREATMENT_CONCEPTS` / `TREATMENT_CONCEPTS_BY_NAME` (Task 1).

2. `TreatmentModalityReader` (Task 3): a `@register_modality("treatments")`
   reader that re-emits the ALREADY-harmonised treatment tables (meds /
   interventions / abx_duration — read-only) as a leakage-anchored
   `TREATMENT_TIMED` modality, following the diagnoses-reader pattern. The
   mapping is factored into pure functions (`_meds_to_treatments`,
   `_interventions_to_treatments`, `_abx_to_treatments`) so it can be
   unit-tested on small in-memory frames without raw I/O. v1 is binary
   occupancy only (`dose`/`dose_unit` null); dose/NEE is Task 4.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from critical_mm.modalities.base import (
    TREATMENT_TIMED,
    ModalityReader,
    empty_treatment_timed,
)
from critical_mm.registry import register_modality

if TYPE_CHECKING:
    from critical_mm.datasets.base import DatasetReader

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TREATMENTS_CSV = _REPO_ROOT / "configs" / "treatments.csv"

_DT_UTC = pl.Datetime("us", "UTC")

Family = Literal["vasopressor", "ventilation", "rrt", "antibiotic", "sedation", "other"]
Support = Literal["binary", "dose"]

class TreatmentConcept(BaseModel):
    """One row of the treatment registry, as parsed from treatments.csv."""

    model_config = ConfigDict(frozen=True)

    name: str
    family: Family
    support: Support
    source: str
    datasets: str
    notes: str

def _row_to_treatment(row: dict[str, str]) -> TreatmentConcept:
    return TreatmentConcept(
        name=row["name"],
        family=row["family"],
        support=row["support"],
        source=row["source"],
        datasets=row["datasets"],
        notes=row["notes"],
    )

def _load_treatments() -> list[TreatmentConcept]:
    """Parse configs/treatments.csv into TreatmentConcept objects, skipping comments."""
    with _TREATMENTS_CSV.open() as fp:
        data_lines = [line for line in fp if not line.lstrip().startswith("#")]
    reader = csv.DictReader(data_lines)
    treatments = [_row_to_treatment(row) for row in reader]
    if len(treatments) != 10:
        raise ValueError(
            f"expected 10 treatment concepts in {_TREATMENTS_CSV.name}, got {len(treatments)}"
        )
    return treatments

TREATMENT_CONCEPTS: list[TreatmentConcept] = _load_treatments()

TREATMENT_CONCEPTS_BY_NAME: dict[str, TreatmentConcept] = {t.name: t for t in TREATMENT_CONCEPTS}

TREATMENT_CONCEPTS_BY_FAMILY: dict[str, list[TreatmentConcept]] = {}
for _t in TREATMENT_CONCEPTS:
    TREATMENT_CONCEPTS_BY_FAMILY.setdefault(_t.family, []).append(_t)

_STAY_BOUND_DATASETS: frozenset[str] = frozenset({"eicu", "omix"})

_DRUG_CLASS_TO_TREATMENT: dict[str, str] = {
    "vasopressor": "vasopressor",
    "sedative": "sedation",
}

_INTERVENTION_TREATMENTS: frozenset[str] = frozenset({"mech_vent", "niv", "rrt"})

_HIRID_RRT_VARIABLEID: int = 10002508
_SICDB_RRT_DATAID: int = 723
_OMIX_CRRT_ROUTE: str = "For CRRT"
_ZIGONG_RRT_COLUMN: str = "Hemodialysis_tube"
_RRT_RAW_DATASETS: frozenset[str] = frozenset({"hirid", "sicdb", "omix", "zigong"})

_HIRID_PEEP_VARIABLEID: int = 2600
_SICDB_PEEP_DATAID: int = 2278
_MIIV_PEEP_ITEMIDS: frozenset[int] = frozenset({220339, 224700})
_EICU_PEEP_LABEL: str = "PEEP"
_PEEP_DOSE_UNIT: str = "cmH2O"
_PEEP_CLIP_MIN: float = 0.0
_PEEP_CLIP_MAX: float = 25.0
_PEEP_RAW_DATASETS: frozenset[str] = frozenset({"hirid", "sicdb", "miiv", "eicu"})

_SICDB_NEE_DATAID: int = 773
_SICDB_GKGH_TO_UG_KG_MIN: float = 1.0e6 / 60.0

_NEE_RAW_DATASETS: frozenset[str] = frozenset({"hirid", "sicdb"})

_NEE_RAW_CLIP_MAX: float = 5.0

def _is_stay_bound(dataset: str) -> bool:
    return dataset in _STAY_BOUND_DATASETS

def _namespaced_keys(dataset: str, *, stay_bound: bool) -> list[pl.Expr]:
    """Map a harmonised table's canonical ids → the TREATMENT_TIMED key trio.

    The harmonised tables already carry canonical ``patient_id`` / ``stay_id``
    (e.g. ``miiv_p123`` / ``eicu_s456``) — the SAME ids the cohort/align layer
    uses — so this is a pass-through, not a reconstruction. ``bound_stay_id`` is
    populated only for stay-bound datasets (so stay-bound rows attach to exactly
    their own stay); patient-bound rows leave it null and attach via
    ``patient_id``. ``source_admission_id`` is the stay id (the admission a
    treatment episode belongs to).
    """
    pid = pl.col("patient_id").cast(pl.Utf8)
    sid = pl.col("stay_id").cast(pl.Utf8)
    bound = sid if stay_bound else pl.lit(None, dtype=pl.Utf8)
    return [
        pid.alias("patient_id"),
        bound.alias("bound_stay_id"),
        sid.alias("source_admission_id"),
    ]

def _finalise(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add null dose payload, drop null-knowable rows, project to schema order."""
    return (
        lf.with_columns(
            pl.lit(None, dtype=pl.Float32).alias("dose"),
            pl.lit(None, dtype=pl.Utf8).alias("dose_unit"),
        )
        .filter(pl.col("knowable_time").is_not_null())
        .select(list(TREATMENT_TIMED.keys()))
    )

def _finalise_dose(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Project a dose-channel frame to schema order.

    Unlike ``_finalise`` (binary occupancy), this preserves the already-computed
    ``dose`` / ``dose_unit`` columns. Drops rows with a null ``knowable_time`` OR
    a null ``dose`` — a dose channel exists only where a usable numeric rate/level
    was charted (presence-only rows are surfaced by the binary helpers instead).
    """
    return (
        lf.with_columns(
            pl.col("dose").cast(pl.Float32),
            pl.col("dose_unit").cast(pl.Utf8),
        )
        .filter(pl.col("knowable_time").is_not_null() & pl.col("dose").is_not_null())
        .select(list(TREATMENT_TIMED.keys()))
    )

def _meds_to_treatments(meds_df: pl.LazyFrame, *, dataset: str) -> pl.LazyFrame:
    """Map harmonised meds → vasopressor / sedation treatment intervals.

    Filters ``drug_class`` to {vasopressor, sedative} and re-emits one row per
    administration: ``start_time = knowable_time = starttime``,
    ``end_time = endtime`` (null for ongoing), ``origin = "meds"``,
    binary-occupancy (dose null). NEVER mutates the source table.
    """
    stay_bound = _is_stay_bound(dataset)
    treatment = (
        pl.col("drug_class").cast(pl.Utf8).replace_strict(_DRUG_CLASS_TO_TREATMENT, default=None)
    )
    mapped = (
        meds_df.filter(pl.col("drug_class").cast(pl.Utf8).is_in(list(_DRUG_CLASS_TO_TREATMENT)))
        .with_columns(
            *_namespaced_keys(dataset, stay_bound=stay_bound),
            treatment.alias("treatment"),
            pl.col("starttime").cast(_DT_UTC).alias("start_time"),
            pl.col("endtime").cast(_DT_UTC).alias("end_time"),
            pl.col("starttime").cast(_DT_UTC).alias("knowable_time"),
            pl.lit("meds").alias("origin"),
        )
        .filter(pl.col("treatment").is_not_null())
    )
    return _finalise(mapped)

def _interventions_to_treatments(interv_df: pl.LazyFrame, *, dataset: str) -> pl.LazyFrame:
    """Map harmonised interventions → mech_vent / niv / rrt treatment intervals.

    Re-emits the ``intervention`` value verbatim as the ``treatment`` name (for
    the v1 set), with ``start_time = knowable_time = starttime``,
    ``end_time = endtime``, ``origin = "interventions"``. NEVER mutates source.
    """
    stay_bound = _is_stay_bound(dataset)
    mapped = interv_df.filter(
        pl.col("intervention").cast(pl.Utf8).is_in(list(_INTERVENTION_TREATMENTS))
    ).with_columns(
        *_namespaced_keys(dataset, stay_bound=stay_bound),
        pl.col("intervention").cast(pl.Utf8).alias("treatment"),
        pl.col("starttime").cast(_DT_UTC).alias("start_time"),
        pl.col("endtime").cast(_DT_UTC).alias("end_time"),
        pl.col("starttime").cast(_DT_UTC).alias("knowable_time"),
        pl.lit("interventions").alias("origin"),
    )
    return _finalise(mapped)

def _abx_to_treatments(abx_df: pl.LazyFrame, *, dataset: str) -> pl.LazyFrame:
    """Map harmonised abx_duration → antibiotic treatment intervals.

    ``read_abx_duration`` carries ONLY ``stay_id`` (no patient timeline), so abx
    rows are ALWAYS stay-bound regardless of the dataset's namespacing — the
    antibiotic episode is intrinsically scoped to its stay. ``treatment =
    "antibiotic"``, abx keeps its OWN ``start_time``/``end_time``,
    ``knowable_time = start_time``, ``origin = "abx"``. NEVER mutates source.
    """
    del dataset
    sid = pl.col("stay_id").cast(pl.Utf8)
    mapped = abx_df.with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("patient_id"),
        sid.alias("bound_stay_id"),
        sid.alias("source_admission_id"),
        pl.lit("antibiotic").alias("treatment"),
        pl.col("starttime").cast(_DT_UTC).alias("start_time"),
        pl.col("endtime").cast(_DT_UTC).alias("end_time"),
        pl.col("starttime").cast(_DT_UTC).alias("knowable_time"),
        pl.lit("abx").alias("origin"),
    )
    return _finalise(mapped)

_SICDB_DEFAULT_ANCHOR = datetime(2013, 1, 1, tzinfo=UTC)
_OMIX_DEFAULT_ANCHOR = datetime(2012, 1, 1, tzinfo=UTC)
_ZIGONG_DEFAULT_ANCHOR = datetime(2019, 1, 1, tzinfo=UTC)

def _offset_to_utc(col: pl.Expr, anchor: datetime, unit: str) -> pl.Expr:
    """Offset (``unit`` ∈ {"s","day","hour"}) → UTC datetime from ``anchor``.

    Mirrors the readers' ``_offset_secs_to_time`` / ``_offset_days_to_time`` /
    ``_offset_hours_to_time`` (day/hour go via minute precision, seconds direct),
    so an RRT row lands on the SAME timeline as that dataset's other events.
    """
    base = pl.lit(anchor).cast(_DT_UTC)
    if unit == "s":
        secs = col.cast(pl.Int64)
        return base.dt.offset_by(secs.cast(pl.Utf8) + pl.lit("s"))
    if unit == "day":
        minutes = (col.cast(pl.Float64) * 1440.0).round(0).cast(pl.Int64)
        return base.dt.offset_by(minutes.cast(pl.Utf8) + pl.lit("m"))
    if unit == "hour":
        minutes = (col.cast(pl.Float64) * 60.0).round(0).cast(pl.Int64)
        return base.dt.offset_by(minutes.cast(pl.Utf8) + pl.lit("m"))
    raise ValueError(f"unknown offset unit {unit!r}")

def _rrt_rows(
    lf: pl.LazyFrame,
    *,
    patient_id: pl.Expr,
    bound_stay_id: pl.Expr,
    source_admission_id: pl.Expr,
    start_time: pl.Expr,
    end_time: pl.Expr,
) -> pl.LazyFrame:
    """Stamp the RRT treatment columns + null dose, project to TREATMENT_TIMED."""
    mapped = lf.with_columns(
        patient_id.cast(pl.Utf8).alias("patient_id"),
        bound_stay_id.cast(pl.Utf8).alias("bound_stay_id"),
        source_admission_id.cast(pl.Utf8).alias("source_admission_id"),
        pl.lit("rrt").alias("treatment"),
        start_time.cast(_DT_UTC).alias("start_time"),
        end_time.cast(_DT_UTC).alias("end_time"),
        start_time.cast(_DT_UTC).alias("knowable_time"),
        pl.lit("rrt_raw").alias("origin"),
    )
    return _finalise(mapped)

def _rrt_hirid(obs_df: pl.LazyFrame) -> pl.LazyFrame:
    """HiRID: observation rows variableid==10002508 AND value==1.0 (Ja) → RRT.

    Patient-bound (shared timeline): ``patient_id == stay_id == "hirid_<patientid>"``,
    ``bound_stay_id`` null. Point event at ``datetime`` (cast to UTC, same as the
    reader's other observation timestamps).
    """
    key = pl.lit("hirid_") + pl.col("patientid").cast(pl.Utf8)
    t = pl.col("datetime")
    rows = obs_df.filter(
        (pl.col("variableid") == _HIRID_RRT_VARIABLEID) & (pl.col("value").cast(pl.Float64) == 1.0)
    )
    return _rrt_rows(
        rows,
        patient_id=key,
        bound_stay_id=pl.lit(None, dtype=pl.Utf8),
        source_admission_id=key,
        start_time=t,
        end_time=t,
    )

def _rrt_sicdb(
    float_df: pl.LazyFrame,
    stay_keys: pl.LazyFrame,
    *,
    anchor_time: datetime = _SICDB_DEFAULT_ANCHOR,
) -> pl.LazyFrame:
    """SICdb: data_float_h DataID==723 AND Val>0 (CRRT bloodflow running) → RRT.

    Patient-bound. ``Offset`` (seconds from the case reference) is anchored exactly
    as the sicdb reader anchors its events (``_offset_secs_to_time``). The stay
    keys frame supplies (_caseid → patient_id/stay_id) via inner join, matching
    the reader's ``_event_join_keys`` contract; rows are bounded to the stay window.
    """
    rows = (
        float_df.filter(
            (pl.col("DataID").cast(pl.Int64) == _SICDB_RRT_DATAID)
            & (pl.col("Val").cast(pl.Float64) > 0.0)
        )
        .with_columns(pl.col("CaseID").cast(pl.Int64).alias("_caseid"))
        .join(stay_keys, on="_caseid", how="inner")
        .filter(
            (pl.col("Offset").cast(pl.Int64) >= pl.col("admit_secs"))
            & (pl.col("Offset").cast(pl.Int64) <= pl.col("discharge_secs"))
        )
    )
    t = _offset_to_utc(pl.col("Offset"), anchor_time, "s")
    return _rrt_rows(
        rows,
        patient_id=pl.col("patient_id"),
        bound_stay_id=pl.lit(None, dtype=pl.Utf8),
        source_admission_id=pl.col("stay_id"),
        start_time=t,
        end_time=t,
    )

def _rrt_omix(
    med_df: pl.LazyFrame,
    stay_keys: pl.LazyFrame,
    *,
    anchor_time: datetime = _OMIX_DEFAULT_ANCHOR,
) -> pl.LazyFrame:
    """OMIX: Medication rows Med_route_Eng=="For CRRT" → RRT interval.

    Stay-bound (every admission independently re-anchored): ``bound_stay_id ==
    stay_id``. Interval ``[Med_startTime, Med_stopTime]`` (Float64 DAYS from
    admission) → UTC via the reader's day-offset anchor. Joined on
    (patient_SN, Hospital_ID) → the reader's stay keys, bounded to the stay window.
    """
    rows = (
        med_df.filter(pl.col("Med_route_Eng").cast(pl.Utf8) == _OMIX_CRRT_ROUTE)
        .join(
            stay_keys,
            left_on=["patient_SN", "Hospital_ID"],
            right_on=["original_patient_sn", "original_hospital_id"],
            how="inner",
        )
        .filter(
            (pl.col("Med_startTime").cast(pl.Float64) >= pl.col("admit_days"))
            & (pl.col("Med_startTime").cast(pl.Float64) <= pl.col("discharge_days"))
        )
    )
    start = _offset_to_utc(pl.col("Med_startTime"), anchor_time, "day")
    stop = _offset_to_utc(pl.col("Med_stopTime"), anchor_time, "day")
    return _rrt_rows(
        rows,
        patient_id=pl.col("patient_id"),
        bound_stay_id=pl.col("stay_id"),
        source_admission_id=pl.col("stay_id"),
        start_time=start,
        end_time=stop,
    )

def _rrt_zigong(
    nursing_df: pl.LazyFrame,
    stay_keys: pl.LazyFrame,
    *,
    anchor_time: datetime = _ZIGONG_DEFAULT_ANCHOR,
) -> pl.LazyFrame:
    """Zigong: dtNursingChart Hemodialysis_tube populated → RRT.

    Patient-bound. ``ChartTime`` (hours from hospital admission) anchored via the
    reader's hour-offset anchor; joined on INP_NO → the reader's stay keys, bounded
    to the stay window. "Populated" = not null, not "" and not "NA" after strip.
    """
    tube = pl.col(_ZIGONG_RRT_COLUMN).cast(pl.Utf8).str.strip_chars()
    rows = (
        nursing_df.filter(tube.is_not_null() & (tube != "") & (tube != "NA"))
        .with_columns(pl.col("INP_NO").cast(pl.Utf8).alias("original_inp_no"))
        .join(stay_keys, on="original_inp_no", how="inner")
        .filter(
            (pl.col("ChartTime").cast(pl.Float64) >= pl.col("admit_hours"))
            & (pl.col("ChartTime").cast(pl.Float64) <= pl.col("disch_hours"))
        )
    )
    t = _offset_to_utc(pl.col("ChartTime"), anchor_time, "hour")
    return _rrt_rows(
        rows,
        patient_id=pl.col("patient_id"),
        bound_stay_id=pl.lit(None, dtype=pl.Utf8),
        source_admission_id=pl.col("stay_id"),
        start_time=t,
        end_time=t,
    )

def _clip_peep(value: pl.Expr) -> pl.Expr:
    """Cast → Float64, NULL out anything outside [0, 25] cmH2O (sentinels).

    Returns a Float64 ``dose`` expr that is null for sub-/supra-physiological
    rows; ``_finalise_dose`` then drops those null-dose rows. We do NOT winsorise
    onto the boundary — a 3.2e10 sentinel must NOT become a 25 cmH2O datapoint.
    """
    v = value.cast(pl.Float64, strict=False)
    return pl.when((v >= _PEEP_CLIP_MIN) & (v <= _PEEP_CLIP_MAX)).then(v).otherwise(None)

def _peep_rows(
    lf: pl.LazyFrame,
    *,
    patient_id: pl.Expr,
    bound_stay_id: pl.Expr,
    source_admission_id: pl.Expr,
    chart_time: pl.Expr,
    peep_value: pl.Expr,
) -> pl.LazyFrame:
    """Stamp the PEEP dose columns (clipped) + project to TREATMENT_TIMED."""
    mapped = lf.with_columns(
        patient_id.cast(pl.Utf8).alias("patient_id"),
        bound_stay_id.cast(pl.Utf8).alias("bound_stay_id"),
        source_admission_id.cast(pl.Utf8).alias("source_admission_id"),
        pl.lit("peep").alias("treatment"),
        chart_time.cast(_DT_UTC).alias("start_time"),
        chart_time.cast(_DT_UTC).alias("end_time"),
        chart_time.cast(_DT_UTC).alias("knowable_time"),
        _clip_peep(peep_value).alias("dose"),
        pl.lit(_PEEP_DOSE_UNIT).alias("dose_unit"),
        pl.lit("peep_raw").alias("origin"),
    )
    return _finalise_dose(mapped)

def _peep_hirid(obs_df: pl.LazyFrame) -> pl.LazyFrame:
    """HiRID: observation rows variableid==2600 ("PEEP setting") → PEEP cmH2O.

    Patient-bound (shared timeline): ``patient_id == source_admission_id ==
    "hirid_<patientid>"``, ``bound_stay_id`` null. Point event at ``datetime``
    (already a UTC datetime, same base as the reader's other observations). Has a
    gross sentinel (max ~3.2e10) → ``_clip_peep`` keeps only [0, 25].
    """
    key = pl.lit("hirid_") + pl.col("patientid").cast(pl.Utf8)
    rows = obs_df.filter(pl.col("variableid") == _HIRID_PEEP_VARIABLEID)
    return _peep_rows(
        rows,
        patient_id=key,
        bound_stay_id=pl.lit(None, dtype=pl.Utf8),
        source_admission_id=key,
        chart_time=pl.col("datetime"),
        peep_value=pl.col("value"),
    )

def _peep_sicdb(
    float_df: pl.LazyFrame,
    stay_keys: pl.LazyFrame,
    *,
    anchor_time: datetime = _SICDB_DEFAULT_ANCHOR,
) -> pl.LazyFrame:
    """SICdb: data_float_h DataID==2278 ("PEEP", mbar≈cmH2O) → PEEP cmH2O.

    Patient-bound. ``Offset`` (seconds from the case reference) is anchored
    exactly as the sicdb reader anchors its events (``_offset_secs_to_time``).
    The stay keys frame supplies (_caseid → patient_id/stay_id) via inner join
    (the reader's ``_event_join_keys`` contract); rows are bounded to the stay
    window. mbar is treated as cmH2O (≈1:1) per the verified source note.
    """
    rows = (
        float_df.filter(pl.col("DataID").cast(pl.Int64) == _SICDB_PEEP_DATAID)
        .with_columns(pl.col("CaseID").cast(pl.Int64).alias("_caseid"))
        .join(stay_keys, on="_caseid", how="inner")
        .filter(
            (pl.col("Offset").cast(pl.Int64) >= pl.col("admit_secs"))
            & (pl.col("Offset").cast(pl.Int64) <= pl.col("discharge_secs"))
        )
    )
    t = _offset_to_utc(pl.col("Offset"), anchor_time, "s")
    return _peep_rows(
        rows,
        patient_id=pl.col("patient_id"),
        bound_stay_id=pl.lit(None, dtype=pl.Utf8),
        source_admission_id=pl.col("stay_id"),
        chart_time=t,
        peep_value=pl.col("Val"),
    )

def _peep_miiv(chart_df: pl.LazyFrame) -> pl.LazyFrame:
    """MIMIC-IV: chartevents itemid in {220339, 224700} → PEEP cmH2O.

    Patient-bound (matches the miiv reader's event namespacing):
    ``patient_id == "miiv_<subject_id>"``, ``source_admission_id ==
    "miiv_<stay_id>"``, ``bound_stay_id`` null. ``charttime`` is a STRING in the
    raw csv, parsed to UTC exactly as the reader parses its chart timestamps
    (``str.to_datetime`` → ``replace_time_zone("UTC")``). Value = ``valuenum``.
    """
    pid = pl.lit("miiv_") + pl.col("subject_id").cast(pl.Utf8)
    sid = pl.lit("miiv_") + pl.col("stay_id").cast(pl.Utf8)
    t = (
        pl.col("charttime")
        .str.to_datetime(time_unit="us", strict=False)
        .dt.replace_time_zone("UTC")
    )
    rows = chart_df.filter(pl.col("itemid").cast(pl.Int64).is_in(list(_MIIV_PEEP_ITEMIDS)))
    return _peep_rows(
        rows,
        patient_id=pid,
        bound_stay_id=pl.lit(None, dtype=pl.Utf8),
        source_admission_id=sid,
        chart_time=t,
        peep_value=pl.col("valuenum"),
    )

def _peep_eicu(
    resp_df: pl.LazyFrame,
    pid_map: pl.LazyFrame,
    *,
    anchor_time: datetime,
) -> pl.LazyFrame:
    """eICU: respiratoryCharting respchartvaluelabel=='PEEP' → PEEP cmH2O.

    Stay-bound (per the eicu reader's namespacing): ``patient_id ==
    "eicu_<uniquepid>"``, ``bound_stay_id == source_admission_id ==
    "eicu_<patientunitstayid>"``. ``respchartvalue`` is a STRING → cast to Float64
    (non-numeric → null → dropped). ``respchartoffset`` (minutes from unit admit)
    is anchored exactly as the reader's ``_offset_to_time``. Has a 400 sentinel →
    ``_clip_peep`` keeps only [0, 25]. ``pid_map`` is the reader's
    ``_patient_pid_map`` (patientunitstayid → uniquepid).
    """
    pid = pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)
    sid = pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)
    t = (
        pl.lit(anchor_time)
        .cast(_DT_UTC)
        .dt.offset_by(pl.col("respchartoffset").cast(pl.Int64).cast(pl.Utf8) + pl.lit("m"))
    )
    rows = resp_df.filter(pl.col("respchartvaluelabel").cast(pl.Utf8) == _EICU_PEEP_LABEL).join(
        pid_map, on="patientunitstayid", how="left"
    )
    return _peep_rows(
        rows,
        patient_id=pid,
        bound_stay_id=sid,
        source_admission_id=sid,
        chart_time=t,
        peep_value=pl.col("respchartvalue").cast(pl.Utf8).str.strip_chars(),
    )

def _clip_nee_raw(rate: pl.Expr) -> pl.Expr:
    """Cast → Float64, NULL out non-positive / non-finite / > 5 µg/kg/min rows.

    Returns a Float64 ``dose`` expr that is null for the absurd / artefactual
    reconstructed rates; ``_finalise_dose`` then drops those null-dose rows. We do
    NOT winsorise onto the boundary — a 13 µg/kg/min must NOT become a 5 datapoint.
    """
    v = rate.cast(pl.Float64, strict=False)
    return pl.when(v.is_finite() & (v > 0.0) & (v <= _NEE_RAW_CLIP_MAX)).then(v).otherwise(None)

def _nee_raw_rows(
    lf: pl.LazyFrame,
    *,
    patient_id: pl.Expr,
    bound_stay_id: pl.Expr,
    source_admission_id: pl.Expr,
    chart_time: pl.Expr,
    nee_rate: pl.Expr,
) -> pl.LazyFrame:
    """Stamp the NEE dose columns (clipped) + project to TREATMENT_TIMED."""
    mapped = lf.with_columns(
        patient_id.cast(pl.Utf8).alias("patient_id"),
        bound_stay_id.cast(pl.Utf8).alias("bound_stay_id"),
        source_admission_id.cast(pl.Utf8).alias("source_admission_id"),
        pl.lit("vasopressor_nee").alias("treatment"),
        chart_time.cast(_DT_UTC).alias("start_time"),
        chart_time.cast(_DT_UTC).alias("end_time"),
        chart_time.cast(_DT_UTC).alias("knowable_time"),
        _clip_nee_raw(nee_rate).alias("dose"),
        pl.lit(_NEE_DOSE_UNIT).alias("dose_unit"),
        pl.lit("nee_raw").alias("origin"),
    )
    return _finalise_dose(mapped)

def _nee_sicdb(
    float_df: pl.LazyFrame,
    stay_keys: pl.LazyFrame,
    *,
    anchor_time: datetime = _SICDB_DEFAULT_ANCHOR,
) -> pl.LazyFrame:
    """SICdb: data_float_h DataID 773 ("NorepinephrinPerHourWeight", g/kg/h) → NEE.

    773 is ALREADY a weight-normalised norepinephrine infusion rate; the NEE is
    just the unit conversion g/kg/h → µg/kg/min (× 1e6 / 60) with the norepinephrine
    factor 1.0 (single drug — the bulk of NEE). Patient-bound: ``Offset`` (seconds
    from the case reference) is anchored exactly as the sicdb reader anchors its
    events (``_offset_secs_to_time``). The stay keys frame supplies (_caseid →
    patient_id/stay_id) via inner join (the reader's ``_event_join_keys`` contract);
    rows are bounded to the stay window. Val ≤ 0 (no infusion) → no NEE; the clip
    drops the rare > 5 µg/kg/min reconstruction artefacts.
    """
    rows = (
        float_df.filter(
            (pl.col("DataID").cast(pl.Int64) == _SICDB_NEE_DATAID)
            & (pl.col("Val").cast(pl.Float64) > 0.0)
        )
        .with_columns(pl.col("CaseID").cast(pl.Int64).alias("_caseid"))
        .join(stay_keys, on="_caseid", how="inner")
        .filter(
            (pl.col("Offset").cast(pl.Int64) >= pl.col("admit_secs"))
            & (pl.col("Offset").cast(pl.Int64) <= pl.col("discharge_secs"))
        )
    )
    t = _offset_to_utc(pl.col("Offset"), anchor_time, "s")
    nee = pl.col("Val").cast(pl.Float64) * _SICDB_GKGH_TO_UG_KG_MIN
    return _nee_raw_rows(
        rows,
        patient_id=pl.col("patient_id"),
        bound_stay_id=pl.lit(None, dtype=pl.Utf8),
        source_admission_id=pl.col("stay_id"),
        chart_time=t,
        nee_rate=nee,
    )

_HIRID_MASS_TO_UG: tuple[tuple[str, float], ...] = (
    ("mg", 1000.0),
    ("µg", 1.0),
    ("ug", 1.0),
    ("mcg", 1.0),
)

def _nee_hirid(
    pharma_df: pl.LazyFrame,
    weight_keys: pl.LazyFrame,
) -> pl.LazyFrame:
    """HiRID: reconstruct the NEE rate from pharma_records per-interval givendose.

    HiRID records no infusion rate — ``givendose`` is the mass delivered since the
    previous record WITHIN an ``infusionid``. So the rate is
    ``givendose_µg / Δgivenat_min / weight_kg`` (the first record of an infusion has
    ``givendose==0`` → no preceding Δt → no rate, which is correct). Per vasopressor
    ``pharmaid`` the generic NEE agent + factor is resolved via the SAME
    ``_HIRID_PHARMAID_TO_NEE`` / ``_NEE_FACTOR`` maps as the meds path (inotropes →
    factor None → no row; non-vasopressor pharmaids → no row). Vasopressin's "U"
    givendose is a unit count, not a mass: U/min → ×2.5 µg/kg/min-equiv (the
    existing convention), then ÷ weight.

    Patient-bound (HiRID is single-stay-per-patient): ``patient_id ==
    source_admission_id == "hirid_<patientid>"``, ``bound_stay_id`` null. ``weight_keys``
    is the (patientid → ``_weight_kg``) map the reader's weight observation supplies;
    a patient with no weight gets a null divisor → null NEE → dropped (fail-safe).
    The point event lands at the interval's END ``givenat`` (same base as the
    reader's other pharma timestamps). The clip drops > 5 µg/kg/min artefacts.
    """
    pid_to_drug = (
        pl.col("pharmaid")
        .cast(pl.Int64, strict=False)
        .replace_strict(_HIRID_PHARMAID_TO_NEE, default=None)
    )
    unit_norm = pl.col("doseunit").cast(pl.Utf8).str.to_lowercase().str.strip_chars()
    mass_ug = pl.lit(None, dtype=pl.Float64)
    for spelling, mult in _HIRID_MASS_TO_UG:
        mass_ug = (
            pl.when(unit_norm == spelling)
            .then(pl.col("givendose").cast(pl.Float64) * mult)
            .otherwise(mass_ug)
        )

    base = (
        pharma_df.with_columns(pid_to_drug.alias("_nee_drug"))
        .filter(pl.col("_nee_drug").is_not_null())
        .with_columns(pl.col("patientid").cast(pl.Int64, strict=False).alias("_pid"))
        .join(
            weight_keys.with_columns(
                pl.col("patientid").cast(pl.Int64, strict=False).alias("_pid")
            ),
            on="_pid",
            how="left",
        )
        .with_columns(pl.col("givenat").cast(_DT_UTC).alias("_t"))
        .sort(["_pid", "infusionid", "_t"])
        .with_columns(
            (
                (pl.col("_t") - pl.col("_t").shift(1).over(["_pid", "infusionid"]))
                .dt.total_seconds()
                .cast(pl.Float64)
                / 60.0
            ).alias("_dt_min")
        )
        .filter(pl.col("_dt_min").is_not_null() & (pl.col("_dt_min") > 0.0))
    )

    weight = pl.col("_weight_kg").cast(pl.Float64)
    mass_rate = mass_ug / pl.col("_dt_min") / weight
    vaso_rate = (
        (pl.col("givendose").cast(pl.Float64) / pl.col("_dt_min"))
        / weight
        * _VASOPRESSIN_UMIN_TO_UG_KG_MIN
    )

    nee = pl.lit(None, dtype=pl.Float64)
    for drug, factor in _NEE_FACTOR.items():
        if factor is None:
            continue
        per_min = vaso_rate if drug == "vasopressin" else mass_rate
        nee = pl.when(pl.col("_nee_drug") == drug).then(per_min * factor).otherwise(nee)

    return _nee_raw_rows(
        base,
        patient_id=pl.lit("hirid_") + pl.col("_pid").cast(pl.Utf8),
        bound_stay_id=pl.lit(None, dtype=pl.Utf8),
        source_admission_id=pl.lit("hirid_") + pl.col("_pid").cast(pl.Utf8),
        chart_time=pl.col("_t"),
        nee_rate=nee,
    )

_NEE_FACTOR: dict[str, float | None] = {
    "norepinephrine": 1.0,
    "epinephrine": 1.0,
    "dopamine": 1.0 / 100.0,
    "phenylephrine": 1.0 / 10.0,
    "vasopressin": 1.0,
    "dobutamine": None,
    "milrinone": None,
}

_NEE_DRUGS: frozenset[str] = frozenset(_NEE_FACTOR)

_NEE_NAME_ALIASES: tuple[tuple[str, str], ...] = (
    ("norepinephrin", "norepinephrine"),
    ("epinephrin", "epinephrine"),
    ("phenylephrin", "phenylephrine"),
    ("dobutamin", "dobutamine"),
    ("dopamin", "dopamine"),
    ("vasopressin", "vasopressin"),
)

_HIRID_PHARMAID_TO_NEE: dict[int, str] = {
    1000462: "norepinephrine",
    1000656: "norepinephrine",
    1000657: "norepinephrine",
    1000658: "norepinephrine",
    71: "epinephrine",
    1000649: "epinephrine",
    1000650: "epinephrine",
    1000655: "epinephrine",
    1000750: "epinephrine",
    426: "dobutamine",
    112: "vasopressin",
    113: "vasopressin",
}

_VASOPRESSIN_UMIN_TO_UG_KG_MIN: float = 2.5

_NEE_DOSE_UNIT: str = "ug/kg/min"

_UG_KG_MIN_UNITS: frozenset[str] = frozenset({"ug/kg/min", "mcg/kg/min"})
_UG_MIN_UNITS: frozenset[str] = frozenset({"ug/min", "mcg/min"})
_U_HR_UNITS: frozenset[str] = frozenset({"u/hr", "u/h", "units/hr", "units/h"})
_U_MIN_UNITS: frozenset[str] = frozenset({"u/min", "units/min"})

def _rate_to_ug_kg_min(
    drug: str,
    rate: pl.Expr,
    unit_norm: pl.Expr,
    weight: float | pl.Expr,
) -> pl.Expr:
    """Normalise a charted infusion rate to µg/kg/min (or its NEE-equivalent).

    Branches on the (lower-cased) ``dose_unit``:
      - already µg/kg/min  -> pass through;
      - µg/min             -> ÷ admission weight (kg);
      - vasopressin U/hr   -> ÷60 (U/min) then × the U/min→µg/kg/min equivalence;
      - vasopressin U/min  -> × the U/min→µg/kg/min equivalence.
    Any other unit (a non-rate bolus unit) yields null -> no NEE row downstream.
    ``weight`` may be a scalar (unit-test) or a per-row expression (real wiring).
    """
    weight_expr = weight if isinstance(weight, pl.Expr) else pl.lit(weight)
    branches = (
        pl.when(unit_norm.is_in(list(_UG_KG_MIN_UNITS)))
        .then(rate)
        .when(unit_norm.is_in(list(_UG_MIN_UNITS)))
        .then(rate / weight_expr)
    )
    if drug == "vasopressin":
        branches = (
            branches.when(unit_norm.is_in(list(_U_HR_UNITS)))
            .then((rate / 60.0) * _VASOPRESSIN_UMIN_TO_UG_KG_MIN)
            .when(unit_norm.is_in(list(_U_MIN_UNITS)))
            .then(rate * _VASOPRESSIN_UMIN_TO_UG_KG_MIN)
        )
    return branches.otherwise(None)

def _nee_canonical_drug(dataset: str) -> pl.Expr:
    """Resolve the raw ``drug`` string → a generic NEE agent name (or null).

    Dataset-aware so the NEE matcher (which keys on the generic names in
    ``_NEE_FACTOR``) can recognise drug identities that those generic names
    cannot match directly:

    - HiRID: ``drug = "pharmaid_<id>"`` (no name) → map the pharmaid → generic
      via ``_HIRID_PHARMAID_TO_NEE`` (applied ONLY for the hirid dataset).
    - everything else (incl. SICdb's German names): a lowercased-substring alias
      pass — first ``_NEE_NAME_ALIASES`` stem found in the name wins (longest
      first), else the already-generic English name passes through unchanged.

    The factors are untouched; this only widens what the matcher RECOGNISES.
    """
    drug_norm = pl.col("drug").cast(pl.Utf8).str.to_lowercase().str.strip_chars()
    if dataset == "hirid":
        pid = drug_norm.str.extract(r"pharmaid_(\d+)", 1).cast(pl.Int64, strict=False)
        return pid.replace_strict(_HIRID_PHARMAID_TO_NEE, default=None)
    canon = pl.lit(None, dtype=pl.Utf8)
    for stem, generic in _NEE_NAME_ALIASES:
        canon = (
            pl.when(canon.is_not_null())
            .then(canon)
            .when(drug_norm.str.contains(stem, literal=True))
            .then(pl.lit(generic))
            .otherwise(canon)
        )
    return pl.when(canon.is_not_null()).then(canon).otherwise(drug_norm)

def _join_admission_weight(meds_view: pl.LazyFrame, stays_view: pl.LazyFrame) -> pl.LazyFrame:
    """Left-join the per-stay admission ``weight`` (kg) onto the meds view.

    The NEE µg/min→µg/kg/min normalisation needs the admission weight. We pull it
    from the canonical ``stays`` table (read-only) and attach it as a private
    ``_weight_kg`` column keyed on ``stay_id`` (the meds' own stay). Rows whose
    stay has a null/absent weight get a null divisor, so any µg/min vasopressor
    on such a stay yields a null NEE and is dropped — fail-safe, never wrong.
    """
    weight = stays_view.select(
        pl.col("stay_id").cast(pl.Utf8),
        pl.col("weight").cast(pl.Float64).alias("_weight_kg"),
    )
    return meds_view.join(weight, on="stay_id", how="left")

def _vasopressor_nee(
    meds_view: pl.LazyFrame,
    *,
    weight: float | pl.Expr,
    dataset: str,
) -> pl.LazyFrame:
    """Pure: derive the per-administration norepinephrine-equivalent dose (NEE).

    Reads the READ-ONLY ``meds`` view only (``drug`` name + ``dose`` rate +
    ``dose_unit``) — ``drug_class`` is NEVER mutated. For each vasopressor
    administration the charted rate is normalised to µg/kg/min and multiplied by
    the drug's potency factor; inotropes (dobutamine/milrinone) and rows without
    a usable rate (null/non-rate unit) emit NO ``vasopressor_nee`` row. Emits one
    ``TREATMENT_TIMED`` row per usable administration with ``treatment =
    "vasopressor_nee"``, ``dose = NEE (µg/kg/min)``, ``dose_unit = "ug/kg/min"``,
    ``origin = "meds"``.
    """
    stay_bound = _is_stay_bound(dataset)
    drug_norm = _nee_canonical_drug(dataset)
    unit_norm = pl.col("dose_unit").cast(pl.Utf8).str.to_lowercase().str.strip_chars()
    rate = pl.col("dose").cast(pl.Float64)

    nee = pl.lit(None, dtype=pl.Float64)
    for drug, factor in _NEE_FACTOR.items():
        if factor is None:
            continue
        contribution = _rate_to_ug_kg_min(drug, rate, unit_norm, weight) * factor
        nee = pl.when(drug_norm == drug).then(contribution).otherwise(nee)

    mapped = meds_view.filter(drug_norm.is_in(list(_NEE_DRUGS))).with_columns(
        *_namespaced_keys(dataset, stay_bound=stay_bound),
        pl.lit("vasopressor_nee").alias("treatment"),
        pl.col("starttime").cast(_DT_UTC).alias("start_time"),
        pl.col("endtime").cast(_DT_UTC).alias("end_time"),
        pl.col("starttime").cast(_DT_UTC).alias("knowable_time"),
        _clip_nee_raw(nee).alias("dose"),
        pl.lit(_NEE_DOSE_UNIT).alias("dose_unit"),
        pl.lit("meds").alias("origin"),
    )
    return _finalise_dose(mapped)

def _sedation_rate_to_treatments(meds_view: pl.LazyFrame, *, dataset: str) -> pl.LazyFrame:
    """Pure: surface the sedative INFUSION RATE as the ``sedation_rate`` channel.

    Reads ``drug_class == "sedative"`` rows from the read-only ``meds`` view and
    re-emits the charted ``dose``/``dose_unit`` verbatim (no unit conversion —
    sedative rate conventions are agent-specific) as ``treatment =
    "sedation_rate"``, ``origin = "meds"``. Rows without a populated dose (the
    binary-occupancy administrations) drop out in ``_finalise_dose``.
    """
    stay_bound = _is_stay_bound(dataset)
    mapped = meds_view.filter(pl.col("drug_class").cast(pl.Utf8) == "sedative").with_columns(
        *_namespaced_keys(dataset, stay_bound=stay_bound),
        pl.lit("sedation_rate").alias("treatment"),
        pl.col("starttime").cast(_DT_UTC).alias("start_time"),
        pl.col("endtime").cast(_DT_UTC).alias("end_time"),
        pl.col("starttime").cast(_DT_UTC).alias("knowable_time"),
        pl.col("dose").alias("dose"),
        pl.col("dose_unit").alias("dose_unit"),
        pl.lit("meds").alias("origin"),
    )
    return _finalise_dose(mapped)

def _insulin_to_treatments(meds_view: pl.LazyFrame, *, dataset: str) -> pl.LazyFrame:
    """Pure: surface the insulin dose as the ``insulin`` channel (drug-name match).

    Insulin is not a distinct harmonised ``drug_class``, so identity comes from a
    case-insensitive substring on the ``drug`` NAME ("insulin", "Insulin -
    Regular", …) in the read-only ``meds`` view. ``drug_class`` is NOT consulted
    or mutated. Re-emits the charted ``dose``/``dose_unit`` verbatim as
    ``treatment = "insulin"``, ``origin = "meds"``; un-dosed rows drop out.
    """
    stay_bound = _is_stay_bound(dataset)
    is_insulin = (
        pl.col("drug").cast(pl.Utf8).str.to_lowercase().str.contains("insulin", literal=True)
    )
    mapped = meds_view.filter(is_insulin).with_columns(
        *_namespaced_keys(dataset, stay_bound=stay_bound),
        pl.lit("insulin").alias("treatment"),
        pl.col("starttime").cast(_DT_UTC).alias("start_time"),
        pl.col("endtime").cast(_DT_UTC).alias("end_time"),
        pl.col("starttime").cast(_DT_UTC).alias("knowable_time"),
        pl.col("dose").alias("dose"),
        pl.col("dose_unit").alias("dose_unit"),
        pl.lit("meds").alias("origin"),
    )
    return _finalise_dose(mapped)

@register_modality("treatments")
class TreatmentModalityReader(ModalityReader):
    """Re-emit the harmonised treatment tables as a TREATMENT_TIMED modality.

    Generic core (Task 3): for a registered ``DatasetReader`` it reads the three
    harmonised treatment tables (``read_meds`` / ``read_interventions`` /
    ``read_abx_duration`` — read-only) and maps them through the pure helpers to
    binary-occupancy treatment intervals. ``read_timed`` unions the three
    streams plus the (currently deferred) per-dataset RRT-from-raw surfacing.
    """

    MODALITY_NAME: ClassVar[str] = "treatments"

    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    def _dataset_reader(self, dataset: str) -> DatasetReader | None:
        """Instantiate the registered ``DatasetReader`` for ``dataset``.

        Resolves the DATASET-SPECIFIC raw root via ``raw_path`` (e.g. zigong ->
        ``data/raw/zigong/DataTables``, not the generic ``data/raw`` — the per-
        dataset readers scan files under their own raw dir). The registry keys on
        ``miiv`` while ``raw_path`` keys on ``mimic_iv``, so that alias is mapped
        here. The synthetic (and any path-less) reader ignores its path args and
        generates in-memory, so when ``raw_path`` has no entry / the dir is absent
        we fall back to the generic raw root — fully offline for the synthetic
        fixture. Returns ``None`` when no reader is registered for the name.
        """
        from critical_mm.io.paths import data_root, raw_path
        from critical_mm.registry import discover_datasets

        registry = discover_datasets()
        cls = registry.get(dataset)
        if cls is None:
            return None
        root = data_root()
        raw_key = "mimic_iv" if dataset == "miiv" else dataset
        try:
            raw_root = raw_path(raw_key)
        except (ValueError, FileNotFoundError):
            raw_root = root / "raw"
        return cls(
            raw_root=raw_root,
            interim_root=root / "interim",
            repo_root=self.repo_root,
        )

    def read_timed(self, dataset: str) -> pl.LazyFrame:
        reader = self._dataset_reader(dataset)
        if reader is None:
            return empty_treatment_timed()
        meds = reader.read_meds()
        meds_w = _join_admission_weight(meds, reader.read_stays())
        parts = [
            _meds_to_treatments(meds, dataset=dataset),
            _interventions_to_treatments(reader.read_interventions(), dataset=dataset),
            _abx_to_treatments(reader.read_abx_duration(), dataset=dataset),
            self._rrt_from_raw(dataset, reader=reader),
            _vasopressor_nee(meds_w, weight=pl.col("_weight_kg"), dataset=dataset),
            _sedation_rate_to_treatments(meds, dataset=dataset),
            _insulin_to_treatments(meds, dataset=dataset),
            self._peep_from_raw(dataset, reader=reader),
            self._nee_from_raw(dataset, reader=reader),
        ]
        return pl.concat(parts, how="vertical_relaxed").select(list(TREATMENT_TIMED.keys()))

    def _rrt_from_raw(self, dataset: str, *, reader: DatasetReader | None = None) -> pl.LazyFrame:
        """Per-dataset RRT-from-raw surfacing (the current plan, Item 1).

        RRT is present via ``read_interventions`` for eICU/MIMIC/NWICU. The
        remaining four datasets have no harmonised RRT intervention, so this lifts
        it from each dataset's verified raw signal (emitting ``treatment="rrt"``,
        ``origin="rrt_raw"``, dose null) WITHOUT touching ``read_interventions``:

        - HiRID:  observation_tables variableid 10002508 == 1.0 (Ja).
        - SICdb:  data_float_h DataID 723, Val > 0 (CRRT bloodflow running).
        - OMIX:   Medication Med_route_Eng == "For CRRT" (interval).
        - Zigong: dtNursingChart Hemodialysis_tube populated.

        Datasets outside that set (incl. the synthetic fixture and any unknown
        name) return ``empty_treatment_timed()``. The reader is reused from
        ``read_timed`` to avoid a second instantiation; the parsers are the pure
        helpers above so the per-dataset PARSING is unit-tested without raw I/O.
        """
        if dataset not in _RRT_RAW_DATASETS:
            return empty_treatment_timed()
        if reader is None:
            reader = self._dataset_reader(dataset)
        if reader is None:
            return empty_treatment_timed()

        if dataset == "hirid":
            return _rrt_hirid(reader._scan_observations())
        if dataset == "sicdb":
            return _rrt_sicdb(
                reader._scan("data_float_h"),
                reader._event_join_keys(),
                anchor_time=reader.anchor_time,
            )
        if dataset == "omix":
            return _rrt_omix(
                reader._scan("Medication.csv").drop(""),
                reader._event_join_keys(),
                anchor_time=reader.anchor_time,
            )
        if dataset == "zigong":
            if not reader._source_exists("dtNursingChart.csv"):
                return empty_treatment_timed()
            return _rrt_zigong(
                reader._scan_nursing(),
                reader._event_join_keys(),
                anchor_time=reader.anchor_time,
            )
        return empty_treatment_timed()

    def _peep_from_raw(self, dataset: str, *, reader: DatasetReader | None = None) -> pl.LazyFrame:
        """Per-dataset PEEP-from-raw surfacing (the current plan, Item 3).

        PEEP (positive end-expiratory pressure) is a ventilator SETTING, not a
        harmonised meds/interventions field, so — like RRT-from-raw — it is lifted
        from each dataset's verified raw ventilator signal (emitting the dose
        channel ``treatment="peep"``, ``dose=<clipped cmH2O>``,
        ``dose_unit="cmH2O"``, ``origin="peep_raw"``) WITHOUT touching the
        harmonised tables:

        - HiRID:  observation_tables variableid 2600 ("PEEP setting", cmH2O).
        - SICdb:  data_float_h DataID 2278 ("PEEP", mbar ≈ cmH2O; Val).
        - MIMIC:  chartevents itemid {220339, 224700} (PEEP set / Total PEEP).
        - eICU:   respiratoryCharting respchartvaluelabel == "PEEP".

        Every parser clips to the physiological [0, 25] cmH2O and drops the
        out-of-range rows (the sources carry gross sentinels). Datasets outside
        the set (nwicu/omix/zigong/synthetic, any unknown name) return
        ``empty_treatment_timed()``. The reader is reused from ``read_timed``; the
        parsers are the pure helpers above so the PARSING + clipping are
        unit-tested without raw I/O.
        """
        if dataset not in _PEEP_RAW_DATASETS:
            return empty_treatment_timed()
        if reader is None:
            reader = self._dataset_reader(dataset)
        if reader is None:
            return empty_treatment_timed()

        if dataset == "hirid":
            return _peep_hirid(reader._scan_observations())
        if dataset == "sicdb":
            return _peep_sicdb(
                reader._scan("data_float_h"),
                reader._event_join_keys(),
                anchor_time=reader.anchor_time,
            )
        if dataset == "miiv":
            return _peep_miiv(
                reader._scan_csv("icu/chartevents.csv.gz").select(
                    "subject_id", "stay_id", "charttime", "itemid", "valuenum"
                )
            )
        if dataset == "eicu":
            return _peep_eicu(
                reader._scan_csv("respiratoryCharting.csv.gz").select(
                    "patientunitstayid",
                    "respchartoffset",
                    "respchartvaluelabel",
                    "respchartvalue",
                ),
                reader._patient_pid_map(),
                anchor_time=reader.anchor_time,
            )
        return empty_treatment_timed()

    def _nee_from_raw(self, dataset: str, *, reader: DatasetReader | None = None) -> pl.LazyFrame:
        """Per-dataset NEE-from-raw surfacing (the current plan, last NEE gap).

        ``vasopressor_nee`` is 0 on HiRID/SICdb because the harmonised ``meds.dose``
        carries no infusion RATE there. This lifts the NEE from each dataset's RAW
        weight-normalised rate signal (emitting the dose channel
        ``treatment="vasopressor_nee"``, ``dose=<µg/kg/min>``,
        ``dose_unit="ug/kg/min"``, ``origin="nee_raw"``) WITHOUT touching the
        harmonised tables. It is PURELY ADDITIVE: for hirid/sicdb the meds-based NEE
        yields nothing, and the distinct ``origin="nee_raw"`` never collides with the
        meds-path ``origin="meds"`` NEE on eicu/miiv (which keep that path untouched):

        - SICdb: data_float_h DataID 773 ("NorepinephrinPerHourWeight", g/kg/h →
          µg/kg/min × 1e6/60; norepinephrine factor 1.0).
        - HiRID: pharma_records — rate = givendose_µg / Δgivenat_min / weight_kg per
          vasopressor pharmaid, × the per-drug NEE factor (norepi/epi 1.0, dobutamine
          0, vasopressin via U→µg/kg/min-equiv). Weight from observation 10000400.

        Datasets outside the set (eicu/miiv/nwicu/omix/zigong/synthetic, any unknown
        name) return ``empty_treatment_timed()``. The reader is reused from
        ``read_timed``; the parsers are the pure helpers above so the reconstruction
        + clipping are unit-tested without raw I/O.
        """
        if dataset not in _NEE_RAW_DATASETS:
            return empty_treatment_timed()
        if reader is None:
            reader = self._dataset_reader(dataset)
        if reader is None:
            return empty_treatment_timed()

        if dataset == "sicdb":
            return _nee_sicdb(
                reader._scan("data_float_h"),
                reader._event_join_keys(),
                anchor_time=reader.anchor_time,
            )
        if dataset == "hirid":
            weight_keys = (
                reader._scan_observations()
                .filter(pl.col("variableid") == 10000400)
                .select(
                    pl.col("patientid"),
                    pl.col("value").cast(pl.Float64).alias("_weight_kg"),
                )
                .filter(pl.col("_weight_kg").is_not_null() & (pl.col("_weight_kg") > 0))
                .group_by("patientid")
                .agg(pl.col("_weight_kg").first())
            )
            return _nee_hirid(
                reader._scan_pharma().select(
                    "patientid", "pharmaid", "givenat", "givendose", "doseunit", "infusionid"
                ),
                weight_keys,
            )
        return empty_treatment_timed()
