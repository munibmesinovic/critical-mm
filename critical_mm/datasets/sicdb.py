"""SICdbReader — harmonise SICdb (Salzburg Intensive Care Database, v1.0.8).

SICdb ships gzipped CSVs of one Austrian centre (2013-2021). All times are
INTEGER SECONDS from a MetaVision/surgery anchor (offset-0); `ICUOffset` marks
actual ICU admission and is non-zero for 55% of stays, so this reader anchors
`admit_time` at `ICUOffset` and bounds every event to `[ICUOffset, TimeOfStay]`
(offset-0 is pre-ICU surgery). Coded ids decode via `d_references`
(`ReferenceGlobalID`); labs map via an explicit whitelist (the German names have
urine/CSF/effusion twins under similar strings), vitals via `DataID`, using the
hourly `Val` directly (already means for vitals, sums for volumes).

Spec: 
Audit: reports/sicdb_dataset_audit.md · Template: critical_mm/datasets/omix.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import polars as pl

from critical_mm.datasets._sicdb_dicts import (
    SICDB_DRUG_CLASS_PATTERNS,
    SICDB_LAB_ID_TO_CONCEPT,
    SICDB_SIGNAL_ID_TO_CONCEPT,
    SICDB_UNIT_MULTIPLIERS,
    SICDB_VENT_RANGE_IDS,
)
from critical_mm.datasets.base import DatasetReader
from critical_mm.registry import register_dataset
from critical_mm.schema import TABLES, empty_frame

_NA_VALUES: list[str] = ["NA", "", "NULL", "null", "nan", "NaN"]
_SEX_MAP: dict[int, str] = {735: "M", 736: "F"}
_DEAD_ICU_STATE: int = 2215
_DEAD_HOSP_TYPE: int = 3130

@register_dataset
class SICdbReader(DatasetReader):
    """Concrete reader for SICdb v1.0.8 (University Hospital Salzburg)."""

    CAPABILITIES: ClassVar[frozenset[str]] = frozenset({"urine"})

    def __init__(
        self,
        *,
        raw_root: Path,
        interim_root: Path,
        repo_root: Path,
        anchor_time: datetime | None = None,
    ) -> None:
        super().__init__(raw_root=raw_root, interim_root=interim_root, repo_root=repo_root)
        self.anchor_time: datetime = anchor_time or datetime(2013, 1, 1, tzinfo=UTC)

    @property
    def dataset_name(self) -> str:
        return "sicdb"

    def _scan(self, name: str) -> pl.LazyFrame:
        return pl.scan_csv(
            self.raw_root / f"{name}.csv.gz",
            infer_schema_length=10_000,
            null_values=_NA_VALUES,
            ignore_errors=True,
        )

    def cache_source_paths(self, table: str) -> list[Path]:
        del table
        if not self.raw_root.exists():
            return []
        return sorted(self.raw_root.glob("*.csv.gz"))

    def _offset_secs_to_time(self, col: str) -> pl.Expr:
        secs = pl.col(col).cast(pl.Int64)
        return (
            pl.lit(self.anchor_time)
            .cast(pl.Datetime("us", "UTC"))
            .dt.offset_by(secs.cast(pl.Utf8) + pl.lit("s"))
        )

    def _stays_with_keys(self) -> pl.LazyFrame:
        """Canonical stays + (_caseid, admit_secs, discharge_secs) for child joins."""
        c = self._scan("cases").with_columns(
            pl.col("CaseID").cast(pl.Int64).alias("_caseid"),
            pl.col("ICUOffset").cast(pl.Int64).alias("admit_secs"),
            pl.col("TimeOfStay").cast(pl.Int64).alias("discharge_secs"),
        )
        return c.with_columns(
            ("sicdb_" + pl.col("PatientID").cast(pl.Utf8)).alias("patient_id"),
            pl.col("PatientID").cast(pl.Utf8).alias("subject_id"),
            ("sicdb_" + pl.col("CaseID").cast(pl.Utf8)).alias("stay_id"),
            pl.lit("sicdb").alias("dataset"),
            pl.lit("sicdb").alias("hospital_id"),
            pl.col("AgeOnAdmission").cast(pl.Float32).alias("age"),
            pl.col("Sex").cast(pl.Int64).replace_strict(_SEX_MAP, default=None).alias("sex"),
            pl.lit(None, dtype=pl.Utf8).alias("ethnicity"),
            pl.when(pl.col("WeightOnAdmission").cast(pl.Float64) > 0)
            .then(pl.col("WeightOnAdmission").cast(pl.Float64) / 1000.0)
            .otherwise(None)
            .cast(pl.Float32)
            .alias("weight"),
            pl.when(pl.col("HeightOnAdmission").cast(pl.Float64) > 0)
            .then(pl.col("HeightOnAdmission").cast(pl.Float64))
            .otherwise(None)
            .cast(pl.Float32)
            .alias("height"),
            self._offset_secs_to_time("ICUOffset").alias("admit_time"),
            self._offset_secs_to_time("TimeOfStay").alias("discharge_time"),
            pl.max_horizontal(
                (pl.col("TimeOfStay").cast(pl.Float64) - pl.col("ICUOffset").cast(pl.Float64))
                / 3600.0,
                pl.lit(0.0),
            )
            .cast(pl.Float32)
            .alias("los_hours"),
            (pl.col("DischargeState").cast(pl.Int64) == _DEAD_ICU_STATE)
            .fill_null(False)
            .alias("mortality_in_icu"),
            (pl.col("HospitalDischargeType").cast(pl.Int64) == _DEAD_HOSP_TYPE)
            .fill_null(False)
            .alias("mortality_in_hospital"),
            pl.lit(None, dtype=pl.Boolean).alias("mortality_30day"),
            pl.col("ICD10MainText").cast(pl.Utf8).alias("admission_diagnosis"),
        )

    def read_stays(self) -> pl.LazyFrame:
        return self._stays_with_keys().select(list(TABLES["stays"][0].keys()))

    def _event_join_keys(self) -> pl.LazyFrame:
        return self._stays_with_keys().select(
            ["_caseid", "patient_id", "stay_id", "admit_secs", "discharge_secs"]
        )

    def _ref(self) -> pl.LazyFrame:
        return self._scan("d_references").select(
            pl.col("ReferenceGlobalID").cast(pl.Int64).alias("_ref_id"),
            pl.col("ReferenceValue").cast(pl.Utf8).alias("_ref_name"),
            pl.col("ReferenceUnit").cast(pl.Utf8).alias("_ref_unit"),
        )

    def _finalize_event_arm(self, arm: pl.LazyFrame, time_col: str) -> pl.LazyFrame:
        """Join stay keys, bound to [admit_secs, discharge_secs], emit canonical cols.

        `arm` must carry: _caseid, concept, value, <time_col>, unit.
        """
        return (
            arm.join(self._event_join_keys(), on="_caseid", how="inner")
            .with_columns(pl.col("value").cast(pl.Float64, strict=False).alias("value"))
            .filter(
                pl.col("value").is_not_null()
                & (pl.col(time_col).cast(pl.Int64) >= pl.col("admit_secs"))
                & (pl.col(time_col).cast(pl.Int64) <= pl.col("discharge_secs"))
            )
            .with_columns(
                self._offset_secs_to_time(time_col).alias("charttime"),
                pl.col("unit").cast(pl.Utf8).alias("unit"),
                pl.col("unit").cast(pl.Utf8).alias("unit_source"),
            )
            .select(
                ["patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"]
            )
        )

    def _events_lab(self) -> pl.LazyFrame:
        arm = (
            self._scan("laboratory")
            .select(
                pl.col("CaseID").cast(pl.Int64).alias("_caseid"),
                pl.col("LaboratoryID").cast(pl.Int64).alias("_id"),
                pl.col("Offset").cast(pl.Int64).alias("_off"),
                pl.col("LaboratoryValue").alias("value"),
            )
            .filter(pl.col("_id").is_in(list(SICDB_LAB_ID_TO_CONCEPT.keys())))
            .with_columns(
                pl.col("_id").replace_strict(SICDB_LAB_ID_TO_CONCEPT, default=None).alias("concept")
            )
            .join(self._ref(), left_on="_id", right_on="_ref_id", how="left")
            .with_columns(pl.col("_ref_unit").alias("unit"))
        )
        return self._finalize_event_arm(arm, "_off")

    def _events_vital(self) -> pl.LazyFrame:
        arm = (
            self._scan("data_float_h")
            .select(
                pl.col("CaseID").cast(pl.Int64).alias("_caseid"),
                pl.col("DataID").cast(pl.Int64).alias("_id"),
                pl.col("Offset").cast(pl.Int64).alias("_off"),
                pl.col("Val").alias("value"),
            )
            .filter(pl.col("_id").is_in(list(SICDB_SIGNAL_ID_TO_CONCEPT.keys())))
            .with_columns(
                pl.col("_id")
                .replace_strict(SICDB_SIGNAL_ID_TO_CONCEPT, default=None)
                .alias("concept")
            )
            .join(self._ref(), left_on="_id", right_on="_ref_id", how="left")
            .with_columns(pl.col("_ref_unit").alias("unit"))
        )
        return self._finalize_event_arm(arm, "_off")

    def read_events_long(self, concepts: list[str] | None = None) -> pl.LazyFrame:
        del concepts
        concat = pl.concat([self._events_lab(), self._events_vital()], how="vertical_relaxed")
        converted = _apply_unit_conversion_sicdb(concat)
        return converted.with_columns(pl.col("value").cast(pl.Float32)).select(
            ["patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"]
        )

    def read_meds(self) -> pl.LazyFrame:
        m = self._scan("medication").select(
            pl.col("CaseID").cast(pl.Int64).alias("_caseid"),
            pl.col("DrugID").cast(pl.Int64).alias("_id"),
            pl.col("Offset").cast(pl.Int64).alias("_off"),
            pl.col("OffsetDrugEnd").cast(pl.Int64).alias("_offend"),
            pl.col("Amount").alias("_amt"),
        )
        return (
            m.join(self._event_join_keys(), on="_caseid", how="inner")
            .filter(
                (pl.col("_off") >= pl.col("admit_secs"))
                & (pl.col("_off") <= pl.col("discharge_secs"))
            )
            .join(self._ref(), left_on="_id", right_on="_ref_id", how="left")
            .with_columns(
                self._offset_secs_to_time("_off").alias("starttime"),
                self._offset_secs_to_time("_offend").alias("endtime"),
                pl.col("_ref_name").cast(pl.Utf8).alias("drug"),
                pl.col("_amt").cast(pl.Float32, strict=False).alias("dose"),
                pl.col("_ref_unit").cast(pl.Utf8).alias("dose_unit"),
                pl.lit(None, dtype=pl.Utf8).alias("route"),
                _classify_sicdb_drug(pl.col("_ref_name")).alias("drug_class"),
            )
            .select(list(TABLES["meds"][0].keys()))
        )

    def read_interventions(self) -> pl.LazyFrame:
        r = self._scan("data_range").select(
            pl.col("CaseID").cast(pl.Int64).alias("_caseid"),
            pl.col("DataID").cast(pl.Int64).alias("_id"),
            pl.col("Offset").cast(pl.Int64).alias("_off"),
            pl.col("OffsetEnd").cast(pl.Int64).alias("_offend"),
        )
        return (
            r.filter(pl.col("_id").is_in(list(SICDB_VENT_RANGE_IDS)))
            .join(self._event_join_keys(), on="_caseid", how="inner")
            .filter(
                (pl.col("_off") >= pl.col("admit_secs"))
                & (pl.col("_off") <= pl.col("discharge_secs"))
            )
            .with_columns(
                self._offset_secs_to_time("_off").alias("starttime"),
                self._offset_secs_to_time("_offend").alias("endtime"),
                pl.lit("mech_vent").alias("intervention"),
            )
            .select(list(TABLES["interventions"][0].keys()))
        )

    def read_diagnoses(self) -> pl.LazyFrame:
        keys = self._stays_with_keys().select(["_caseid", "patient_id", "stay_id"])
        return (
            self._scan("cases")
            .select(
                pl.col("CaseID").cast(pl.Int64).alias("_caseid"),
                pl.col("ICD10Main").cast(pl.Utf8).alias("icd_code"),
            )
            .filter(pl.col("icd_code").is_not_null())
            .join(keys, on="_caseid", how="inner")
            .with_columns(
                pl.lit("10").alias("icd_version"),
                pl.lit(1, dtype=pl.Int32).alias("diagnosis_position"),
            )
            .select(list(TABLES["diagnoses"][0].keys()))
        )

    def read_abx_duration(self) -> pl.LazyFrame:
        return (
            self.read_meds()
            .filter(pl.col("drug_class") == "antibiotic")
            .select(list(TABLES["abx_duration"][0].keys()))
        )

    def read_notes(self) -> pl.LazyFrame:
        return empty_frame("notes")

    def read_microbio(self) -> pl.LazyFrame:
        return empty_frame("microbio")

def _classify_sicdb_drug(name_col: pl.Expr) -> pl.Expr:
    low = name_col.cast(pl.Utf8).str.to_lowercase()
    expr = pl.lit("other")
    for substr, cls in reversed(SICDB_DRUG_CLASS_PATTERNS):
        expr = pl.when(low.str.contains(substr, literal=True)).then(pl.lit(cls)).otherwise(expr)
    return expr.cast(pl.Utf8)

def _apply_unit_conversion_sicdb(events: pl.LazyFrame) -> pl.LazyFrame:
    """Scale (concept, src_unit) values, relabel `unit` to canonical, clamp to valid_range."""
    from critical_mm.concepts import CONCEPTS_BY_NAME
    from critical_mm.datasets._units_helper import _VALID_RANGE_LOOKUP

    scaled = pl.col("value")
    for (concept, src_unit_lower), mult in SICDB_UNIT_MULTIPLIERS.items():
        match = (pl.col("concept") == concept) & (
            pl.col("unit").str.to_lowercase() == src_unit_lower
        )
        scaled = pl.when(match).then(pl.col("value") * mult).otherwise(scaled)

    canon_unit = {
        name: cc.canonical_unit for name, cc in CONCEPTS_BY_NAME.items() if cc.canonical_unit
    }
    events = events.with_columns(
        scaled.alias("value"),
        pl.col("concept").replace_strict(canon_unit, default="unknown").alias("unit"),
    )
    ranged = events.join(_VALID_RANGE_LOOKUP, on="concept", how="left")
    return (
        ranged.filter(
            pl.col("_cmm_range_lo").is_null()
            | (
                (pl.col("value") >= pl.col("_cmm_range_lo"))
                & (pl.col("value") <= pl.col("_cmm_range_hi"))
            )
        )
        .drop("_cmm_range_lo", "_cmm_range_hi")
        .filter(
            pl.col("patient_id").is_not_null()
            & pl.col("concept").is_not_null()
            & pl.col("value").is_not_null()
        )
    )
