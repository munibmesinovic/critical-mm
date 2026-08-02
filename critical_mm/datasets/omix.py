"""OMIXReader — harmonise OMIX005817 (Zhejiang Provincial ICU, Jin et al. 2023).

The released dataset has three structural quirks that drive this reader's design:

1. Hospital_ID is corrupted by R `write.csv` scientific-notation serialization
   on 5,682 of 8,180 admissions, collapsing them onto 4 indistinguishable
   strings. Cohort dedup drops all (patient_SN, Hospital_ID) groups with ≥2
   members (546 stays in 222 groups), leaving 7,634 distinct stays.

2. All timestamps are days-since-admission (Float64); no calendar dates. The
   reader reconstructs UTC datetimes against a deterministic `anchor_time`
   (the real admission year is unknown by design), mirroring the eICU reader.

3. The NursingChart_VitalSign + NursingChart_IO tables (post-2018 only;
   ~4,190 stays) carry hourly resolution + ventilator + invasive BP + I/O.
   Pre-2018 stays (~3,990) have only the sparse VitalSign table (~4h cadence,
   9 vital types, no vent/invasive BP/I/O). The mask-channel pipeline handles
   the era split implicitly; no explicit era flag.

Session-19 conformance + correctness pass (2026-05-28). The session-18 reader
emitted a non-canonical schema (patient_id/time/Unit_measure, eager DataFrames)
that `validate_frame`/`build_base_cohort` reject; it had never run end-to-end.
This rewrite makes every read_* emit the canonical schema as a LazyFrame
(stay_id, datetime admit/discharge/charttime, unit/unit_source) and fixes:
  - mortality_in_icu = eventual `Status=='Dead'` (NOT death-within-24h; the
    sub-day rule produced 0 mortality24 positives once the task's los>=30h
    filter dropped every sub-day death). ~394 positives over the los>=30 cohort.
  - events bounded to [admit, discharge] per stay (was: only time>=0; leaked
    post-discharge / post-death rows as features).
  - sex de-inverted: raw PtAdmiTable Sex is swapped vs the paper (raw "Female"
    n=5,215 = the paper's male count); reader now maps raw Male→female, Female→male.
  - lact / cai unit multipliers removed: both have canonical_unit mmol/L, so the
    mmol/L→mg/dL multiplier silently corrupted lactate (x9) and clamped out all
    ionized calcium (x4 then dropped by the (0.5,2.0) valid_range).
  - patient_id = the patient (patient_SN), stay_id = the stay, so
    PatientGroupedKFold dedupes multi-admission patients across splits.

4. ICU length-of-stay is reconstructed from HospitalTransfer ICU segments where
   available, but only a minority of stays carry ICU transfer endpoints, so for
   the majority the admit/discharge window falls back to the monitoring-event span
   (capped at hospital discharge), or to the hospital window when no events exist.
   Therefore OMIX `los_hours` is ICU-scoped only where transfer data exists and
   monitoring-span-scoped otherwise — it must not be silently compared as pure
   ward-transfer ICU-LoS against the other datasets.

Notable gaps in the released Lab table (vs ricu-faithful concept registries):
- Na, K, Cl absent (Anion gap is present — measured but stripped)
- Mg, fibrinogen, methemoglobin absent
- TnI present (44k rows) merged into canonical `tnt` per NWICU precedent.

References:
- Audit: reports/omix_dataset_audit.md
- Paper: data/raw/OMIX005817/s41597-023-01952-3.pdf (Jin et al. 2023)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import polars as pl

from critical_mm.datasets._drug_classifier import classify_drug_expr_omix as classify_drug_expr
from critical_mm.datasets._omix_dicts import (
    _OMIX_ICU_DEPTS,
    OMIX_LAB_ITEMNAME_TO_CONCEPT,
    OMIX_NURSINGCHART_IO_CN_TO_CONCEPT,
    OMIX_NURSINGCHART_VS_CN_TO_CONCEPT,
)
from critical_mm.datasets.base import DatasetReader
from critical_mm.registry import register_dataset
from critical_mm.schema import TABLES, empty_frame

_AGE_CUT_MIDPOINT: dict[str, int] = {
    "(0,18]": 9,
    "(18,30]": 24,
    "(30,40]": 35,
    "(40,50]": 45,
    "(50,60]": 55,
    "(60,70]": 65,
    "(70,80]": 75,
    "(80,90]": 85,
    "(90,150]": 95,
}

_NA_VALUES: list[str] = ["NA", "", "NULL", "null", "nan", "NaN"]

_OMIX_VITALSIGN_DESC_TO_CONCEPT: dict[str, str] = {
    "Respiratory rate": "resp",
    "Temperature": "temp",
    "Heart Rate": "hr",
    "Pulse rate": "hr",
    "Systolic Blood pressure": "sbp",
    "Diastolic Blood pressure": "dbp",
    "Oxygen saturation (Pulse Oxymetry)": "o2sat",
}

_OMIX_VENT_TRIGGER_ITEMS: list[str] = [
    "呼吸机模式",
    "呼气末正压",
    "气道峰压",
    "分钟通气量",
    "呼吸频率(设)",
]

@register_dataset
class OMIXReader(DatasetReader):
    """Concrete reader for OMIX005817 (Zhejiang Provincial ICU)."""

    CAPABILITIES: ClassVar[frozenset[str]] = frozenset(
        {
            "urine",
            "microbio",
            "abx_duration",
        }
    )

    def __init__(
        self,
        *,
        raw_root: Path,
        interim_root: Path,
        repo_root: Path,
        anchor_time: datetime | None = None,
    ) -> None:
        super().__init__(raw_root=raw_root, interim_root=interim_root, repo_root=repo_root)
        self.anchor_time: datetime = anchor_time or datetime(2012, 1, 1, tzinfo=UTC)
        self._icu_window_cache: pl.DataFrame | None = None
        self._mon_window_cache: pl.DataFrame | None = None

    @property
    def dataset_name(self) -> str:
        return "omix"

    @property
    def _csv_root(self) -> Path:
        return self.raw_root / "DataTable2"

    def cache_source_paths(self, table: str) -> list[Path]:
        """Coarse cache key — return every OMIX CSV."""
        if not self._csv_root.exists():
            return []
        return sorted(self._csv_root.glob("*.csv"))

    def _offset_days_to_time(self, days_col: str) -> pl.Expr:
        """day-offset column → UTC Datetime relative to anchor_time (minute precision)."""
        minutes = (pl.col(days_col).cast(pl.Float64) * 1440.0).round(0).cast(pl.Int64)
        return (
            pl.lit(self.anchor_time)
            .cast(pl.Datetime("us", "UTC"))
            .dt.offset_by(minutes.cast(pl.Utf8) + pl.lit("m"))
        )

    def _scan(self, name: str) -> pl.LazyFrame:
        return pl.scan_csv(
            self._csv_root / name,
            infer_schema_length=10_000,
            ignore_errors=True,
            encoding="utf8-lossy",
            null_values=_NA_VALUES,
            schema_overrides={"Hospital_ID": pl.String, "patient_SN": pl.String},
        )

    def _icu_window_from_transfer(self) -> pl.DataFrame:
        """Per-stay ICU occupancy (day-offsets) from HospitalTransfer events.

        icu_in = min TransferIn over rows entering ICU; icu_out = max TransferOut
        over rows leaving ICU. Columns [patient_SN, Hospital_ID, icu_in, icu_out];
        either may be null. Memoized (called repeatedly via _stays_with_keys).
        """
        if self._icu_window_cache is not None:
            return self._icu_window_cache
        empty: pl.DataFrame = pl.DataFrame(
            schema={
                "patient_SN": pl.Utf8,
                "Hospital_ID": pl.Utf8,
                "icu_in": pl.Float64,
                "icu_out": pl.Float64,
            }
        )
        if not (self._csv_root / "HospitalTransfer.csv").exists():
            self._icu_window_cache = empty
            return empty
        ht = self._scan("HospitalTransfer.csv")
        icu = list(_OMIX_ICU_DEPTS)
        ent = (
            ht.filter(pl.col("TransferTo_Dept_Eng").is_in(icu))
            .group_by(["patient_SN", "Hospital_ID"])
            .agg(pl.col("TransferIn_DateTime").cast(pl.Float64).min().alias("icu_in"))
        )
        ext = (
            ht.filter(pl.col("TransferFrom_Dept_Eng").is_in(icu))
            .group_by(["patient_SN", "Hospital_ID"])
            .agg(pl.col("TransferOut_DateTime").cast(pl.Float64).max().alias("icu_out"))
        )
        self._icu_window_cache = ent.join(
            ext, on=["patient_SN", "Hospital_ID"], how="full", coalesce=True
        ).collect()
        return self._icu_window_cache

    def _monitoring_window(self) -> pl.DataFrame:
        """Per-stay [first,last] charttime (day-offsets) across the 4 event CSVs.

        Reads from raw CSVs (NOT read_events_long). Case-4 fallback only for
        stays with no HospitalTransfer data. Memoized.
        """
        if self._mon_window_cache is not None:
            return self._mon_window_cache
        specs = [
            ("Lab.csv", "Lab_DateTime"),
            ("VitalSign.csv", "VitalSign_DateTime"),
            ("NursingChart_VitalSign.csv", "Event_DateTime"),
            ("NursingChart_IO.csv", "Event_DateTime"),
        ]
        arms: list[pl.LazyFrame] = []
        for fname, tcol in specs:
            if self._source_has_column(fname, tcol):
                arms.append(
                    self._scan(fname).select(
                        "patient_SN",
                        "Hospital_ID",
                        pl.col(tcol).cast(pl.Float64).alias("_t"),
                    )
                )
        empty: pl.DataFrame = pl.DataFrame(
            schema={
                "patient_SN": pl.Utf8,
                "Hospital_ID": pl.Utf8,
                "mon_first": pl.Float64,
                "mon_last": pl.Float64,
            }
        )
        if not arms:
            self._mon_window_cache = empty
            return empty
        self._mon_window_cache = (
            pl.concat(arms, how="vertical_relaxed")
            .filter(pl.col("_t") >= 0.0)
            .group_by(["patient_SN", "Hospital_ID"])
            .agg(
                pl.col("_t").min().alias("mon_first"),
                pl.col("_t").max().alias("mon_last"),
            )
            .collect()
        )
        return self._mon_window_cache

    def _stays_with_keys(self) -> pl.LazyFrame:
        """Canonical stays + the join keys child tables need.

        Carries the canonical SCHEMA_STAYS columns PLUS
        (original_patient_sn, original_hospital_id, discharge_days) for child-table
        joins + per-stay event-window bounding. `read_stays` projects to canonical.
        """
        adm = pl.read_csv(
            self._csv_root / "PtAdmiTable.csv",
            infer_schema_length=10_000,
            ignore_errors=True,
            encoding="utf8-lossy",
            null_values=_NA_VALUES,
            schema_overrides={"Hospital_ID": pl.String, "patient_SN": pl.String},
        )
        if adm.columns and adm.columns[0] == "":
            adm = adm.drop("")

        groups = adm.group_by(["patient_SN", "Hospital_ID"]).agg(pl.len().alias("__n"))
        singletons = groups.filter(pl.col("__n") == 1).select(["patient_SN", "Hospital_ID"])
        adm = adm.join(singletons, on=["patient_SN", "Hospital_ID"], how="inner")

        adm = adm.with_columns(
            pl.col("Discharge_DateTime").cast(pl.Float64, strict=False).alias("discharge_days"),
        )
        adm = adm.filter(pl.col("discharge_days").is_not_null())

        adm = adm.join(
            self._icu_window_from_transfer(),
            on=["patient_SN", "Hospital_ID"],
            how="left",
        )
        adm = adm.join(
            self._monitoring_window(),
            on=["patient_SN", "Hospital_ID"],
            how="left",
        )

        hosp_disch = pl.col("discharge_days")
        both_ok = (
            pl.col("icu_in").is_not_null()
            & pl.col("icu_out").is_not_null()
            & (pl.col("icu_out") > pl.col("icu_in"))
        )
        admit_days_expr = (
            pl.when(both_ok)
            .then(pl.col("icu_in"))
            .when(pl.col("icu_in").is_not_null() & pl.col("icu_out").is_null())
            .then(pl.col("icu_in"))
            .when(pl.col("icu_in").is_null() & pl.col("icu_out").is_not_null())
            .then(pl.lit(0.0))
            .otherwise(pl.coalesce(pl.col("mon_first"), pl.lit(0.0)))
        )
        disch_days_expr = (
            pl.when(both_ok)
            .then(pl.col("icu_out"))
            .when(pl.col("icu_in").is_not_null() & pl.col("icu_out").is_null())
            .then(hosp_disch)
            .when(pl.col("icu_in").is_null() & pl.col("icu_out").is_not_null())
            .then(pl.col("icu_out"))
            .otherwise(
                pl.min_horizontal(
                    pl.coalesce(pl.col("mon_last"), hosp_disch),
                    hosp_disch,
                )
            )
        )
        adm = adm.with_columns(
            admit_days_expr.cast(pl.Float64).alias("admit_days"),
            disch_days_expr.cast(pl.Float64).alias("discharge_days"),
        )

        dead = (pl.col("StatusOnDischarge") == "Dead").fill_null(False)

        sex = (
            pl.when(pl.col("Sex") == "Male")
            .then(pl.lit("F"))
            .when(pl.col("Sex") == "Female")
            .then(pl.lit("M"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
        )

        out = adm.with_columns(
            pl.col("patient_SN").alias("original_patient_sn"),
            pl.col("Hospital_ID").alias("original_hospital_id"),
            pl.col("patient_SN").cast(pl.Utf8).alias("subject_id"),
            ("omix_" + pl.col("patient_SN").cast(pl.Utf8)).alias("patient_id"),
            (
                "omix_"
                + pl.col("patient_SN").cast(pl.Utf8)
                + pl.lit("__")
                + pl.col("Hospital_ID").cast(pl.Utf8)
            ).alias("stay_id"),
            pl.lit("omix").alias("dataset"),
            pl.col("Hospital_ID").cast(pl.Utf8).alias("hospital_id"),
            pl.col("Age_cut")
            .replace_strict(_AGE_CUT_MIDPOINT, default=None)
            .cast(pl.Float32)
            .alias("age"),
            sex.alias("sex"),
            pl.lit(None, dtype=pl.Utf8).alias("ethnicity"),
            pl.lit(None, dtype=pl.Float32).alias("weight"),
            pl.lit(None, dtype=pl.Float32).alias("height"),
            self._offset_days_to_time("admit_days").alias("admit_time"),
            self._offset_days_to_time("discharge_days").alias("discharge_time"),
            ((pl.col("discharge_days") - pl.col("admit_days")) * 24.0)
            .cast(pl.Float32)
            .alias("los_hours"),
            dead.cast(pl.Boolean).alias("mortality_in_icu"),
            dead.cast(pl.Boolean).alias("mortality_in_hospital"),
            pl.lit(None, dtype=pl.Boolean).alias("mortality_30day"),
            pl.lit(None, dtype=pl.Utf8).alias("admission_diagnosis"),
        )
        keep = [
            *TABLES["stays"][0].keys(),
            "original_patient_sn",
            "original_hospital_id",
            "admit_days",
            "discharge_days",
        ]
        return out.select(keep).lazy()

    def read_stays(self) -> pl.LazyFrame:
        """Canonical stays table (SCHEMA_STAYS) — projects _stays_with_keys."""
        return self._stays_with_keys().select(list(TABLES["stays"][0].keys()))

    def _event_join_keys(self) -> pl.LazyFrame:
        return self._stays_with_keys().select(
            [
                "original_patient_sn",
                "original_hospital_id",
                "patient_id",
                "stay_id",
                "admit_days",
                "discharge_days",
            ]
        )

    def _finalize_event_arm(self, arm: pl.LazyFrame, time_col: str, unit_col: str) -> pl.LazyFrame:
        """Join stay keys, bound to [admit, discharge], emit canonical events_long columns.

        `arm` must carry: patient_SN, Hospital_ID, concept, value, <time_col>, <unit_col>.
        """
        return (
            arm.join(
                self._event_join_keys(),
                left_on=["patient_SN", "Hospital_ID"],
                right_on=["original_patient_sn", "original_hospital_id"],
                how="inner",
            )
            .with_columns(pl.col("value").cast(pl.Float64, strict=False).alias("value"))
            .filter(
                pl.col("value").is_not_null()
                & (pl.col(time_col).cast(pl.Float64) >= pl.col("admit_days"))
                & (pl.col(time_col).cast(pl.Float64) <= pl.col("discharge_days"))
            )
            .with_columns(
                self._offset_days_to_time(time_col).alias("charttime"),
                pl.col(unit_col).cast(pl.Utf8).alias("unit"),
                pl.col(unit_col).cast(pl.Utf8).alias("unit_source"),
            )
            .select(
                ["patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"]
            )
        )

    def _events_lab(self) -> pl.LazyFrame:
        arm = (
            self._scan("Lab.csv")
            .drop("")
            .filter(pl.col("Lab_itemName_Eng").is_in(list(OMIX_LAB_ITEMNAME_TO_CONCEPT.keys())))
            .with_columns(
                pl.col("Lab_itemName_Eng")
                .replace_strict(OMIX_LAB_ITEMNAME_TO_CONCEPT, default=None)
                .alias("concept"),
                pl.col("Lab_results").alias("value"),
            )
        )
        return self._finalize_event_arm(arm, "Lab_DateTime", "Unit_measure")

    def _events_vital(self) -> pl.LazyFrame:
        arm = (
            self._scan("VitalSign.csv")
            .drop("")
            .filter(pl.col("VitalSign_DESC").is_in(list(_OMIX_VITALSIGN_DESC_TO_CONCEPT.keys())))
            .with_columns(
                pl.col("VitalSign_DESC")
                .replace_strict(_OMIX_VITALSIGN_DESC_TO_CONCEPT, default=None)
                .alias("concept"),
                pl.col("VitalSign_value").alias("value"),
            )
        )
        return self._finalize_event_arm(arm, "VitalSign_DateTime", "VitalSign_unit")

    def _events_nursing_vital(self) -> pl.LazyFrame:
        arm = (
            self._scan("NursingChart_VitalSign.csv")
            .drop("")
            .filter(
                pl.col("NursingEvent_item").is_in(list(OMIX_NURSINGCHART_VS_CN_TO_CONCEPT.keys()))
            )
            .with_columns(
                pl.col("NursingEvent_item")
                .replace_strict(OMIX_NURSINGCHART_VS_CN_TO_CONCEPT, default=None)
                .alias("concept"),
                pl.col("NursingEvent_val").alias("value"),
            )
        )
        return self._finalize_event_arm(arm, "Event_DateTime", "NursingEvent_Unit")

    def _events_io(self) -> pl.LazyFrame:
        arm = (
            self._scan("NursingChart_IO.csv")
            .drop("")
            .filter(pl.col("NursingIO_item").is_in(list(OMIX_NURSINGCHART_IO_CN_TO_CONCEPT.keys())))
            .with_columns(
                pl.col("NursingIO_item")
                .replace_strict(OMIX_NURSINGCHART_IO_CN_TO_CONCEPT, default=None)
                .alias("concept"),
                pl.col("NursingIO_val").str.extract(r"^(\d+(?:\.\d+)?)").alias("value"),
            )
        )
        return self._finalize_event_arm(arm, "Event_DateTime", "NursingIO_Unit")

    def _source_has_column(self, fname: str, col: str) -> bool:
        """True iff source CSV exists and carries `col` (cheap header read).

        Guards read_events_long against an absent/empty source table (e.g. a
        pre-2018-only subset, or a degenerate test fixture).
        """
        path = self._csv_root / fname
        if not path.exists():
            return False
        try:
            return col in self._scan(fname).collect_schema().names()
        except Exception:
            return False

    def read_events_long(self, concepts: list[str] | None = None) -> pl.LazyFrame:
        """Canonical events_long: concat of up-to-4 per-source arms + unit conversion."""
        del concepts
        specs = [
            ("Lab.csv", "Lab_itemName_Eng", self._events_lab),
            ("VitalSign.csv", "VitalSign_DESC", self._events_vital),
            ("NursingChart_VitalSign.csv", "NursingEvent_item", self._events_nursing_vital),
            ("NursingChart_IO.csv", "NursingIO_item", self._events_io),
        ]
        arms = [build() for fname, col, build in specs if self._source_has_column(fname, col)]
        if not arms:
            return empty_frame("events_long")
        concat = pl.concat(arms, how="vertical_relaxed")
        converted = _apply_unit_conversion_omix(concat)
        return converted.with_columns(pl.col("value").cast(pl.Float32)).select(
            ["patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"]
        )

    def read_meds(self) -> pl.LazyFrame:
        """Canonical meds table (SCHEMA_MEDS)."""
        m = self._scan("Medication.csv").drop("").filter(pl.col("Med_DESC_Eng").is_not_null())
        return (
            m.join(
                self._event_join_keys(),
                left_on=["patient_SN", "Hospital_ID"],
                right_on=["original_patient_sn", "original_hospital_id"],
                how="inner",
            )
            .filter(
                (pl.col("Med_startTime").cast(pl.Float64) >= pl.col("admit_days"))
                & (pl.col("Med_startTime").cast(pl.Float64) <= pl.col("discharge_days"))
            )
            .with_columns(
                self._offset_days_to_time("Med_startTime").alias("starttime"),
                self._offset_days_to_time("Med_stopTime").alias("endtime"),
                pl.col("Med_DESC_Eng").cast(pl.Utf8).alias("drug"),
                pl.col("SingleDose").cast(pl.Float32, strict=False).alias("dose"),
                pl.col("Med_unit").cast(pl.Utf8).alias("dose_unit"),
                pl.col("Med_route_Eng").cast(pl.Utf8).alias("route"),
                classify_drug_expr("Med_DESC_Eng").alias("drug_class"),
            )
            .select(list(TABLES["meds"][0].keys()))
        )

    def read_microbio(self) -> pl.LazyFrame:
        """Canonical microbio table (SCHEMA_MICROBIO).

        organism is left null: SEP-3 `susp_inf_alt` consumes only (stay_id,
        charttime) — the sampling event itself is the signal — so the provisional
        pinyin positive/negative decode is NOT needed for the current cascade
        (decode verification deferred; see omix-limitations
        `omix_microbio_pinyin_decode_provisional`).
        """
        mb = (
            self._scan("MicrobiologyCulture.csv")
            .drop("")
            .filter(pl.col("MicrobiologyCulture_sample_Eng").is_not_null())
        )
        return (
            mb.join(
                self._event_join_keys(),
                left_on=["patient_SN", "Hospital_ID"],
                right_on=["original_patient_sn", "original_hospital_id"],
                how="inner",
            )
            .filter(
                (pl.col("MicrobiologyCulture_DateTime").cast(pl.Float64) >= pl.col("admit_days"))
                & (
                    pl.col("MicrobiologyCulture_DateTime").cast(pl.Float64)
                    <= pl.col("discharge_days")
                )
            )
            .with_columns(
                self._offset_days_to_time("MicrobiologyCulture_DateTime").alias("charttime"),
                pl.col("MicrobiologyCulture_sample_Eng").cast(pl.Utf8).alias("specimen_type"),
                pl.lit(None, dtype=pl.Utf8).alias("organism"),
            )
            .select(list(TABLES["microbio"][0].keys()))
        )

    def read_interventions(self) -> pl.LazyFrame:
        """Canonical interventions table (SCHEMA_INTERVENTIONS). mech_vent from
        the 5 NursingChart_VitalSign ventilator-setting trigger items."""
        vent = (
            self._scan("NursingChart_VitalSign.csv")
            .drop("")
            .filter(pl.col("NursingEvent_item").is_in(_OMIX_VENT_TRIGGER_ITEMS))
        )
        return (
            vent.join(
                self._event_join_keys(),
                left_on=["patient_SN", "Hospital_ID"],
                right_on=["original_patient_sn", "original_hospital_id"],
                how="inner",
            )
            .filter(
                (pl.col("Event_DateTime").cast(pl.Float64) >= pl.col("admit_days"))
                & (pl.col("Event_DateTime").cast(pl.Float64) <= pl.col("discharge_days"))
            )
            .with_columns(
                self._offset_days_to_time("Event_DateTime").alias("starttime"),
                pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("endtime"),
                pl.lit("mech_vent").alias("intervention"),
            )
            .select(list(TABLES["interventions"][0].keys()))
        )

    def read_diagnoses(self) -> pl.LazyFrame:
        """Canonical diagnoses table (SCHEMA_DIAGNOSES). ICD-10-CN suffixes stripped
        to the 3-char root; semicolon multi-codes exploded. Diagnoses are static
        (no charttime in the canonical schema), so past-history rows are retained."""
        keys = self._stays_with_keys().select(
            ["original_patient_sn", "original_hospital_id", "patient_id", "stay_id"]
        )
        return (
            self._scan("Diagnosis.csv")
            .drop("")
            .filter(pl.col("ICD10_code").is_not_null())
            .with_columns(pl.col("ICD10_code").str.split(";").alias("__codes"))
            .explode("__codes")
            .with_columns(pl.col("__codes").str.extract(r"^([A-Z]\d{2})").alias("icd_code"))
            .filter(pl.col("icd_code").is_not_null())
            .join(
                keys,
                left_on=["patient_SN", "Hospital_ID"],
                right_on=["original_patient_sn", "original_hospital_id"],
                how="inner",
            )
            .with_columns(
                pl.lit("10").alias("icd_version"),
                pl.col("icd_code")
                .cum_count()
                .over("stay_id")
                .cast(pl.Int32)
                .alias("diagnosis_position"),
            )
            .select(list(TABLES["diagnoses"][0].keys()))
        )

    def read_notes(self) -> pl.LazyFrame:
        """OMIX FirstNote + ProgressNote are absent from the PhysioNet release."""
        return empty_frame("notes")

    def read_abx_duration(self) -> pl.LazyFrame:
        """Canonical abx_duration (SCHEMA_ABX_DURATION) — antibiotic admin intervals.

        Surrogate path (like NWICU): filter meds to drug_class == 'antibiotic'.
        """
        return (
            self.read_meds()
            .filter(pl.col("drug_class") == "antibiotic")
            .select(list(TABLES["abx_duration"][0].keys()))
        )

_OMIX_UNIT_MULTIPLIERS: dict[tuple[str, str], float] = {
    ("crea", "umol/l"): 0.011312,
    ("bun", "mmol/l"): 2.8014,
    ("glu", "mmol/l"): 18.0156,
    ("bili", "umol/l"): 0.058467,
    ("bili_dir", "umol/l"): 0.058467,
    ("alb", "g/l"): 0.1,
    ("hgb", "g/l"): 0.1,
    ("ca", "mmol/l"): 4.008,
    ("phos", "mmol/l"): 3.0974,
}

def _apply_unit_conversion_omix(events: pl.LazyFrame) -> pl.LazyFrame:
    """Apply concept-specific unit scaling + canonical unit labelling + valid_range clamp.

    Works on the canonical-events schema (patient_id, stay_id, charttime, concept,
    value, unit, unit_source). Does NOT call drop_null_required (the inline path).
    """
    from critical_mm.concepts import CONCEPTS_BY_NAME

    scaled_value = pl.col("value")
    for (concept, src_unit_lower), mult in _OMIX_UNIT_MULTIPLIERS.items():
        match = (pl.col("concept") == concept) & (
            pl.col("unit").str.to_lowercase() == src_unit_lower
        )
        scaled_value = pl.when(match).then(pl.col("value") * mult).otherwise(scaled_value)

    canon_unit_map = {
        name: cc.canonical_unit for name, cc in CONCEPTS_BY_NAME.items() if cc.canonical_unit
    }
    events = events.with_columns(
        scaled_value.alias("value"),
        pl.col("concept").replace_strict(canon_unit_map, default="unknown").alias("unit"),
    )

    from critical_mm.datasets._units_helper import _VALID_RANGE_LOOKUP

    ranged = events.join(_VALID_RANGE_LOOKUP, on="concept", how="left")
    in_range = ranged.filter(
        pl.col("_cmm_range_lo").is_null()
        | (
            (pl.col("value") >= pl.col("_cmm_range_lo"))
            & (pl.col("value") <= pl.col("_cmm_range_hi"))
        )
    ).drop("_cmm_range_lo", "_cmm_range_hi")
    return in_range.filter(
        pl.col("patient_id").is_not_null()
        & pl.col("concept").is_not_null()
        & pl.col("value").is_not_null()
    )

