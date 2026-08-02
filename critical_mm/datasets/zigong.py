"""ZigongReader — harmonise the Zigong Fourth People's Hospital infection cohort.

Zigong (Sichuan, China; the dataset Burger et al. 2024 / arXiv:2411.16346 used)
is the 7th benchmarked dataset, integrated following the SICdb→V10 precedent
(itself a clone of OMIX→V2). Template = omix.py (offset-time + transfer-window
reconstruction); borrows SICdb's clean _finalize_event_arm / unit-conversion shape.

Structural facts that drive this reader (spec 2026-06-13, §2):
- Files are UTF-8 (NOT GB18030 — the brief's GB assumption was wrong).
- All time columns are HOURS offset from hospital admission (t0 = 0); there are
  no calendar dates. The reader reconstructs UTC datetimes against a
  deterministic `anchor_time` (the real admission year is unknown by design).
- PATIENT_ID ↔ INP_NO are strictly 1:1 in dtBaseline (the crosswalk). The big
  event tables (dtLab, dtNursingChart) are keyed on INP_NO; dtDrugs & dtOutCome
  are keyed on PATIENT_ID only and reach a stay through the crosswalk.
- `stay_id = "zigong_" + INP_NO`, `patient_id = "zigong_" + PATIENT_ID`.
- ICU window from dtTransfer (filter to the ICU ∪ EICU dept set, merge contiguous
  segments): admit = min ICU StartTime; discharge = dtBaseline.ICU_discharge_time
  (authoritative, 100% populated).
- Mortality = in-hospital discharge-status death flag (dtICD.Status_Discharge
  == 'Dead'), 161/2790 patients; the SF-36 follow-up Death_Date is NOT used.
  NEVER use raw Follow_Vital (the 47% SF-36 long-term follow-up signal); do NOT
  pre-bake a 24h death cut (that reproduces the documented OMIX sub-day bug).
- Reader-level derivations (novel — spec §4.5):
    MAP = (sbp + 2·dbp) / 3 where `map` is absent at a timestamp but sbp+dbp exist.
    GCS = summed E + M + V parsed from the prefix-coded Chinese level strings,
          aligned to same-ChartTime triplets, in [3, 15].
- Safeguards (spec §5): NURSING_DESC dropped at scan; strict-quoted CSV for the
  wide vitals table; `?`-strip before cast; events bounded to [admit, discharge].

Templates: critical_mm/datasets/omix.py, critical_mm/datasets/sicdb.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import polars as pl

from critical_mm.datasets._zigong_dicts import (
    _ZIGONG_GCS_EYE_COLUMN,
    _ZIGONG_GCS_EYE_RANGE,
    _ZIGONG_GCS_MOTOR_COLUMN,
    _ZIGONG_GCS_MOTOR_RANGE,
    _ZIGONG_GCS_SEP,
    _ZIGONG_GCS_VERBAL_COLUMN,
    _ZIGONG_GCS_VERBAL_RANGE,
    _ZIGONG_ICU_DEPTS,
    _ZIGONG_STRICT_ICU_DEPTS,
    _ZIGONG_VENT_COLUMNS,
    ZIGONG_DRUG_CLASS_HINTS,
    ZIGONG_FRACTION_TO_PERCENT,
    ZIGONG_LAB_ITEM_TO_CONCEPT,
    ZIGONG_NURSINGCHART_SOURCE_UNIT,
    ZIGONG_NURSINGCHART_TO_CONCEPT,
    ZIGONG_UNIT_MULTIPLIERS,
)
from critical_mm.datasets.base import DatasetReader
from critical_mm.registry import register_dataset
from critical_mm.schema import TABLES, empty_frame

_NA_VALUES: list[str] = ["NA", "", "NULL", "null", "nan", "NaN"]

_NURSING_DESC_COLUMN: str = "NURSING_DESC"

_VITAL_COLUMNS: tuple[str, ...] = tuple(ZIGONG_NURSINGCHART_TO_CONCEPT.keys())

_AGE_CAP: float = 120.0

@register_dataset
class ZigongReader(DatasetReader):
    """Concrete reader for the Zigong Fourth People's Hospital cohort."""

    CAPABILITIES: ClassVar[frozenset[str]] = frozenset({"urine", "abx_duration"})

    def __init__(
        self,
        *,
        raw_root: Path,
        interim_root: Path,
        repo_root: Path,
        anchor_time: datetime | None = None,
        include_eicu: bool = True,
    ) -> None:
        super().__init__(raw_root=raw_root, interim_root=interim_root, repo_root=repo_root)
        self.anchor_time: datetime = anchor_time or datetime(2019, 1, 1, tzinfo=UTC)
        self.include_eicu = include_eicu

    @property
    def dataset_name(self) -> str:
        return "zigong"

    @property
    def _icu_depts(self) -> frozenset[str]:
        return _ZIGONG_ICU_DEPTS if self.include_eicu else _ZIGONG_STRICT_ICU_DEPTS

    def _scan(self, name: str) -> pl.LazyFrame:
        return pl.scan_csv(
            self.raw_root / name,
            infer_schema_length=10_000,
            ignore_errors=True,
            encoding="utf8",
            null_values=_NA_VALUES,
            schema_overrides={"PATIENT_ID": pl.String, "INP_NO": pl.String},
        )

    def _scan_nursing(self) -> pl.LazyFrame:
        """dtNursingChart scan with the §5.1 safeguards.

        Drops NURSING_DESC at scan (its embedded newlines would otherwise mangle
        real rows), uses strict quoted CSV instead of ignore_errors for the wide
        vitals table, and never trusts ragged lines.
        """
        path = self.raw_root / "dtNursingChart.csv"
        header = pl.scan_csv(path, n_rows=0, encoding="utf8").collect_schema().names()
        overrides = {c: pl.String for c in header}
        lf = pl.scan_csv(
            path,
            infer_schema_length=50_000,
            encoding="utf8",
            null_values=_NA_VALUES,
            quote_char='"',
            truncate_ragged_lines=False,
            schema_overrides=overrides,
        )
        cols = lf.collect_schema().names()
        if _NURSING_DESC_COLUMN in cols:
            lf = lf.drop(_NURSING_DESC_COLUMN)
        return lf

    def _offset_hours_to_time(self, col: str) -> pl.Expr:
        """hours-offset column → UTC Datetime relative to anchor_time (min precision)."""
        minutes = (pl.col(col).cast(pl.Float64) * 60.0).round(0).cast(pl.Int64)
        return (
            pl.lit(self.anchor_time)
            .cast(pl.Datetime("us", "UTC"))
            .dt.offset_by(minutes.cast(pl.Utf8) + pl.lit("m"))
        )

    @staticmethod
    def _clean_numeric(col: str) -> pl.Expr:
        """Strip the `?`-prefix corruption + censor markers BEFORE casting (spec §5.2)."""
        return (
            pl.col(col)
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.replace_all(r"^[?<>=]+", "")
            .str.strip_chars()
            .cast(pl.Float64, strict=False)
        )

    def _source_exists(self, name: str) -> bool:
        """True iff the source CSV is present (guards each arm against an absent
        table — e.g. a degenerate test fixture that supplies only some files)."""
        return (self.raw_root / name).exists()

    def cache_source_paths(self, table: str) -> list[Path]:
        del table
        if not self.raw_root.exists():
            return []
        return sorted(self.raw_root.glob("*.csv"))

    def _icu_window(self) -> pl.LazyFrame:
        """Per-stay ICU admit (min ICU StartTime) from dtTransfer.

        Filters dtTransfer to the ICU ∪ EICU dept set, then takes the earliest
        StartTime per INP_NO. Contiguous/overlapping ICU segments collapse to
        their minimum start under the group-by min — administrative micro-splits
        do not change the admit point. Columns: [INP_NO, icu_admit_hours].
        """
        return (
            self._scan("dtTransfer.csv")
            .filter(pl.col("TransferDept").is_in(list(self._icu_depts)))
            .group_by("INP_NO")
            .agg(pl.col("StartTime").cast(pl.Float64, strict=False).min().alias("icu_admit_hours"))
        )

    def _outcome(self) -> pl.LazyFrame:
        """Per-admission in-hospital death flag from dtICD.Status_Discharge.

        Mortality is the in-hospital discharge-status death flag
        (``Status_Discharge == 'Dead'``), keyed on INP_NO — exactly 161/2790
        patients (5.8%), matching the dataset descriptor and Burger et al. The
        dtOutCome SF-36 follow-up death date (``Death_Date``) is a long-term
        post-discharge signal and is DELIBERATELY ignored (it over-counts
        in-hospital deaths ~3x). Mirrors omix.py's ``StatusOnDischarge`` flag.

        Guarded by ``_source_exists`` like every other arm: an absent dtICD
        (a degenerate fixture supplying only dtBaseline/dtLab) yields an empty
        [INP_NO, dead] frame so the left-join in ``_stays_with_keys`` leaves
        ``dead`` null → ``fill_null(False)`` downstream (no death flag).
        """
        if not self._source_exists("dtICD.csv"):
            return pl.LazyFrame(schema={"INP_NO": pl.String, "dead": pl.Boolean})
        return (
            self._scan("dtICD.csv")
            .group_by("INP_NO")
            .agg(
                (pl.col("Status_Discharge") == "Dead")
                .max()
                .fill_null(False)
                .alias("dead"),
            )
        )

    def _stays_with_keys(self) -> pl.LazyFrame:
        """Canonical stays + (original_inp_no, original_patient_id, admit_hours,
        disch_hours, death_hours, hosp_disch_hours) for child joins/bounding."""
        base = self._scan("dtBaseline.csv").with_columns(
            pl.col("ICU_discharge_time").cast(pl.Float64, strict=False).alias("disch_hours"),
            pl.col("DISCHARGE_DATE_TIME").cast(pl.Float64, strict=False).alias("hosp_disch_hours"),
        )
        base = (
            base.join(self._icu_window(), on="INP_NO", how="left")
            .join(self._outcome(), on="INP_NO", how="left")
            .with_columns(
                pl.coalesce(pl.col("icu_admit_hours"), pl.lit(0.0)).alias("admit_hours"),
            )
        )

        sex = (
            pl.when(pl.col("SEX").str.to_lowercase().str.starts_with("m"))
            .then(pl.lit("M"))
            .when(pl.col("SEX").str.to_lowercase().str.starts_with("f"))
            .then(pl.lit("F"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
        )

        return base.with_columns(
            pl.col("INP_NO").cast(pl.Utf8).alias("original_inp_no"),
            pl.col("PATIENT_ID").cast(pl.Utf8).alias("original_patient_id"),
            ("zigong_" + pl.col("PATIENT_ID").cast(pl.Utf8)).alias("patient_id"),
            pl.col("PATIENT_ID").cast(pl.Utf8).alias("subject_id"),
            ("zigong_" + pl.col("INP_NO").cast(pl.Utf8)).alias("stay_id"),
            pl.lit("zigong").alias("dataset"),
            pl.lit("zigong").alias("hospital_id"),
            pl.col("Age")
            .cast(pl.Float64, strict=False)
            .clip(0.0, _AGE_CAP)
            .cast(pl.Float32)
            .alias("age"),
            sex.alias("sex"),
            pl.lit(None, dtype=pl.Utf8).alias("ethnicity"),
            pl.lit(None, dtype=pl.Float32).alias("weight"),
            pl.lit(None, dtype=pl.Float32).alias("height"),
            self._offset_hours_to_time("admit_hours").alias("admit_time"),
            self._offset_hours_to_time("disch_hours").alias("discharge_time"),
            pl.max_horizontal(
                pl.col("disch_hours") - pl.col("admit_hours"),
                pl.lit(0.0),
            )
            .cast(pl.Float32)
            .alias("los_hours"),
            pl.col("dead").fill_null(False).alias("mortality_in_icu"),
            pl.col("dead").fill_null(False).alias("mortality_in_hospital"),
            pl.lit(None, dtype=pl.Boolean).alias("mortality_30day"),
            pl.col("InfectionSite").cast(pl.Utf8).alias("admission_diagnosis"),
        )

    def read_stays(self) -> pl.LazyFrame:
        return (
            self._stays_with_keys()
            .filter(
                pl.col("disch_hours").is_not_null()
                & pl.col("sex").is_not_null()
                & pl.col("age").is_not_null()
            )
            .select(list(TABLES["stays"][0].keys()))
        )

    def _event_join_keys(self) -> pl.LazyFrame:
        return self._stays_with_keys().select(
            [
                "original_inp_no",
                "original_patient_id",
                "patient_id",
                "stay_id",
                "admit_hours",
                "disch_hours",
            ]
        )

    def _finalize_event_arm(self, arm: pl.LazyFrame, time_col: str) -> pl.LazyFrame:
        """Join stay keys (on INP_NO), bound to [admit_hours, disch_hours], emit
        canonical events_long columns.

        `arm` must carry: original_inp_no, concept, value, <time_col>, unit.
        """
        return (
            arm.join(
                self._event_join_keys(),
                on="original_inp_no",
                how="inner",
            )
            .with_columns(pl.col("value").cast(pl.Float64, strict=False).alias("value"))
            .filter(
                pl.col("value").is_not_null()
                & (pl.col(time_col).cast(pl.Float64) >= pl.col("admit_hours"))
                & (pl.col(time_col).cast(pl.Float64) <= pl.col("disch_hours"))
            )
            .with_columns(
                self._offset_hours_to_time(time_col).alias("charttime"),
                pl.col("unit").cast(pl.Utf8).alias("unit"),
                pl.col("unit").cast(pl.Utf8).alias("unit_source"),
            )
            .select(
                ["patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"]
            )
        )

    def _events_lab(self) -> pl.LazyFrame:
        if not self._source_exists("dtLab.csv"):
            return empty_frame("events_long")
        arm = (
            self._scan("dtLab.csv")
            .filter(pl.col("Item").is_in(list(ZIGONG_LAB_ITEM_TO_CONCEPT.keys())))
            .with_columns(
                pl.col("INP_NO").cast(pl.Utf8).alias("original_inp_no"),
                pl.col("Item")
                .replace_strict(ZIGONG_LAB_ITEM_TO_CONCEPT, default=None)
                .alias("concept"),
                self._clean_numeric("LabValue").alias("value"),
                pl.col("Unit").cast(pl.Utf8).alias("unit"),
                pl.col("LabTime").cast(pl.Float64, strict=False).alias("_t"),
            )
        )
        return self._finalize_event_arm(arm, "_t")

    def _events_vital(self) -> pl.LazyFrame:
        """Unpivot the wide dtNursingChart vitals into long events."""
        if not self._source_exists("dtNursingChart.csv"):
            return empty_frame("events_long")
        nc = self._scan_nursing()
        cols = nc.collect_schema().names()
        present = [c for c in _VITAL_COLUMNS if c in cols]
        if not present:
            return empty_frame("events_long")
        base = nc.select(
            pl.col("INP_NO").cast(pl.Utf8).alias("original_inp_no"),
            pl.col("ChartTime").cast(pl.Float64, strict=False).alias("_t"),
            *[self._clean_numeric(c).alias(c) for c in present],
        )
        arm = (
            base.unpivot(
                index=["original_inp_no", "_t"],
                on=present,
                variable_name="_col",
                value_name="value",
            )
            .filter(pl.col("value").is_not_null())
            .with_columns(
                pl.col("_col")
                .replace_strict(ZIGONG_NURSINGCHART_TO_CONCEPT, default=None)
                .alias("concept"),
                pl.col("_col")
                .replace_strict(ZIGONG_NURSINGCHART_SOURCE_UNIT, default=None)
                .alias("unit"),
            )
        )
        return self._finalize_event_arm(arm, "_t")

    def _events_map_derived(self) -> pl.LazyFrame:
        """Derived MAP = (sbp + 2·dbp) / 3 (spec §4.5).

        Built lazily from the harmonised sbp/dbp events: at each (stay, charttime)
        where both sbp and dbp exist, emit a `map` event. The `map` column in the
        raw nursing chart is an empty placeholder, so no real MAP is displaced.
        The derived value passes through the canonical-unit + valid_range clamp.
        Same-charttime sbp/dbp are aggregated by mean (group_by, not pivot — stays
        lazy and never materialises the whole frame).
        """
        sbp_dbp = self._events_vital().filter(pl.col("concept").is_in(["sbp", "dbp"]))
        agg = sbp_dbp.group_by(["patient_id", "stay_id", "charttime"]).agg(
            pl.col("value").filter(pl.col("concept") == "sbp").mean().alias("sbp"),
            pl.col("value").filter(pl.col("concept") == "dbp").mean().alias("dbp"),
        )
        return (
            agg.filter(pl.col("sbp").is_not_null() & pl.col("dbp").is_not_null())
            .with_columns(
                ((pl.col("sbp") + 2.0 * pl.col("dbp")) / 3.0).alias("value"),
                pl.lit("map").alias("concept"),
                pl.lit(None, dtype=pl.Utf8).alias("unit"),
                pl.lit(None, dtype=pl.Utf8).alias("unit_source"),
            )
            .select(
                ["patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"]
            )
        )

    def _events_gcs_derived(self) -> pl.LazyFrame:
        """Summed GCS = E + M + V (spec §4.5).

        Temporal-alignment rule: SAME-ChartTime triplets only. The three
        components are charted together on the same nursing-chart row, so we
        recover (E, M, V) per (INP_NO, ChartTime) row, require all three present,
        and sum to a GCS in [3, 15]. Charts a single derived `gcs` event per row.
        """
        if not self._source_exists("dtNursingChart.csv"):
            return empty_frame("events_long")
        nc = self._scan_nursing()
        cols = nc.collect_schema().names()
        needed = (_ZIGONG_GCS_EYE_COLUMN, _ZIGONG_GCS_MOTOR_COLUMN, _ZIGONG_GCS_VERBAL_COLUMN)
        if not all(c in cols for c in needed):
            return empty_frame("events_long")

        def comp(column: str, lo: int, hi: int) -> pl.Expr:
            return (
                pl.col(column)
                .cast(pl.Utf8)
                .str.split(_ZIGONG_GCS_SEP)
                .list.get(0)
                .str.strip_chars()
                .str.extract(r"^(\d+)", 1)
                .cast(pl.Int64, strict=False)
            )

        e = comp(_ZIGONG_GCS_EYE_COLUMN, *_ZIGONG_GCS_EYE_RANGE)
        m = comp(_ZIGONG_GCS_MOTOR_COLUMN, *_ZIGONG_GCS_MOTOR_RANGE)
        v = comp(_ZIGONG_GCS_VERBAL_COLUMN, *_ZIGONG_GCS_VERBAL_RANGE)
        e_ok = e.is_between(*_ZIGONG_GCS_EYE_RANGE)
        m_ok = m.is_between(*_ZIGONG_GCS_MOTOR_RANGE)
        v_ok = v.is_between(*_ZIGONG_GCS_VERBAL_RANGE)
        arm = (
            nc.select(
                pl.col("INP_NO").cast(pl.Utf8).alias("original_inp_no"),
                pl.col("ChartTime").cast(pl.Float64, strict=False).alias("_t"),
                e.alias("_e"),
                m.alias("_m"),
                v.alias("_v"),
                e_ok.alias("_e_ok"),
                m_ok.alias("_m_ok"),
                v_ok.alias("_v_ok"),
            )
            .filter(pl.col("_e_ok") & pl.col("_m_ok") & pl.col("_v_ok"))
            .with_columns(
                (pl.col("_e") + pl.col("_m") + pl.col("_v")).cast(pl.Float64).alias("value"),
                pl.lit("gcs").alias("concept"),
                pl.lit(None, dtype=pl.Utf8).alias("unit"),
            )
        )
        return self._finalize_event_arm(arm, "_t")

    def read_events_long(self, concepts: list[str] | None = None) -> pl.LazyFrame:
        """Canonical events_long: labs ⊕ vitals ⊕ derived MAP ⊕ derived GCS,
        then unit conversion + valid_range clamp."""
        del concepts
        arms = [
            self._events_lab(),
            self._events_vital(),
            self._events_map_derived(),
            self._events_gcs_derived(),
        ]
        concat = pl.concat(arms, how="vertical_relaxed")
        converted = _apply_unit_conversion_zigong(concat)
        return converted.with_columns(pl.col("value").cast(pl.Float32)).select(
            ["patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"]
        )

    def read_meds(self) -> pl.LazyFrame:
        """Canonical meds (SCHEMA_MEDS). dtDrugs is keyed on PATIENT_ID only, so
        each row reaches a stay through the dtBaseline crosswalk."""
        if not self._source_exists("dtDrugs.csv"):
            return empty_frame("meds")
        m = self._scan("dtDrugs.csv").filter(pl.col("DrugName").is_not_null())
        return (
            m.join(
                self._event_join_keys(),
                left_on="PATIENT_ID",
                right_on="original_patient_id",
                how="inner",
            )
            .filter(
                (pl.col("Drug_time").cast(pl.Float64, strict=False) >= pl.col("admit_hours"))
                & (pl.col("Drug_time").cast(pl.Float64, strict=False) <= pl.col("disch_hours"))
            )
            .with_columns(
                self._offset_hours_to_time("Drug_time").alias("starttime"),
                pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("endtime"),
                pl.col("DrugName").cast(pl.Utf8).alias("drug"),
                pl.lit(None, dtype=pl.Float32).alias("dose"),
                pl.lit(None, dtype=pl.Utf8).alias("dose_unit"),
                pl.lit(None, dtype=pl.Utf8).alias("route"),
                _classify_zigong_drug(pl.col("DrugName")).alias("drug_class"),
            )
            .select(list(TABLES["meds"][0].keys()))
        )

    def read_interventions(self) -> pl.LazyFrame:
        """Canonical interventions (SCHEMA_INTERVENTIONS). mech_vent where any
        dtNursingChart ventilator column (mode / intubation depth) is populated."""
        if not self._source_exists("dtNursingChart.csv"):
            return empty_frame("interventions")
        nc = self._scan_nursing()
        cols = nc.collect_schema().names()
        present = [c for c in _ZIGONG_VENT_COLUMNS if c in cols]
        if not present:
            return empty_frame("interventions")
        any_vent = pl.any_horizontal(
            *[pl.col(c).cast(pl.Utf8).str.strip_chars().is_not_null() for c in present]
        )
        arm = nc.select(
            pl.col("INP_NO").cast(pl.Utf8).alias("original_inp_no"),
            pl.col("ChartTime").cast(pl.Float64, strict=False).alias("_t"),
            *[c for c in present],
        ).filter(any_vent)
        return (
            arm.join(
                self._event_join_keys(),
                on="original_inp_no",
                how="inner",
            )
            .filter(
                (pl.col("_t") >= pl.col("admit_hours")) & (pl.col("_t") <= pl.col("disch_hours"))
            )
            .with_columns(
                self._offset_hours_to_time("_t").alias("starttime"),
                pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("endtime"),
                pl.lit("mech_vent").alias("intervention"),
            )
            .select(list(TABLES["interventions"][0].keys()))
        )

    def read_diagnoses(self) -> pl.LazyFrame:
        """Canonical diagnoses (SCHEMA_DIAGNOSES). dtICD codes are Chinese
        ICD-10 variants (e.g. `J32.000X001`) stripped to the 3-char root;
        morphology `M…/…` codes are dropped (spec §2)."""
        if not self._source_exists("dtICD.csv"):
            return empty_frame("diagnoses")
        keys = self._stays_with_keys().select(["original_inp_no", "patient_id", "stay_id"])
        codes = self._scan("dtICD.csv").filter(pl.col("ICD_Code").is_not_null())
        return (
            codes.with_columns(pl.col("INP_NO").cast(pl.Utf8).alias("original_inp_no"))
            .filter(~pl.col("ICD_Code").cast(pl.Utf8).str.contains(r"^[A-Za-z]\d{3,}/"))
            .with_columns(
                pl.col("ICD_Code")
                .cast(pl.Utf8)
                .str.extract(r"^([A-Za-z]\d{2})", 1)
                .alias("icd_code")
            )
            .filter(pl.col("icd_code").is_not_null())
            .join(keys, on="original_inp_no", how="inner")
            .with_columns(
                pl.lit("10").alias("icd_version"),
                pl.col("Diagnosis_seq").cast(pl.Int32, strict=False).alias("diagnosis_position"),
            )
            .select(list(TABLES["diagnoses"][0].keys()))
        )

    def read_notes(self) -> pl.LazyFrame:
        return empty_frame("notes")

    def read_microbio(self) -> pl.LazyFrame:
        return empty_frame("microbio")

    def read_abx_duration(self) -> pl.LazyFrame:
        """Canonical abx_duration (SCHEMA_ABX_DURATION) — antibiotic admin
        intervals. Surrogate path (like NWICU/OMIX/SICdb): filter meds to
        drug_class == 'antibiotic'."""
        return (
            self.read_meds()
            .filter(pl.col("drug_class") == "antibiotic")
            .select(list(TABLES["abx_duration"][0].keys()))
        )

def _classify_zigong_drug(name_col: pl.Expr) -> pl.Expr:
    """substring(lowercased drug name) → drug_class; default 'other'."""
    low = name_col.cast(pl.Utf8).str.to_lowercase()
    expr = pl.lit("other")
    for substr, cls in reversed(list(ZIGONG_DRUG_CLASS_HINTS.items())):
        expr = pl.when(low.str.contains(substr, literal=True)).then(pl.lit(cls)).otherwise(expr)
    return expr.cast(pl.Utf8)

def _apply_unit_conversion_zigong(events: pl.LazyFrame) -> pl.LazyFrame:
    """Scale (concept, src_unit) values, relabel `unit` to canonical, clamp to
    valid_range. Mirrors the SICdb/OMIX inline path. Derived MAP/GCS carry a
    null source unit and pass through the scaling unchanged."""
    from critical_mm.concepts import CONCEPTS_BY_NAME
    from critical_mm.datasets._units_helper import _VALID_RANGE_LOOKUP

    scaled = pl.col("value")
    for (concept, src_unit_lower), mult in ZIGONG_UNIT_MULTIPLIERS.items():
        match = (pl.col("concept") == concept) & (
            pl.col("unit").str.to_lowercase() == src_unit_lower
        )
        scaled = pl.when(match).then(pl.col("value") * mult).otherwise(scaled)

    unit_absent = pl.col("unit").is_null() | (pl.col("unit").str.strip_chars() == "")
    for concept, mult in ZIGONG_FRACTION_TO_PERCENT.items():
        match = (pl.col("concept") == concept) & unit_absent
        scaled = pl.when(match).then(scaled * mult).otherwise(scaled)

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
