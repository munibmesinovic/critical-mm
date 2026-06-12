"""Diagnosis-code modality: per-dataset timed frames (pre-admission context).

knowable_time per source (spec 2026-05-31 rev 2, table sec.3):
- miiv/nwicu billing: the code's hadm dischtime
- eicu: diagnosis offset<=0 (active), admissionDx/pastHistory @ intime (anchor)
- omix: anchor + day-offset (stamped), else that admission's dischtime
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import ClassVar

import polars as pl

from critical_mm.io.paths import raw_path
from critical_mm.modalities.base import TIMED_SCHEMA, ModalityReader, empty_timed
from critical_mm.registry import register_modality

_DT_UTC = pl.Datetime("us", "UTC")

@register_modality("diagnoses")
class DiagnosesModalityReader(ModalityReader):
    MODALITY_NAME: ClassVar[str] = "diagnoses"
    _EICU_ANCHOR: ClassVar[_dt.datetime] = _dt.datetime(2014, 1, 1, tzinfo=_dt.UTC)
    _OMIX_ANCHOR: ClassVar[_dt.datetime] = _dt.datetime(2012, 1, 1, tzinfo=_dt.UTC)
    _SICDB_ANCHOR: ClassVar[_dt.datetime] = _dt.datetime(2013, 1, 1, tzinfo=_dt.UTC)

    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    def read_timed(self, dataset: str) -> pl.LazyFrame:
        if dataset in ("mimic_iv", "miiv"):
            return self._miiv_nwicu_billing("mimic_iv", "miiv")
        if dataset == "nwicu":
            return self._miiv_nwicu_billing("nwicu", "nwicu")
        if dataset == "eicu":
            return self._eicu()
        if dataset == "omix":
            return self._omix()
        if dataset == "sicdb":
            return self._sicdb()
        if dataset == "synthetic":
            return self._synthetic()
        return empty_timed()

    def _synthetic(self) -> pl.LazyFrame:
        """Deterministic in-memory timed frame for the end-to-end leakage test.

        Covers patient-bound billing (prior vs current) and stay-bound active
        (pre-admit vs within-window) + admission_dx. No real data / no PHI; a
        separate generator that does not touch the SyntheticReader's locked
        ``synthetic_summary.json`` fixture.
        """

        def d(day: int) -> _dt.datetime:
            return _dt.datetime(2020, 1, day, tzinfo=_dt.UTC)

        rows = {
            "patient_id": ["syn_P1", "syn_P1", "syn_P1", "syn_P1", "syn_P1"],
            "bound_stay_id": [None, None, "syn_S2", "syn_S2", "syn_S2"],
            "source_admission_id": [
                "syn_hadm_prior",
                "syn_hadm_curr",
                "syn_S2",
                "syn_S2",
                "syn_S2",
            ],
            "code": ["E11", "N17", "I10", "J45", "ADMDX"],
            "code_system": ["icd10", "icd10", "icd10", "icd10", "apache"],
            "knowable_time": [d(5), d(20), d(9), d(15), d(10)],
            "origin": ["billing", "billing", "active", "active", "admission_dx"],
        }
        return pl.DataFrame(rows, schema=TIMED_SCHEMA).lazy()

    def _miiv_nwicu_billing(self, dataset: str, prefix: str) -> pl.LazyFrame:
        raw = raw_path(dataset)

        def scan(rel: str) -> pl.LazyFrame:
            return pl.scan_csv(
                raw / rel,
                infer_schema_length=10_000,
                null_values=["", "NA"],
                ignore_errors=True,
            )

        diag = (
            scan("hosp/diagnoses_icd.csv.gz")
            .select("subject_id", "hadm_id", "icd_code", "icd_version")
            .filter(pl.col("icd_code").is_not_null())
        )
        adm = scan("hosp/admissions.csv.gz").select("hadm_id", "dischtime")
        joined = diag.join(adm, on="hadm_id", how="inner")
        return (
            joined.with_columns(
                (pl.lit(f"{prefix}_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
                pl.lit(None, dtype=pl.Utf8).alias("bound_stay_id"),
                (pl.lit(f"{prefix}_hadm_") + pl.col("hadm_id").cast(pl.Utf8)).alias(
                    "source_admission_id"
                ),
                pl.col("icd_code").cast(pl.Utf8).alias("code"),
                pl.when(pl.col("icd_version").cast(pl.Utf8) == "9")
                .then(pl.lit("icd9"))
                .otherwise(pl.lit("icd10"))
                .alias("code_system"),
                pl.col("dischtime")
                .str.to_datetime(format="%Y-%m-%d %H:%M:%S", time_unit="us", strict=False)
                .dt.replace_time_zone("UTC")
                .alias("knowable_time"),
                pl.lit("billing").alias("origin"),
            )
            .filter(pl.col("knowable_time").is_not_null())
            .select(list(TIMED_SCHEMA.keys()))
        )

    def _eicu(self) -> pl.LazyFrame:
        raw = raw_path("eicu")

        def scan(rel: str) -> pl.LazyFrame:
            return pl.scan_csv(
                raw / rel,
                infer_schema_length=10_000,
                null_values=["", "NA"],
                ignore_errors=True,
            )

        anchor = pl.lit(self._EICU_ANCHOR).cast(_DT_UTC)
        pid = scan("patient.csv.gz").select("patientunitstayid", "uniquepid")

        def with_keys(lf: pl.LazyFrame) -> pl.LazyFrame:
            return lf.join(pid, on="patientunitstayid", how="left").with_columns(
                (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias(
                    "bound_stay_id"
                ),
            )

        first_code = pl.col("icd9code").str.split(",").list.get(0).str.strip_chars()
        active = (
            with_keys(
                scan("diagnosis.csv.gz")
                .filter(pl.col("icd9code").is_not_null() & (pl.col("diagnosisoffset") <= 0))
                .select("patientunitstayid", "icd9code", "diagnosisoffset")
            )
            .with_columns(
                pl.col("bound_stay_id").alias("source_admission_id"),
                first_code.alias("code"),
                pl.when(first_code.str.contains(r"^[A-Z]"))
                .then(pl.lit("icd10"))
                .otherwise(pl.lit("icd9"))
                .alias("code_system"),
                anchor.dt.offset_by(
                    pl.col("diagnosisoffset").cast(pl.Int64).cast(pl.Utf8) + pl.lit("m")
                ).alias("knowable_time"),
                pl.lit("active").alias("origin"),
            )
            .filter(pl.col("code").is_not_null())
            .select(list(TIMED_SCHEMA.keys()))
        )

        admdx = (
            with_keys(
                scan("admissionDx.csv.gz")
                .select("patientunitstayid", "admitdxpath")
                .filter(pl.col("admitdxpath").is_not_null())
            )
            .with_columns(
                pl.col("bound_stay_id").alias("source_admission_id"),
                pl.col("admitdxpath").cast(pl.Utf8).alias("code"),
                pl.lit("apache").alias("code_system"),
                anchor.alias("knowable_time"),
                pl.lit("admission_dx").alias("origin"),
            )
            .select(list(TIMED_SCHEMA.keys()))
        )

        past = (
            with_keys(
                scan("pastHistory.csv.gz")
                .select("patientunitstayid", "pasthistorypath")
                .filter(pl.col("pasthistorypath").is_not_null())
            )
            .with_columns(
                pl.col("bound_stay_id").alias("source_admission_id"),
                pl.col("pasthistorypath").cast(pl.Utf8).alias("code"),
                pl.lit("past_history").alias("code_system"),
                anchor.alias("knowable_time"),
                pl.lit("past_history").alias("origin"),
            )
            .select(list(TIMED_SCHEMA.keys()))
        )

        return pl.concat([active, admdx, past], how="diagonal")

    def _omix(self) -> pl.LazyFrame:
        raw = raw_path("omix")
        stays_path = (
            self.repo_root / "data" / "processed" / "base_cohort" / "omix" / "stays.parquet"
        )
        if not stays_path.exists():
            return empty_timed()
        stays = (
            pl.scan_parquet(stays_path)
            .select(
                pl.col("hospital_id").cast(pl.Utf8),
                pl.col("patient_id").cast(pl.Utf8),
                pl.col("stay_id").cast(pl.Utf8),
                "discharge_time",
            )
            .unique(subset=["hospital_id"])
        )
        anchor = pl.lit(self._OMIX_ANCHOR).cast(_DT_UTC)
        dx = (
            pl.scan_csv(
                raw / "DataTable2" / "Diagnosis.csv",
                infer_schema_length=10_000,
                null_values=["", "NA"],
                ignore_errors=True,
            )
            .select("Hospital_ID", "ICD10_code", "Diagnosis_DateTime")
            .filter(pl.col("ICD10_code").is_not_null())
            .with_columns(pl.col("ICD10_code").str.split(";").alias("__codes"))
            .explode("__codes")
            .with_columns(pl.col("__codes").str.extract(r"^([A-Z]\d{2}(?:\.\d+)?)").alias("code"))
            .filter(pl.col("code").is_not_null())
            .with_columns(pl.col("Hospital_ID").cast(pl.Utf8).alias("__hid"))
            .join(stays, left_on="__hid", right_on="hospital_id", how="inner")
        )
        stamped_min = (
            (pl.col("Diagnosis_DateTime").cast(pl.Float64) * 1440.0).round(0).cast(pl.Int64)
        )
        knowable = (
            pl.when(pl.col("Diagnosis_DateTime").is_not_null())
            .then(anchor.dt.offset_by(stamped_min.cast(pl.Utf8) + pl.lit("m")))
            .otherwise(pl.col("discharge_time").cast(_DT_UTC))
        )
        origin = (
            pl.when(pl.col("Diagnosis_DateTime").is_not_null())
            .then(pl.lit("active"))
            .otherwise(pl.lit("billing"))
        )
        return (
            dx.with_columns(
                pl.col("patient_id"),
                pl.col("stay_id").alias("bound_stay_id"),
                pl.col("stay_id").alias("source_admission_id"),
                pl.col("code"),
                pl.lit("icd10").alias("code_system"),
                knowable.alias("knowable_time"),
                origin.alias("origin"),
            )
            .filter(pl.col("knowable_time").is_not_null())
            .select(list(TIMED_SCHEMA.keys()))
        )

    def _sicdb(self) -> pl.LazyFrame:
        """SICdb ICD10Main = the single primary admission diagnosis per stay.

        It is knowable at admission and carries no per-diagnosis timestamp, so
        ``knowable_time`` is anchored at the reader's synthetic origin
        (2013-01-01). Every stay's ``admit_time`` is ``anchor + ICUOffset`` with
        ``ICUOffset >= 0``, so ``knowable_time <= intime`` holds for all stays —
        the modality is leakage-safe. Source = the harmonised interim
        (``stay_id`` already canonical ``sicdb_<CaseID>``).
        """
        dx_path = self.repo_root / "data" / "interim" / "sicdb" / "diagnoses.parquet"
        if not dx_path.exists():
            return empty_timed()
        anchor = pl.lit(self._SICDB_ANCHOR).cast(_DT_UTC)
        return (
            pl.scan_parquet(dx_path)
            .filter(pl.col("icd_code").is_not_null())
            .with_columns(
                pl.col("patient_id").cast(pl.Utf8),
                pl.col("stay_id").cast(pl.Utf8).alias("bound_stay_id"),
                pl.col("stay_id").cast(pl.Utf8).alias("source_admission_id"),
                pl.col("icd_code").cast(pl.Utf8).alias("code"),
                pl.lit("icd10").alias("code_system"),
                anchor.alias("knowable_time"),
                pl.lit("admission_dx").alias("origin"),
            )
            .select(list(TIMED_SCHEMA.keys()))
        )
