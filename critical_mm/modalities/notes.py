"""Note-modality readers: miiv (prose), omix (Chinese prose), eicu (serialized).

Leakage semantics per source: see spec 2026-06-01 rev2 §3. All readers emit the
canonical NOTE_TIMED_SCHEMA; alignment + the discharge exclusion live in
align_notes_to_cohort (notes_base).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import ClassVar

import polars as pl

from critical_mm.io.paths import raw_path
from critical_mm.modalities.notes_base import NOTE_TIMED_SCHEMA, NoteReader, empty_note_timed
from critical_mm.modalities.notes_clean import (
    clean_text_expr,
    compose_omix_text,
    decode_gb18030,
    drop_implausible_offset_min,
    serialize_eicu_rows,
)
from critical_mm.registry import register_note_reader

_DT_UTC = pl.Datetime("us", "UTC")
_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _map_miiv_notes(raw: pl.LazyFrame) -> pl.LazyFrame:
    """Map a raw discharge/radiology-shaped frame to NOTE_TIMED_SCHEMA (pure, testable).

    raw columns: note_id, subject_id, hadm_id, note_type_raw, note_seq,
    storetime(str|datetime), text.
    """
    st = pl.col("storetime")
    st_dt = (
        pl.when(st.cast(pl.Utf8).is_not_null())
        .then(st.cast(pl.Utf8).str.to_datetime(format=_TS_FMT, time_unit="us", strict=False))
        .otherwise(None)
        .dt.replace_time_zone("UTC")
    )
    note_type = (
        pl.when(pl.col("note_type_raw") == "DS")
        .then(pl.lit("discharge"))
        .otherwise(pl.lit("radiology"))
    )
    return (
        raw.with_columns(
            (pl.lit("miiv_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
            pl.lit(None, dtype=pl.Utf8).alias("bound_stay_id"),
            pl.col("note_id").cast(pl.Utf8).alias("source_admission_id"),
            pl.col("hadm_id").cast(pl.Utf8).alias("hadm_id"),
            pl.col("note_id").cast(pl.Utf8).alias("note_id"),
            note_type.alias("note_type"),
            note_type.alias("origin"),
            pl.lit("en").alias("language"),
            clean_text_expr("text").alias("text"),
            st_dt.alias("knowable_time"),
        )
        .filter(pl.col("knowable_time").is_not_null())
        .select(list(NOTE_TIMED_SCHEMA.keys()))
        .sort("knowable_time", descending=True)
        .unique(subset=["patient_id", "note_type", "text"], keep="first", maintain_order=True)
    )


@register_note_reader("notes")
class MiivNoteReader(NoteReader):
    MODALITY_NAME: ClassVar[str] = "notes"

    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    def read_timed(self, dataset: str) -> pl.LazyFrame:
        if dataset in ("mimic_iv", "miiv"):
            return self._miiv()
        return empty_note_timed()

    def _miiv(self) -> pl.LazyFrame:
        note_dir = self.repo_root / "data" / "raw" / "mimic-iv-note-2.2" / "note"
        if not (note_dir / "discharge.csv.gz").exists():
            return empty_note_timed()

        def scan(rel: str) -> pl.LazyFrame:
            return (
                pl.scan_csv(note_dir / rel, infer_schema_length=10_000, ignore_errors=True)
                .select(
                    "note_id", "subject_id", "hadm_id", "note_type", "note_seq", "storetime", "text"
                )
                .rename({"note_type": "note_type_raw"})
            )

        disch = scan("discharge.csv.gz")
        rad = scan("radiology.csv.gz")
        return _map_miiv_notes(pl.concat([disch, rad], how="diagonal"))


@register_note_reader("notes_omix")
class OmixNoteReader(NoteReader):
    MODALITY_NAME: ClassVar[str] = "notes_omix"
    _OMIX_ANCHOR: ClassVar[_dt.datetime] = _dt.datetime(2012, 1, 1, tzinfo=_dt.UTC)

    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    def read_timed(self, dataset: str) -> pl.LazyFrame:
        if dataset != "omix":
            return empty_note_timed()
        try:
            report = raw_path("omix") / "DataTable2" / "ExamReport.csv"
        except FileNotFoundError:
            return empty_note_timed()
        stays_p = self.repo_root / "data" / "processed" / "base_cohort" / "omix" / "stays.parquet"
        if not report.exists() or not stays_p.exists():
            return empty_note_timed()
        df = decode_gb18030(
            report,
            infer_schema_length=10_000,
            ignore_errors=True,
            schema_overrides={"Hospital_ID": pl.Utf8},
        )
        if df.is_empty():
            return empty_note_timed()
        stays = (
            pl.scan_parquet(stays_p)
            .select(
                pl.col("hospital_id").cast(pl.Utf8),
                pl.col("patient_id").cast(pl.Utf8),
                pl.col("stay_id").cast(pl.Utf8),
            )
            .unique(subset=["hospital_id"])
        )
        anchor = pl.lit(self._OMIX_ANCHOR).cast(_DT_UTC)
        stamped_min = (
            (pl.col("ExamReport_DateTime").cast(pl.Float64) * 1440.0).round(0).cast(pl.Int64)
        )
        lf = (
            compose_omix_text(df.lazy())
            .with_columns(
                pl.col("Hospital_ID").cast(pl.Utf8).alias("__hid"),
            )
            .join(stays, left_on="__hid", right_on="hospital_id", how="inner")
        )
        lf = lf.sort(
            ["__hid", "ExamReport_DateTime", "ExamReport_Category", "ExamReport_item_Eng", "text"]
        ).with_columns(
            pl.int_range(0, pl.len()).over(["__hid", "ExamReport_DateTime"]).alias("__seq")
        )
        return (
            lf.with_columns(
                pl.col("patient_id"),
                pl.col("stay_id").alias("bound_stay_id"),
                pl.col("stay_id").alias("source_admission_id"),
                pl.lit(None, dtype=pl.Utf8).alias("hadm_id"),
                (
                    pl.lit("omix_exam_")
                    + pl.col("__hid")
                    + pl.lit("_")
                    + pl.col("ExamReport_DateTime").cast(pl.Utf8)
                    + pl.lit("_")
                    + pl.col("__seq").cast(pl.Utf8)
                ).alias("note_id"),
                pl.lit("exam_report").alias("note_type"),
                pl.lit("exam_report").alias("origin"),
                pl.lit("zh").alias("language"),
                anchor.dt.offset_by(stamped_min.cast(pl.Utf8) + pl.lit("m")).alias("knowable_time"),
            )
            .filter(pl.col("knowable_time").is_not_null())
            .select(list(NOTE_TIMED_SCHEMA.keys()))
        )


_EICU_SOURCES: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "carePlanGeneral.csv.gz",
        "cplitemoffset",
        "cplgroup",
        "cplitemvalue",
        "eicu_careplan",
        "cpgen",
    ),
    (
        "carePlanInfectiousDisease.csv.gz",
        "cplinfectdiseaseoffset",
        "infectdiseasesite",
        "infectdiseaseassessment",
        "eicu_careplan",
        "cpinfect",
    ),
    (
        "nurseAssessment.csv.gz",
        "nurseassessoffset",
        "celllabel",
        "cellattributevalue",
        "eicu_assessment",
        "nurse",
    ),
    (
        "physicalExam.csv.gz",
        "physicalexamoffset",
        "physicalexampath",
        "physicalexamvalue",
        "eicu_exam",
        "pexam",
    ),
)


@register_note_reader("notes_eicu")
class EicuNoteReader(NoteReader):
    MODALITY_NAME: ClassVar[str] = "notes_eicu"
    _EICU_ANCHOR: ClassVar[_dt.datetime] = _dt.datetime(2014, 1, 1, tzinfo=_dt.UTC)

    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    def read_timed(self, dataset: str) -> pl.LazyFrame:
        if dataset != "eicu":
            return empty_note_timed()
        raw = raw_path("eicu")
        if not (raw / "carePlanGeneral.csv.gz").exists():
            return empty_note_timed()
        pid = pl.scan_csv(
            raw / "patient.csv.gz", infer_schema_length=10_000, ignore_errors=True
        ).select("patientunitstayid", "uniquepid")
        anchor = pl.lit(self._EICU_ANCHOR).cast(_DT_UTC)
        parts: list[pl.LazyFrame] = []
        for rel, off_col, lab_col, val_col, ntype, src_tag in _EICU_SOURCES:
            if not (raw / rel).exists():
                continue
            src = (
                pl.scan_csv(raw / rel, infer_schema_length=10_000, ignore_errors=True)
                .select(
                    pl.col("patientunitstayid"),
                    pl.col(off_col).cast(pl.Int64).alias("offset_min"),
                    pl.col(lab_col).cast(pl.Utf8).alias("label"),
                    pl.col(val_col).cast(pl.Utf8).alias("value"),
                )
                .filter(pl.col("offset_min").is_not_null())
            )
            src = drop_implausible_offset_min(src, col="offset_min").with_columns(
                (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                (pl.col("offset_min") // 60).alias("hour"),
            )
            doc = serialize_eicu_rows(src, note_type=ntype)
            doc = doc.with_columns(pl.lit(src_tag).alias("__src"))
            bucket_off = src.group_by(["stay_id", "hour"]).agg(
                pl.col("offset_min").min().alias("__minoff")
            )
            doc = doc.join(bucket_off, on=["stay_id", "hour"], how="left")
            parts.append(doc)
        if not parts:
            return empty_note_timed()
        alldocs = (
            pl.concat(parts, how="diagonal")
            .with_columns(pl.col("stay_id").str.slice(5).cast(pl.Int64).alias("__puid"))
            .join(pid, left_on="__puid", right_on="patientunitstayid", how="left")
        )
        return (
            alldocs.with_columns(
                (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                pl.col("stay_id").alias("bound_stay_id"),
                pl.col("stay_id").alias("source_admission_id"),
                pl.lit(None, dtype=pl.Utf8).alias("hadm_id"),
                (
                    pl.col("stay_id")
                    + pl.lit("_")
                    + pl.col("hour").cast(pl.Utf8)
                    + pl.lit("_")
                    + pl.col("__src")
                ).alias("note_id"),
                pl.col("note_type"),
                pl.col("note_type").alias("origin"),
                pl.lit("en").alias("language"),
                pl.col("text"),
                anchor.dt.offset_by(pl.col("__minoff").cast(pl.Utf8) + pl.lit("m")).alias(
                    "knowable_time"
                ),
            )
            .filter(pl.col("knowable_time").is_not_null())
            .select(list(NOTE_TIMED_SCHEMA.keys()))
        )
