"""Base contract for note-modality readers/encoders + the timed/aligned schemas.

A note reader aligns a clinical free-text (or serialized semi-structured) stream
to the locked task cohorts. The leakage contract is the signed ``delta_h_signed``
(``(knowable_time - intime)/3600``): pre-admission context is ``<= 0``;
within-window is ``> 0`` and bounded above by the stay's ICU discharge. Discharge
summaries are admitted to the pre-admission view ONLY, from a prior admission.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import polars as pl

from critical_mm.schema import Schema

_DT_UTC = pl.Datetime("us", "UTC")

NOTE_TIMED_SCHEMA: Schema = {
    "patient_id": pl.Utf8(),
    "bound_stay_id": pl.Utf8(),
    "source_admission_id": pl.Utf8(),
    "hadm_id": pl.Utf8(),
    "note_id": pl.Utf8(),
    "note_type": pl.Utf8(),
    "origin": pl.Utf8(),
    "language": pl.Utf8(),
    "text": pl.Utf8(),
    "knowable_time": _DT_UTC,
}

NOTE_ALIGNED_SCHEMA: Schema = {
    "stay_id": pl.Utf8(),
    "note_id": pl.Utf8(),
    "note_type": pl.Utf8(),
    "origin": pl.Utf8(),
    "language": pl.Utf8(),
    "delta_h_signed": pl.Float64(),
    "within_window": pl.Boolean(),
    "prior_visit_idx": pl.Int32(),
}

def empty_note_timed() -> pl.LazyFrame:
    """Zero-row LazyFrame with the canonical note timed schema."""
    return pl.LazyFrame(schema=NOTE_TIMED_SCHEMA)

class NoteReader(ABC):
    """Contract: produce a task-independent note timed frame for one modality."""

    MODALITY_NAME: ClassVar[str]

    @abstractmethod
    def read_timed(self, dataset: str) -> pl.LazyFrame:
        """Return the NOTE_TIMED_SCHEMA frame for ``dataset`` (or ``empty_note_timed()``)."""

def align_notes_to_cohort(timed: pl.LazyFrame, cohort: pl.LazyFrame) -> pl.LazyFrame:
    """Apply the signed-delta leakage contract; emit NOTE_ALIGNED_SCHEMA.

    cohort columns required: patient_id, stay_id, intime, discharge_time, hadm_id.
    Keep rule per note:
      pre-admission  : knowable_time <= intime
      within-window  : intime < knowable_time <= discharge_time AND note_type != 'discharge'
    AND a structural exclusion: a 'discharge' note is dropped from a stay whose
    hadm_id equals the note's hadm_id (its own admission).
    """
    cols = ["patient_id", "stay_id", "intime", "discharge_time"]
    cohort_h = cohort.select(*cols, pl.col("hadm_id").alias("__stay_hadm"))
    pb = timed.filter(pl.col("bound_stay_id").is_null()).join(
        cohort_h, on="patient_id", how="inner"
    )
    sb = (
        timed.filter(pl.col("bound_stay_id").is_not_null())
        .join(
            cohort_h.select(
                pl.col("stay_id").alias("__cstay"), "intime", "discharge_time", "__stay_hadm"
            ),
            left_on="bound_stay_id",
            right_on="__cstay",
            how="inner",
        )
        .with_columns(pl.col("bound_stay_id").alias("stay_id"))
    )
    joined = pl.concat([pb, sb], how="diagonal")
    pre = pl.col("knowable_time") <= pl.col("intime")
    win = (
        (pl.col("knowable_time") > pl.col("intime"))
        & (pl.col("knowable_time") <= pl.col("discharge_time"))
        & (pl.col("note_type") != "discharge")
    )
    own_discharge = (pl.col("note_type") == "discharge") & (
        pl.col("hadm_id") == pl.col("__stay_hadm")
    )
    kept = joined.filter((pre | win) & ~own_discharge.fill_null(False))
    with_delta = kept.with_columns(
        ((pl.col("knowable_time") - pl.col("intime")).dt.total_seconds() / 3600.0).alias(
            "delta_h_signed"
        ),
        pl.col("knowable_time").min().over(["stay_id", "source_admission_id"]).alias("__adm_kt"),
    )
    with_idx = with_delta.with_columns(
        (pl.col("delta_h_signed") > 0).alias("within_window"),
        (pl.col("__adm_kt").rank("dense").over("stay_id") - 1)
        .cast(pl.Int32)
        .alias("prior_visit_idx"),
    )
    return with_idx.select(list(NOTE_ALIGNED_SCHEMA.keys()))

def visible_notes_stay_level(aligned: pl.LazyFrame, cutoff_h: float) -> pl.LazyFrame:
    """Stay-level visibility (mortality24, kidney_function): notes with
    delta_h_signed <= cutoff_h (context + within-window up to the observation cutoff)."""
    return aligned.filter(pl.col("delta_h_signed") <= cutoff_h)

def visible_notes_per_hour(aligned: pl.LazyFrame, dyn_grid: pl.LazyFrame) -> pl.LazyFrame:
    """Per-(stay,hour) visibility (aki, sepsis, los): at prediction hour ``h`` a note
    is visible iff delta_h_signed <= h. ``dyn_grid`` has columns (stay_id, hour).

    This mirrors the DL loader's per-timestep row visibility (rows 0..h at hour h).
    Returns the aligned columns + ``hour``.
    """
    return dyn_grid.join(aligned, on="stay_id", how="inner").filter(
        pl.col("delta_h_signed") <= pl.col("hour")
    )

class NoteEncoder(ABC):
    """Contract: encode a list of texts to fixed-dim vectors.

    Concrete encoders are implemented in Plan 2.
    """

    ENCODER_ID: ClassVar[str]

    @abstractmethod
    def encode(self, texts: list[str]) -> object:
        """Return an (n, d) float array for ``texts``.

        Returns np.ndarray; typed loosely here to avoid a numpy import.
        """

