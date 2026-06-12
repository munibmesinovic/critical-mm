"""Materialisation helpers for the notes modality: aligned view + coverage row."""

from __future__ import annotations

import polars as pl

from critical_mm.modalities.align import normalize_stay_id
from critical_mm.modalities.notes_base import align_notes_to_cohort

def build_notes_aligned_view(
    timed: pl.LazyFrame, stays: pl.LazyFrame, split: pl.LazyFrame, dataset: str
) -> pl.LazyFrame:
    """Aligned per-task view: signed-delta contract + split join.

    stays needs: patient_id, stay_id, admit_time, discharge_time, hadm_id (hadm_id
    null for stay-bound datasets). split needs: stay_id, split.
    """
    timed = timed.with_columns(pl.col("bound_stay_id").cast(pl.Utf8))
    cohort = stays.select(
        "patient_id",
        "stay_id",
        pl.col("admit_time").alias("intime"),
        "discharge_time",
        "hadm_id",
    )
    aligned = align_notes_to_cohort(timed, cohort)
    split_n = split.with_columns(normalize_stay_id(pl.col("stay_id"), dataset).alias("stay_id"))
    aligned_n = aligned.with_columns(normalize_stay_id(pl.col("stay_id"), dataset).alias("stay_id"))
    return aligned_n.join(split_n.select("stay_id", "split"), on="stay_id", how="inner")

def notes_coverage_row(
    task: str, dataset: str, view: pl.DataFrame, n_cohort_stays: int
) -> dict[str, object]:
    """One coverage row for a materialised (task, dataset) notes view."""
    n_stays = view["stay_id"].n_unique()
    if view.height:
        bt = view.group_by("note_type").len()
        by_note_type = {
            str(k): int(v)
            for k, v in zip(bt["note_type"].to_list(), bt["len"].to_list(), strict=True)
        }
        n_within = int(view["within_window"].sum())
    else:
        by_note_type = {}
        n_within = 0
    return {
        "task": task,
        "dataset": dataset,
        "n_cohort_stays": n_cohort_stays,
        "n_stays_with_note": n_stays,
        "frac_stays_with_note": (n_stays / n_cohort_stays) if n_cohort_stays else 0.0,
        "n_notes": view.height,
        "n_within_window": n_within,
        "by_note_type": by_note_type,
    }
