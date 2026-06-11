"""Materialization helpers: per-task aligned view + coverage report row."""

from __future__ import annotations

import polars as pl

from critical_mm.modalities.align import align_to_cohort, attach_split


def build_aligned_view(
    timed: pl.LazyFrame, stays: pl.LazyFrame, split: pl.LazyFrame, dataset: str
) -> pl.LazyFrame:
    """Aligned per-task view: invariant + split join. stays needs patient_id, stay_id, admit_time.

    Casts bound_stay_id to Utf8 if the column arrives as Null dtype (all-None
    timed frames inferred from Python literals skip the string type).
    """
    timed = timed.with_columns(pl.col("bound_stay_id").cast(pl.Utf8))
    cohort = stays.select("patient_id", "stay_id", pl.col("admit_time").alias("intime"))
    aligned = align_to_cohort(timed, cohort)
    return attach_split(aligned, split, dataset)


def coverage_row(
    task: str, dataset: str, view: pl.DataFrame, n_cohort_stays: int
) -> dict[str, object]:
    """One coverage report row for a materialized (task, dataset) view.

    Reports the fields needed to judge whether the modality is worth modelling:
    fraction of the task cohort that carries any leakage-safe code, mean codes
    and mean prior-visit depth per covered stay, and the per-origin breakdown.
    """
    n_stays = view["stay_id"].n_unique()
    if view.height:
        by_origin = view.group_by("origin").len()
        origin_counts = {
            str(o): int(c)
            for o, c in zip(by_origin["origin"].to_list(), by_origin["len"].to_list(), strict=True)
        }
        mean_obj = (
            view.group_by("stay_id")
            .agg((pl.col("prior_visit_idx").max() + 1).alias("nvisits"))["nvisits"]
            .mean()
        )
        mean_prior_visits = float(mean_obj) if isinstance(mean_obj, int | float) else 0.0
    else:
        origin_counts = {}
        mean_prior_visits = 0.0
    return {
        "task": task,
        "dataset": dataset,
        "n_cohort_stays": n_cohort_stays,
        "n_stays_with_code": n_stays,
        "frac_stays_with_code": (n_stays / n_cohort_stays) if n_cohort_stays else 0.0,
        "n_codes": view.height,
        "mean_codes_per_stay": (view.height / n_stays) if n_stays else 0.0,
        "mean_prior_visits_per_stay": mean_prior_visits,
        "origins": sorted(origin_counts),
        "origin_counts": origin_counts,
    }
