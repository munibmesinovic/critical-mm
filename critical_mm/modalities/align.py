"""Cohort alignment for diagnosis modality — the single leakage invariant.

A code enters a stay's stream iff ``knowable_time <= intime``. Patient-bound
rows (bound_stay_id null) attach to every stay of the patient; stay-bound
rows (eICU) attach only to their own stay.
"""

from __future__ import annotations

import polars as pl

_DS_PREFIXES = {
    "mimic_iv": "miiv",
    "miiv": "miiv",
    "eicu": "eicu",
    "nwicu": "nwicu",
    "omix": "omix",
}

def normalize_stay_id(expr: pl.Series | pl.Expr, dataset: str) -> pl.Expr:
    """Return a String stay_id of the form ``<prefix>_<id>``.

    Cohort/timed frames carry ``<prefix>_<id>`` strings; many split/outc
    parquets carry a bare Int32 id. This normalizes both to the prefixed
    string so joins never silently miss.
    """
    prefix = _DS_PREFIXES.get(dataset, dataset)
    e: pl.Expr = pl.lit(expr) if isinstance(expr, pl.Series) else expr
    e = e.cast(pl.Utf8)
    already = e.str.starts_with(f"{prefix}_")
    return pl.when(already).then(e).otherwise(pl.lit(f"{prefix}_") + e)

def align_to_cohort(timed: pl.LazyFrame, cohort: pl.LazyFrame) -> pl.LazyFrame:
    """Apply ``knowable_time <= intime``; emit ALIGNED_SCHEMA.

    cohort columns required: patient_id, stay_id, intime (Datetime UTC).
    """
    pb = timed.filter(pl.col("bound_stay_id").is_null()).join(
        cohort.select("patient_id", "stay_id", "intime"), on="patient_id", how="inner"
    )
    sb = (
        timed.filter(pl.col("bound_stay_id").is_not_null())
        .join(
            cohort.select(pl.col("stay_id").alias("__cstay"), "intime"),
            left_on="bound_stay_id",
            right_on="__cstay",
            how="inner",
        )
        .with_columns(pl.col("bound_stay_id").alias("stay_id"))
    )
    joined = pl.concat([pb, sb], how="diagonal")
    kept = joined.filter(pl.col("knowable_time") <= pl.col("intime"))
    with_delta = kept.with_columns(
        ((pl.col("intime") - pl.col("knowable_time")).dt.total_seconds() / 3600.0).alias("delta_h"),
        pl.col("knowable_time").min().over(["stay_id", "source_admission_id"]).alias("__adm_kt"),
    )
    with_idx = with_delta.with_columns(
        (pl.col("__adm_kt").rank("dense").over("stay_id") - 1)
        .cast(pl.Int32)
        .alias("prior_visit_idx")
    )
    return with_idx.select("stay_id", "code", "code_system", "origin", "delta_h", "prior_visit_idx")

def attach_split(aligned: pl.LazyFrame, split: pl.LazyFrame, dataset: str) -> pl.LazyFrame:
    """Inner-join the locked split onto an aligned stream.

    Normalizes both stay_id sides to the prefixed String form first, so an
    Int32-bare split parquet still matches the String-prefixed aligned stream.
    """
    split_norm = split.with_columns(normalize_stay_id(pl.col("stay_id"), dataset).alias("stay_id"))
    aligned_norm = aligned.with_columns(
        normalize_stay_id(pl.col("stay_id"), dataset).alias("stay_id")
    )
    return aligned_norm.join(split_norm.select("stay_id", "split"), on="stay_id", how="inner")
