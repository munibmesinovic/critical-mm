"""Materialisation helpers for the treatment modality: aligned view + coverage.

Mirrors ``critical_mm.modalities.build`` (diagnoses) and
``critical_mm.modalities.notes_build`` (notes): a per-task aligned view builder
plus a coverage-report row. The diagnoses ``build_aligned_view`` / ``coverage_row``
cannot be reused directly because the diagnoses alignment hard-codes the
``code`` / ``code_system`` projection (``align_to_cohort`` selects those columns);
the treatment timed frame instead carries ``treatment`` / ``dose`` / ``dose_unit``.

Treatments are TIME-VARYING (a vasopressor active hour 12→48 of a stay), so —
unlike diagnoses (pre-admission context, ``knowable_time <= intime``) — the
alignment follows the NOTES contract: keep the full in-stay temporal signal with
a SIGNED ``delta_h = (knowable_time - intime)/3600`` (in-stay events positive,
recent pre-admission history negative), bounded to the stay, and defer the
leakage/visibility gate to consumption (the Plan-C per-hour block builder gates
``delta_h <= h``; the stay-level builder gates pre-cutoff). We do NOT apply a
``knowable_time <= intime`` filter here — that would discard every in-stay
treatment. ``normalize_stay_id`` is reused for the split join verbatim.
"""

from __future__ import annotations

import polars as pl

from critical_mm.modalities.align import normalize_stay_id
from critical_mm.modalities.base import TREATMENT_ALIGNED
from critical_mm.schema import LOS_CAP_HOURS

_DT_UTC = pl.Datetime("us", "UTC")

_PRE_ADMIT_FLOOR_HOURS: float = -24.0 * 30.0

def align_treatments_to_cohort(timed: pl.LazyFrame, cohort: pl.LazyFrame) -> pl.LazyFrame:
    """Keep the in-stay (+ recent pre-admission) treatment stream; emit ``TREATMENT_ALIGNED``.

    Follows the NOTES (time-varying) contract, NOT the diagnoses
    (pre-admission-only) one. Patient-bound rows (``bound_stay_id`` null) attach
    to every stay of the patient; stay-bound rows attach only to their own stay.
    For each kept row, ``delta_h = (knowable_time - intime)`` in hours is SIGNED:
    in-stay events are positive, recent pre-admission history negative. Rows are
    bounded to ``_PRE_ADMIT_FLOOR_HOURS <= delta_h <= LOS_CAP_HOURS`` so in-stay
    treatments survive (the headline fix) while ancient/far-future stamps drop.
    NO ``knowable_time <= intime`` gate is applied — that is the consumption
    layer's job (per-hour ``delta_h <= h``; stay-level pre-cutoff), exactly as
    the notes modality defers visibility to consumption.

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
    with_delta = joined.with_columns(
        ((pl.col("knowable_time") - pl.col("intime")).dt.total_seconds() / 3600.0).alias("delta_h"),
        (
            (pl.col("end_time").cast(_DT_UTC) - pl.col("intime")).dt.total_seconds() / 3600.0
        ).alias("end_delta_h"),
    )
    kept = with_delta.filter(
        (pl.col("delta_h") >= _PRE_ADMIT_FLOOR_HOURS)
        & (pl.col("delta_h") <= float(LOS_CAP_HOURS))
    )
    with_delta = kept.with_columns(
        pl.col("knowable_time").min().over(["stay_id", "source_admission_id"]).alias("__adm_kt"),
    )
    prior_visit_idx = (
        (pl.col("__adm_kt").rank("dense").over("stay_id") - 1)
        .cast(pl.Int32)
        .alias("prior_visit_idx")
    )
    with_idx = with_delta.with_columns(
        prior_visit_idx,
        pl.col("dose").cast(pl.Float32),
        pl.col("dose_unit").cast(pl.Utf8),
    )
    return with_idx.select(list(TREATMENT_ALIGNED.keys()))

def build_treatments_aligned_view(
    timed: pl.LazyFrame, stays: pl.LazyFrame, split: pl.LazyFrame, dataset: str
) -> pl.LazyFrame:
    """Aligned per-task treatment view: leakage invariant + locked split join.

    Mirrors ``modalities.build.build_aligned_view``. ``stays`` needs
    ``patient_id`` / ``stay_id`` / ``admit_time``; ``split`` needs ``stay_id`` /
    ``split``. ``bound_stay_id`` is cast to Utf8 first so an all-None timed frame
    (Null dtype) still joins.
    """
    timed = timed.with_columns(pl.col("bound_stay_id").cast(pl.Utf8))
    cohort = stays.select("patient_id", "stay_id", pl.col("admit_time").alias("intime"))
    aligned = align_treatments_to_cohort(timed, cohort)
    split_n = split.with_columns(normalize_stay_id(pl.col("stay_id"), dataset).alias("stay_id"))
    aligned_n = aligned.with_columns(normalize_stay_id(pl.col("stay_id"), dataset).alias("stay_id"))
    return aligned_n.join(split_n.select("stay_id", "split"), on="stay_id", how="inner")

def treatment_coverage_row(
    task: str, dataset: str, view: pl.DataFrame, n_cohort_stays: int
) -> dict[str, object]:
    """One coverage-report row for a materialised (task, dataset) treatment view.

    Reports the fraction of the task cohort carrying any leakage-safe treatment
    interval, the mean intervals per covered stay, and the per-origin and
    per-treatment-concept breakdowns. Mirrors ``modalities.build.coverage_row``
    and ``notes_build.notes_coverage_row``.
    """
    n_stays = view["stay_id"].n_unique()
    if view.height:
        by_origin = view.group_by("origin").len()
        origin_counts = {
            str(o): int(c)
            for o, c in zip(by_origin["origin"].to_list(), by_origin["len"].to_list(), strict=True)
        }
        by_treatment = view.group_by("treatment").len()
        treatment_counts = {
            str(t): int(c)
            for t, c in zip(
                by_treatment["treatment"].to_list(), by_treatment["len"].to_list(), strict=True
            )
        }
    else:
        origin_counts = {}
        treatment_counts = {}
    return {
        "task": task,
        "dataset": dataset,
        "n_cohort_stays": n_cohort_stays,
        "n_stays_with_treatment": n_stays,
        "frac_stays_with_treatment": (n_stays / n_cohort_stays) if n_cohort_stays else 0.0,
        "n_treatment_rows": view.height,
        "mean_treatments_per_stay": (view.height / n_stays) if n_stays else 0.0,
        "origins": sorted(origin_counts),
        "origin_counts": origin_counts,
        "treatments": sorted(treatment_counts),
        "treatment_counts": treatment_counts,
    }
