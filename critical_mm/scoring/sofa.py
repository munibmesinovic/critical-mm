"""SOFA — Sequential Organ Failure Assessment (Vincent et al. 1996).

Per-hour 6-organ score (range 0–24) computed per (stay_id, hour) row,
ported from ricu's `callback-sofa.R` (commit caee690). The intended
consumer is the `sep3_alt` composition; this module is otherwise
independent of CRITICAL-MM tasks.

Aggregation flow (mirrors ricu sofa_score):
1. Bucket each input concept to a per-(stay, hour) grid via carry-forward
   join (most-recent observation at or before end-of-hour).
2. Derive composite inputs:
     pafi = po2 / fio2 (after both are carried forward)
     vent = boolean: any mech_vent interventions row overlaps the hour
     vaso = boolean: any drug_class=='vasopressor' meds row overlaps
     urine24 = sum of urine outputs over a 24h trailing window
3. Compute per-organ raw scores (Int32, 0..4). Missing inputs yield 0,
   matching ricu's `fifelse(..., else 0L)` cascade (sofa.R:135 onwards).
4. Apply ricu's 24h rolling-max within each stay (sofa.R:97-99).
5. Sum the six components horizontally → total SOFA (Int32, 0..24).

Thresholds verbatim from Vincent 1996 / ricu sofa.R:
  Respiration PaO2/FiO2 < {400, 300, 200, 100}, with cap at 200 when no vent
  Coagulation platelets < {150, 100, 50, 20}
  Liver bilirubin in [1.2, 2.0, 6.0, 12.0]
  CNS GCS in {<6, <10, <13, <15} (i.e. score 0 only for GCS == 15)
  Renal max of (crea cascade, urine24 cascade); urine24 path scores 3 or 4 only

v1 simplification (DEVIATION, see report):
  Cardiovascular drops ricu's dose-dependent score-2 and score-4 tiers.
  v1 cardio = 3 if any vasopressor active this hour, else 1 if map<70, else 0.
  Full dose-decomposed cardio is deferred to v2 (HiRID/eICU/NWICU meds schemas
  need per-drug dose parsing first).

Renal urine24 false-positive guard: `_hourly_urine24` uses
`rolling_sum(min_samples=1)` over a per-hour grid where empty hours
are NULL (not 0). When all 24 hours in a trailing window are NULL,
rolling_sum returns NULL. Without this, ricu's
`is_true(uri < 200)` cascade would auto-trigger stage 4 on every
HiRID and NWICU stay (no urine table) and on MIMIC-IV/eICU stays
with very sparse urine. The guard is documented in the report.
"""

from __future__ import annotations

import warnings

import polars as pl

from critical_mm.schema import LOS_CAP_HOURS, Schema

_RESP_PAFI_THRESHOLDS = (100.0, 200.0, 300.0, 400.0)
_COAG_PLT_THRESHOLDS = (20.0, 50.0, 100.0, 150.0)
_LIVER_BILI_THRESHOLDS = (1.2, 2.0, 6.0, 12.0)
_CARDIO_MAP_THRESHOLD = 70.0
_CNS_GCS_THRESHOLDS = (6.0, 10.0, 13.0, 15.0)
_RENAL_CREA_THRESHOLDS = (1.2, 2.0, 3.5, 5.0)
_RENAL_URINE_THRESHOLDS_24H = (200.0, 500.0)
_SOFA_WINDOW_HOURS = 24

_SOFA_OUTPUT_SCHEMA: Schema = {
    "patient_id": pl.Utf8(),
    "stay_id": pl.Utf8(),
    "hour": pl.Int32(),
    "sofa": pl.Int32(),
    "sofa_resp": pl.Int32(),
    "sofa_coag": pl.Int32(),
    "sofa_liver": pl.Int32(),
    "sofa_cardio": pl.Int32(),
    "sofa_cns": pl.Int32(),
    "sofa_renal": pl.Int32(),
}


def _per_hour_grid(base_cohort: pl.DataFrame, max_hour: int) -> pl.DataFrame:
    """Per-(stay, hour) grid covering [0, floor(min(los, max_hour))] inclusive.

    AUDIT (2026-05-16): stays with NULL ``los_hours`` are dropped here.
    Without the filter, ``pl.int_ranges(0, NULL)`` produces a NULL list →
    explode yields one row with ``hour = NULL``, which silently propagates
    through all six scorers and the rolling-max, emitting a single
    schema-valid-but-semantically-malformed output row per such stay.
    ``build_base_cohort`` already filters NULL los_hours, so this guard is
    defence-in-depth for callers passing a raw cohort frame.
    """
    upper = (
        pl.col("los_hours").cast(pl.Float64).clip(0.0, float(max_hour)).floor().cast(pl.Int32) + 1
    )
    return (
        base_cohort.filter(pl.col("los_hours").is_not_null())
        .select("patient_id", "stay_id", "admit_time", "los_hours")
        .with_columns(pl.int_ranges(0, upper).alias("hour"))
        .explode("hour")
        .with_columns(pl.col("hour").cast(pl.Int32))
    )


def _to_events_lf(events_long: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Adapter: accept DataFrame OR LazyFrame for events_long.

    Audit round 10c (2026-05-19, B1 phase B): production passes a LazyFrame
    from Task.build so the streaming engine pipelines through each
    per-concept filter without materializing all 50M eICU rows. Tests pass
    small DataFrame fixtures; `.lazy()` is free on those.
    """
    return events_long.lazy() if isinstance(events_long, pl.DataFrame) else events_long


def _hourly_carry(
    events_long: pl.DataFrame | pl.LazyFrame,
    concept: str,
    grid: pl.DataFrame,
) -> pl.DataFrame:
    """Carry-forward most-recent observation of `concept` per (stay, hour).

    Returns [patient_id, stay_id, hour, value Float64]. value is null for
    hours preceding the first observation in that stay. Implements
    ricu's `fill_gaps()` semantic via Polars join_asof with
    strategy='backward' and by='stay_id'.

    Duplicate-charttime tiebreaker: when two observations share the same
    `charttime` (real-data common, e.g. two labs reporting at the same
    minute), Polars `join_asof(strategy='backward')` picks the LAST row
    in `(stay_id, charttime)` sort order. With a stable sort the
    tiebreaker is the input row order — readers should pre-dedupe if
    determinism across reader-version bumps is required.

    Audit round 10c: `events_long` accepts a LazyFrame; the per-concept
    filter + select streams via `_stream_collect` so only the small
    (concept-filtered) frame materializes for the join_asof.
    """
    events_lf = _to_events_lf(events_long)
    obs = _stream_collect(
        events_lf.filter((pl.col("concept") == concept) & pl.col("value").is_not_null()).select(
            "stay_id", "charttime", pl.col("value").cast(pl.Float64)
        )
    ).sort("stay_id", "charttime")
    enriched = grid.with_columns(
        (
            pl.col("admit_time")
            + pl.duration(hours=pl.col("hour") + 1)
            - pl.duration(microseconds=1)
        ).alias("hour_end"),
    ).sort("stay_id", "hour_end")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        joined = enriched.join_asof(
            obs,
            left_on="hour_end",
            right_on="charttime",
            by="stay_id",
            strategy="backward",
        )
    return joined.select("patient_id", "stay_id", "hour", "value")


def _stream_collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Streaming-engine collect with eager fallback. Mirrors cohorts.base."""
    try:
        return lf.collect(engine="streaming")
    except (TypeError, ValueError):
        return lf.collect()


def _interval_overlaps_hour() -> pl.Expr:
    """Boolean: a (`starttime`, `endtime`) interval overlaps a one-hour bin
    delimited by (`hour_start`, `hour_end`).

    Used by `_hourly_vent_flag` (mech_vent intervals) and
    `_hourly_vasopressor_flag` (vasopressor drug intervals). NULL endtime
    means "still active" — overlap reduces to `starttime < hour_end`.
    Both callers must have `starttime`, `endtime`, `hour_start`, `hour_end`
    in scope. deferred cleanup of duplicated overlap logic.
    """
    return (pl.col("starttime") < pl.col("hour_end")) & (
        pl.col("endtime").is_null() | (pl.col("endtime") > pl.col("hour_start"))
    )


def _consolidate_null_endtimes(intervals: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Collapse rows whose endtime is NULL to one row per group at min(starttime).

    Audit round 10aj (2026-05-20): NULL endtime is semantically "active from
    starttime onward forever" in `_interval_overlaps_hour`. Two NULL-endtime
    rows for the same group are therefore redundant — the earlier subsumes
    the later. Keeping only the earliest per group is exact w.r.t. the
    overlap check, but cuts row count from O(n_obs_per_stay) to O(1) per
    group.

    Motivation: hirid emits each mechanical-ventilation observation as a
    point-in-time row with NULL endtime (mean 391/stay, max 15,643/stay).
    Without consolidation, `enriched.join(vents, on="stay_id")` materialises
    a per-stay cartesian (LOS_hours × n_obs_per_stay) before the overlap
    filter — for hirid sepsis that intermediate exceeded 1.3B rows and
    OOM-killed at 100 GiB. With consolidation per (stay_id) the same join
    produces ≤ LOS_hours rows per stay.

    Non-NULL-endtime rows are returned as-is — they describe bounded
    intervals that may carry meaningful diversity (different
    starttime+endtime combinations for the same group).

    Args:
        intervals: rows containing at least ``starttime`` (Datetime[µs, UTC])
            and ``endtime`` (Datetime[µs, UTC], nullable). Additional
            columns are preserved.
        group_cols: column names whose combined value defines a "group" for
            consolidation. Typical: ``["stay_id"]`` for vent, or
            ``["stay_id", "drug"]`` for per-drug vasopressors.

    Returns:
        Frame with the same schema as ``intervals``. NULL-endtime rows are
        collapsed to one row per group at the minimum starttime; non-NULL-
        endtime rows are passed through unchanged.
    """
    if intervals.height == 0:
        return intervals
    null_mask = pl.col("endtime").is_null()
    null_rows = intervals.filter(null_mask)
    if null_rows.height == 0:
        return intervals
    real_rows = intervals.filter(~null_mask)
    other_cols = [c for c in intervals.columns if c not in (*group_cols, "starttime", "endtime")]
    null_consolidated = null_rows.group_by(group_cols).agg(
        pl.col("starttime").min().alias("starttime"),
        pl.col("endtime").first().alias("endtime"),
        *[pl.col(c).first().alias(c) for c in other_cols],
    )
    null_consolidated = null_consolidated.select(intervals.columns)
    if real_rows.height == 0:
        return null_consolidated
    return pl.concat([null_consolidated, real_rows], how="vertical")


def _hourly_pafi(events_long: pl.DataFrame | pl.LazyFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """Per-hour PaO2/FiO2 ratio. Both po2 and fio2 carried forward.

    Audit round 10aa (2026-05-20): FiO2 is stored as PERCENTAGE
    (canonical unit `%`, range 21-100 per `configs/concepts_loinc.csv`),
    but the SOFA respiratory thresholds (<100 / <200 / <300 / <400) are
    defined for PaO2 [mmHg] / FiO2 [fraction 0.0-1.0]. Pre-fix we divided
    by the raw percentage, yielding ratios 100× too small (e.g. healthy
    PaO2=96, FiO2=40 % → 96/40 = 2.4 instead of 96/0.40 = 240). With
    `_sofa_resp`'s `<200 cap when not vented`, this drove non-vented
    SOFA-resp scores to 2 for everyone, badly over-counting the
    baseline; vented patients pegged at score 4. Divide by 100 first.
    """
    po2 = _hourly_carry(events_long, "po2", grid).rename({"value": "_po2"})
    fio2 = _hourly_carry(events_long, "fio2", grid).rename({"value": "_fio2"})
    joined = po2.join(fio2, on=["patient_id", "stay_id", "hour"], how="inner")
    return joined.with_columns(
        pl.when(pl.col("_fio2") > 0)
        .then(pl.col("_po2") / (pl.col("_fio2") / 100.0))
        .otherwise(None)
        .alias("pafi")
    ).select("patient_id", "stay_id", "hour", "pafi")


def _hourly_vent_flag(interventions: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """True iff a mech_vent interval overlaps [admit + h*1h, admit + (h+1)*1h)."""
    vents = interventions.filter(pl.col("intervention") == "mech_vent").select(
        "stay_id", "starttime", "endtime"
    )
    vents = _consolidate_null_endtimes(vents, ["stay_id"])
    enriched = grid.with_columns(
        (pl.col("admit_time") + pl.duration(hours=pl.col("hour"))).alias("hour_start"),
        (pl.col("admit_time") + pl.duration(hours=pl.col("hour") + 1)).alias("hour_end"),
    )
    joined = enriched.join(vents, on="stay_id", how="left")
    flag = (
        joined.with_columns(_interval_overlaps_hour().fill_null(False).alias("_overlap"))
        .group_by("patient_id", "stay_id", "hour")
        .agg(pl.col("_overlap").any().alias("vent"))
    )
    return flag.select("patient_id", "stay_id", "hour", "vent")


def _hourly_vasopressor_flag(meds: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """Per-hour flags for SOFA cardio: vaso_strong + vaso_dobu.

    Audit round 10h (2026-05-19): split into two boolean cols so _sofa_cardio
    can distinguish score-3 (any non-dobutamine pressor) from score-2 (dobu
    alone) per Vincent 1996 / ricu sofa.R:218. Pre-split everything was
    "vaso=true→score 3", which collapsed ricu's score-2 tier.

    Returns [patient_id, stay_id, hour, vaso_strong, vaso_dobu]:
        vaso_strong = any drug_class=='vasopressor' AND drug name NOT
                      containing 'dobutamine' overlaps the hour.
        vaso_dobu = any drug name containing 'dobutamine' overlaps.
    """
    vaso_meds = meds.filter(pl.col("drug_class") == "vasopressor").select(
        "stay_id", "starttime", "endtime", "drug"
    )
    vaso_meds = vaso_meds.with_columns(
        pl.col("drug")
        .str.to_lowercase()
        .str.contains("dobutamine", literal=True)
        .fill_null(False)
        .alias("_is_dobu")
    )
    vaso_meds = _consolidate_null_endtimes(vaso_meds, ["stay_id", "_is_dobu"])
    enriched = grid.with_columns(
        (pl.col("admit_time") + pl.duration(hours=pl.col("hour"))).alias("hour_start"),
        (pl.col("admit_time") + pl.duration(hours=pl.col("hour") + 1)).alias("hour_end"),
    )
    joined = enriched.join(vaso_meds, on="stay_id", how="left")
    flag = (
        joined.with_columns(
            _interval_overlaps_hour().fill_null(False).alias("_overlap"),
            pl.col("_is_dobu").fill_null(False).alias("_is_dobu"),
        )
        .with_columns(
            (pl.col("_overlap") & ~pl.col("_is_dobu")).alias("_strong"),
            (pl.col("_overlap") & pl.col("_is_dobu")).alias("_dobu"),
        )
        .group_by("patient_id", "stay_id", "hour")
        .agg(
            pl.col("_strong").any().alias("vaso_strong"),
            pl.col("_dobu").any().alias("vaso_dobu"),
        )
    )
    return flag.select("patient_id", "stay_id", "hour", "vaso_strong", "vaso_dobu")


def _hourly_urine24(events_long: pl.DataFrame | pl.LazyFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """24h rolling sum of urine output ending at hour t.

    Returns [patient_id, stay_id, hour, urine24 Float64]. Semantics:
        urine24[t] = sum of urine in the 24-row trailing window [t-23, t]
                     of the per-stay-hour grid.
        urine24[t] = NULL if no urine measurements fall in that window.

    The NULL-when-empty-window behavior matches ricu's
    `is_true(NA < 200) = FALSE` cascade: without it, the renal cascade
    would auto-trigger stage 4 on stays with no urine data (HiRID and
    NWICU wholesale, MIMIC-IV/eICU stays with very sparse urine). With
    NULL semantics, the cascade falls to the crea path or `else 0L`.

    Implementation: bucket events to (stay, hour) summing within bucket,
    DO NOT fill_null afterwards, then `rolling_sum(min_samples=1)` over
    the per-stay-hour grid. The rolling sum returns NULL whenever all
    24 hours in the window are NULL.
    """
    events_lf = _to_events_lf(events_long)
    urine = _stream_collect(
        events_lf.filter((pl.col("concept") == "urine") & pl.col("value").is_not_null()).select(
            "stay_id", "charttime", pl.col("value").cast(pl.Float64)
        )
    )
    if urine.height == 0:
        return grid.select("patient_id", "stay_id", "hour").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("urine24")
        )
    admit_lookup = grid.select("stay_id", "admit_time").unique()
    bucketed = (
        urine.join(admit_lookup, on="stay_id", how="inner")
        .with_columns(
            ((pl.col("charttime") - pl.col("admit_time")).dt.total_seconds() / 3600.0)
            .floor()
            .cast(pl.Int32)
            .alias("hour")
        )
        .group_by("stay_id", "hour")
        .agg(pl.col("value").sum().alias("_urine_h"))
    )
    enriched = grid.join(bucketed, on=["stay_id", "hour"], how="left").sort("stay_id", "hour")
    out = enriched.with_columns(
        pl.col("_urine_h")
        .rolling_sum(window_size=24, min_samples=1)
        .over("stay_id")
        .alias("urine24")
    )
    return out.select("patient_id", "stay_id", "hour", "urine24")


def _rolling_max_24h(df: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """24h rolling-max of `score_col` within each stay.

    Matches ricu sofa_score():97-99 `slide(worst_val_fun=max_or_na,
    win_length=24h, full_window=FALSE)`. Polars `rolling_max` with
    `min_samples=1` mimics `full_window=FALSE`. NULL scores are skipped
    (rolling_max ignores nulls in min_samples accounting).
    """
    return df.with_columns(
        pl.col(score_col)
        .rolling_max(window_size=_SOFA_WINDOW_HOURS, min_samples=1)
        .over("stay_id")
        .alias(score_col)
    )


def _sofa_coag(plt_df: pl.DataFrame) -> pl.DataFrame:
    """Coagulation: 4 - findInterval(plt, [20, 50, 100, 150]).

    Ricu sofa.R:182-184. Missing plt yields 0 (matches ricu's
    `fifelse(... , 0L)` fallthrough — no measurement is treated as
    "no organ failure detected", same as a healthy plt).
    """
    v = pl.col("value")
    score = (
        pl.when(v < 20)
        .then(4)
        .when(v < 50)
        .then(3)
        .when(v < 100)
        .then(2)
        .when(v < 150)
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
    )
    return plt_df.select("stay_id", "hour", score.alias("sofa_coag"))


def _sofa_liver(bili_df: pl.DataFrame) -> pl.DataFrame:
    """Liver: findInterval(bili, [1.2, 2.0, 6.0, 12.0]). Ricu sofa.R:188-190.

    Missing bili → 0 (ricu `is_true(NA<x)` cascade falls to `else 0L`).
    """
    v = pl.col("value")
    score = (
        pl.when(v.is_null())
        .then(0)
        .when(v < 1.2)
        .then(0)
        .when(v < 2.0)
        .then(1)
        .when(v < 6.0)
        .then(2)
        .when(v < 12.0)
        .then(3)
        .otherwise(4)
        .cast(pl.Int32)
    )
    return bili_df.select("stay_id", "hour", score.alias("sofa_liver"))


def _sofa_cns(gcs_df: pl.DataFrame) -> pl.DataFrame:
    """CNS: 4 - findInterval(gcs, [6, 10, 13, 15]). Ricu sofa.R:223-225.

    Domain note: GCS is bounded to [3, 15]. Score 0 corresponds only to
    GCS == 15 by the >= 15 branch.

    Missing gcs → 0 (ricu fallthrough).
    """
    v = pl.col("value")
    score = (
        pl.when(v.is_null())
        .then(0)
        .when(v < 6)
        .then(4)
        .when(v < 10)
        .then(3)
        .when(v < 13)
        .then(2)
        .when(v < 15)
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
    )
    return gcs_df.select("stay_id", "hour", score.alias("sofa_cns"))


def _sofa_resp(pafi_df: pl.DataFrame, vent_df: pl.DataFrame) -> pl.DataFrame:
    """Respiratory: PaO2/FiO2 ratio with mech-vent gate.

    Ricu sofa.R:135-164. If pafi<200 AND no mech vent, cap pafi at 200
    (lines 157-158). Then 4 if pafi<100, 3 if <200, 2 if <300, 1 if <400,
    else 0. Missing pafi → 0 (ricu fallthrough).
    """
    merged = pafi_df.join(vent_df, on=["stay_id", "hour"], how="left").with_columns(
        pl.col("vent").fill_null(False)
    )
    capped = merged.with_columns(
        pl.when(pl.col("value").lt(200) & pl.col("vent").not_())
        .then(pl.lit(200.0))
        .otherwise(pl.col("value"))
        .alias("_pafi"),
    )
    v = pl.col("_pafi")
    score = (
        pl.when(v.is_null())
        .then(0)
        .when(v < 100)
        .then(4)
        .when(v < 200)
        .then(3)
        .when(v < 300)
        .then(2)
        .when(v < 400)
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
    )
    return capped.select("stay_id", "hour", score.alias("sofa_resp"))


def _hourly_vasopressor_rates(meds: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """Per-hour max infusion rate (mcg/kg/min) per vasopressor drug.

    Audit round 10ab (2026-05-20). Returns one row per (patient, stay,
    hour) with 4 columns of max rates over any infusion interval that
    overlaps the hour:
        dopa_rate, norepi_rate, epi_rate, dobu_rate

    Rate columns are NULL when no overlapping infusion exists OR when
    the row's dose_unit is not `mcg/kg/min` (e.g. eICU's raw infusionDrug
    drugrate is mostly mcg/min or ml/hr without per-kg normalisation;
    those rows degrade to the v1.5 boolean fallback in `_sofa_cardio`).
    """
    rate_meds = meds.filter(
        (pl.col("drug_class") == "vasopressor")
        & pl.col("dose").is_not_null()
        & (pl.col("dose_unit") == "mcg/kg/min")
    ).select("stay_id", "starttime", "endtime", "drug", "dose")
    rate_meds = _consolidate_null_endtimes(rate_meds, ["stay_id", "drug", "dose"])
    is_dopa = pl.col("drug").str.to_lowercase().str.contains("dopamine", literal=True)
    is_norepi = pl.col("drug").str.to_lowercase().str.contains("norepinephrine", literal=True)
    is_epi = (
        pl.col("drug").str.to_lowercase().str.contains("epinephrine", literal=True) & ~is_norepi
    )
    is_dobu = pl.col("drug").str.to_lowercase().str.contains("dobutamine", literal=True)
    enriched = grid.with_columns(
        (pl.col("admit_time") + pl.duration(hours=pl.col("hour"))).alias("hour_start"),
        (pl.col("admit_time") + pl.duration(hours=pl.col("hour") + 1)).alias("hour_end"),
    )
    joined = enriched.join(rate_meds, on="stay_id", how="left")
    overlap = _interval_overlaps_hour().fill_null(False)
    return (
        joined.with_columns(
            overlap.alias("_overlap"),
            is_dopa.alias("_is_dopa"),
            is_norepi.alias("_is_norepi"),
            is_epi.alias("_is_epi"),
            is_dobu.alias("_is_dobu"),
        )
        .with_columns(
            pl.when(pl.col("_overlap") & pl.col("_is_dopa"))
            .then(pl.col("dose"))
            .otherwise(None)
            .alias("_dopa"),
            pl.when(pl.col("_overlap") & pl.col("_is_norepi"))
            .then(pl.col("dose"))
            .otherwise(None)
            .alias("_norepi"),
            pl.when(pl.col("_overlap") & pl.col("_is_epi"))
            .then(pl.col("dose"))
            .otherwise(None)
            .alias("_epi"),
            pl.when(pl.col("_overlap") & pl.col("_is_dobu"))
            .then(pl.col("dose"))
            .otherwise(None)
            .alias("_dobu"),
        )
        .group_by("patient_id", "stay_id", "hour")
        .agg(
            pl.col("_dopa").max().alias("dopa_rate"),
            pl.col("_norepi").max().alias("norepi_rate"),
            pl.col("_epi").max().alias("epi_rate"),
            pl.col("_dobu").max().alias("dobu_rate"),
        )
        .select(
            "patient_id", "stay_id", "hour", "dopa_rate", "norepi_rate", "epi_rate", "dobu_rate"
        )
    )


def _sofa_cardio(
    map_df: pl.DataFrame,
    vaso_df: pl.DataFrame,
    vaso_rates_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Cardiovascular — ricu-faithful dose tiers when rates available.

    Audit round 10ab (2026-05-20): per-drug µg/kg/min thresholds when
    `vaso_rates_df` carries non-null rates for the hour (ricu sofa.R:218
    + Vincent 1996):

        score = 4 if dopa>15 OR norepi>0.1 OR epi>0.1
        score = 3 if dopa>5 OR norepi<=0.1 OR epi<=0.1
        score = 2 if dopa<=5 OR any dobutamine
        score = 1 if map < 70 (no vasopressor active)
        score = 0 else

    Fallback to the v1.5 boolean tier (audit round 10h) when rates are
    not available for the hour (e.g. eICU's infusionDrug doesn't carry
    `mcg/kg/min`, NWICU has no dose data, miiv bolus boluses). The
    fallback preserves recall on datasets without rate parity.
    """
    merged = map_df.join(vaso_df, on=["stay_id", "hour"], how="left").with_columns(
        pl.col("vaso_strong").fill_null(False),
        pl.col("vaso_dobu").fill_null(False),
    )
    if vaso_rates_df is not None and vaso_rates_df.height > 0:
        merged = merged.join(
            vaso_rates_df.select(
                "stay_id", "hour", "dopa_rate", "norepi_rate", "epi_rate", "dobu_rate"
            ),
            on=["stay_id", "hour"],
            how="left",
        )
    else:
        merged = merged.with_columns(
            pl.lit(None, dtype=pl.Float32).alias("dopa_rate"),
            pl.lit(None, dtype=pl.Float32).alias("norepi_rate"),
            pl.lit(None, dtype=pl.Float32).alias("epi_rate"),
            pl.lit(None, dtype=pl.Float32).alias("dobu_rate"),
        )
    map_v = pl.col("value")
    dopa = pl.col("dopa_rate")
    norepi = pl.col("norepi_rate")
    epi = pl.col("epi_rate")
    dobu = pl.col("dobu_rate")
    score = (
        pl.when((dopa > 15) | (norepi > 0.1) | (epi > 0.1))
        .then(4)
        .when(
            (dopa > 5)
            | (norepi <= 0.1)
            | (epi <= 0.1)
            | (pl.col("vaso_strong") & dopa.is_null() & norepi.is_null() & epi.is_null())
        )
        .then(3)
        .when((dopa <= 5) | dobu.is_not_null() | pl.col("vaso_dobu"))
        .then(2)
        .when(map_v < 70)
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
    )
    return merged.select("stay_id", "hour", score.alias("sofa_cardio"))


def _sofa_renal(crea_df: pl.DataFrame, urine24_df: pl.DataFrame) -> pl.DataFrame:
    """Renal: max of (crea-derived, urine-derived) score per ricu sofa.R:229-249.

    Cascade (descending):
      4 if crea >= 5 OR urine24 < 200
      3 elif (crea in [3.5, 5)) OR urine24 < 500
      2 elif crea in [2, 3.5)
      1 elif crea in [1.2, 2)
      0 otherwise (matches ricu fallthrough; missing crea AND missing urine24 → 0)

    Note: the urine path only triggers scores 3 and 4. A renal score < 3
    can only come from crea.

    Three-valued logic note: ricu uses `is_true(crea >= 5 | uri < 200)`
    which is TRUE iff at least one side is TRUE (NA | TRUE = TRUE; NA | NA
    = NA = FALSE). Polars three-valued boolean semantics inside `pl.when`
    treat NULL as not-matching, which gives the same effect as `is_true`.
    """
    merged = crea_df.join(urine24_df, on=["stay_id", "hour"], how="left")
    crea = pl.col("value")
    uri = pl.col("urine24")
    score = (
        pl.when((crea >= 5) | (uri < 200))
        .then(4)
        .when(((crea >= 3.5) & (crea < 5)) | (uri < 500))
        .then(3)
        .when((crea >= 2) & (crea < 3.5))
        .then(2)
        .when((crea >= 1.2) & (crea < 2))
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
    )
    return merged.select("stay_id", "hour", score.alias("sofa_renal"))


def compute_sofa_per_hour(
    base_cohort: pl.DataFrame,
    events_long: pl.DataFrame | pl.LazyFrame,
    meds: pl.DataFrame,
    interventions: pl.DataFrame,
    *,
    max_hour: int = LOS_CAP_HOURS,
    keep_components: bool = True,
) -> pl.DataFrame:
    """Compute SOFA total + 6 organ-component scores at hourly grain per stay.

    Inputs:
        base_cohort: [stay_id, patient_id, admit_time, los_hours, ...]
            One row per stay (already filtered by build_base_cohort).
        events_long: long-format observations. Filtered internally to
            concepts {po2, fio2, plt, bili, map, gcs, crea, urine}.
            Concepts not present yield score 0 for that organ.
        meds: [stay_id, starttime, endtime, drug_class, ...]. Filtered to
            drug_class == 'vasopressor' for the cardio surrogate.
        interventions: [stay_id, starttime, endtime, intervention].
            Filtered to intervention == 'mech_vent' for the resp gate.
        max_hour: int, default LOS_CAP_HOURS=168. Stays are truncated
            to [0, min(floor(los_hours), max_hour)].
        keep_components: if True (default), the returned frame keeps
            the six per-organ columns. If False, only the total.

    Returns:
        pl.DataFrame with one row per (stay_id, hour). Columns:
            patient_id Utf8
            stay_id Utf8
            hour Int32 (0..min(floor(los_hours), max_hour) inclusive)
            sofa Int32 (0..24)
            sofa_resp Int32 (0..4) [only if keep_components]
            sofa_coag Int32 (0..4) [only if keep_components]
            sofa_liver Int32 (0..4) [only if keep_components]
            sofa_cardio Int32 (0..3) [only if keep_components, v1 simplified]
            sofa_cns Int32 (0..4) [only if keep_components]
            sofa_renal Int32 (0..4) [only if keep_components]

    Per ricu semantics: each component is the 24h rolling maximum of the
    per-hour raw score. Missing components contribute 0 to the total
    (ricu sofa.R fallthrough; see module docstring).

    Empty inputs (zero-row base_cohort) return an empty frame with the
    declared schema.
    """
    if base_cohort.height == 0:
        cols = (
            _SOFA_OUTPUT_SCHEMA
            if keep_components
            else {
                k: v
                for k, v in _SOFA_OUTPUT_SCHEMA.items()
                if k in ("patient_id", "stay_id", "hour", "sofa")
            }
        )
        return pl.DataFrame(schema=cols)

    grid = _per_hour_grid(base_cohort, max_hour)
    base_keys = grid.select("patient_id", "stay_id", "hour")

    plt_df = _hourly_carry(events_long, "plt", grid)
    bili_df = _hourly_carry(events_long, "bili", grid)
    crea_df = _hourly_carry(events_long, "crea", grid)
    gcs_df = _hourly_carry(events_long, "gcs", grid)
    map_df = _hourly_carry(events_long, "map", grid)

    pafi_df = _hourly_pafi(events_long, grid).rename({"pafi": "value"})
    vent_df = _hourly_vent_flag(interventions, grid)
    vaso_df = _hourly_vasopressor_flag(meds, grid)
    vaso_rates_df = _hourly_vasopressor_rates(meds, grid)
    urine24_df = _hourly_urine24(events_long, grid)

    resp = _sofa_resp(pafi_df, vent_df)
    coag = _sofa_coag(plt_df)
    liver = _sofa_liver(bili_df)
    cardio = _sofa_cardio(map_df, vaso_df, vaso_rates_df)
    cns = _sofa_cns(gcs_df)
    renal = _sofa_renal(crea_df, urine24_df)

    resp = _rolling_max_24h(resp, "sofa_resp")
    coag = _rolling_max_24h(coag, "sofa_coag")
    liver = _rolling_max_24h(liver, "sofa_liver")
    cardio = _rolling_max_24h(cardio, "sofa_cardio")
    cns = _rolling_max_24h(cns, "sofa_cns")
    renal = _rolling_max_24h(renal, "sofa_renal")

    merged = (
        base_keys.join(
            resp.select("stay_id", "hour", "sofa_resp"), on=["stay_id", "hour"], how="left"
        )
        .join(coag.select("stay_id", "hour", "sofa_coag"), on=["stay_id", "hour"], how="left")
        .join(liver.select("stay_id", "hour", "sofa_liver"), on=["stay_id", "hour"], how="left")
        .join(cardio.select("stay_id", "hour", "sofa_cardio"), on=["stay_id", "hour"], how="left")
        .join(cns.select("stay_id", "hour", "sofa_cns"), on=["stay_id", "hour"], how="left")
        .join(renal.select("stay_id", "hour", "sofa_renal"), on=["stay_id", "hour"], how="left")
    )
    component_cols = (
        "sofa_resp",
        "sofa_coag",
        "sofa_liver",
        "sofa_cardio",
        "sofa_cns",
        "sofa_renal",
    )
    total = pl.sum_horizontal(*[pl.col(c) for c in component_cols]).cast(pl.Int32).alias("sofa")
    merged = merged.with_columns(total)

    output_cols: list[str] = ["patient_id", "stay_id", "hour", "sofa"]
    if keep_components:
        output_cols += list(component_cols)
    return merged.select(output_cols)
