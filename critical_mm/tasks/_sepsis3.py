"""Sepsis-3 logic — abx_cont, susp_inf_alt, sep3_alt port.

This module composes 's per-hour SOFA + 's microbio table into the
full SEP-3 label. Used by `Sepsis.build_labels` for datasets that satisfy
`Sepsis.supports_sep3_arm(dataset)` (mimic_iv, eicu).

References:
  - Singer M, Deutschman CS, Seymour CW, et al. The Third International
    Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).
    JAMA. 2016;315(8):801-810.
  - ricu callback-sep3.R (commit caee690).
  - Spec: §2 + §5 Phase 3.

Audit round 10z (2026-05-20): ported ricu's two-window asymmetric
susp_inf scheme (abx_win=24h, samp_win=72h, si_mode='and'). Pre-fix
this module used a single symmetric ±48h window; the simpler scheme
under-detected 48-72h pre-abx samp pairs (dominant cause of miiv's
residual recall gap to the ricu reference).

Implementation of `abx_cont` (Task 2), `susp_inf_alt` (Task 3), and
`sep3_alt` (Task 4); integration via `Sepsis.build_labels` (Task 5).
"""

from __future__ import annotations

import polars as pl

_ABX_WIN_HOURS = 72
_ABX_MAX_GAP_HOURS = 24
_SI_ABX_AFTER_HOURS = 24
_SI_SAMP_BEFORE_HOURS = 72
_SOFA_BASELINE_HOURS = 48
_SOFA_LOOKAHEAD_HOURS = 24
_SOFA_THRESHOLD = 2

_ABX_WIN_US = _ABX_WIN_HOURS * 3600 * 1_000_000
_ABX_MAX_GAP_US = _ABX_MAX_GAP_HOURS * 3600 * 1_000_000
_SI_ABX_AFTER_US = _SI_ABX_AFTER_HOURS * 3600 * 1_000_000
_SI_SAMP_BEFORE_US = _SI_SAMP_BEFORE_HOURS * 3600 * 1_000_000

_ABX_CONT_SCHEMA: dict[str, pl.DataType] = {
    "stay_id": pl.Utf8(),
    "episode_start_time": pl.Datetime("us", "UTC"),
}
_SUSP_INF_SCHEMA: dict[str, pl.DataType] = {
    "stay_id": pl.Utf8(),
    "susp_inf_time": pl.Datetime("us", "UTC"),
}
_SEP3_SCHEMA: dict[str, pl.DataType] = {
    "stay_id": pl.Utf8(),
    "onset_time": pl.Datetime("us", "UTC"),
}

def abx_cont_ricu(
    abx_duration_df: pl.DataFrame,
    base_cohort: pl.DataFrame,
) -> pl.DataFrame:
    """ricu-faithful ``abx_cont`` callback port (audit round 10m, Phase B).

    Verbatim port of
    ``external/YAIB-cohorts/ricu-extensions/callbacks/callback-sepsis.R::abx_cont``
    (lines 9-49). Differs from :func:`abx_cont` in three concrete ways:

    1. **Input.** Takes the ricu-faithful ``abx_duration_df`` (rows
       ``[stay_id, starttime, endtime]`` from
       ``data/interim/<ds>/abx_duration.parquet``), NOT a harmonised
       meds frame filtered by ``drug_class == "antibiotic"``. The
       per-(dataset, source-table) regex / itemid match and
       duration-callback semantics from
       ``concept-dict.json#abx_duration`` are already baked in.
    2. **Pre-collapse.** Groups by ``(stay_id, starttime)`` and takes
       ``max(endtime)`` per group — mirrors ricu's
       ``abx[, .(dur_var = max(get(adur))), by = c(aid, aind)]``.
       Real-data inputevents.csv.gz (miiv) frequently has same-stay
       same-starttime duplicates that pre-collapse merges.
    3. **Post-death drop.** Drops abx rows whose ``starttime > death_time``
       BEFORE the cross-product. added a window-end
       truncation; this adds the candidate-set filter ricu's
       ``abx_death[is.na(get(dind)) | get(aind) <= get(dind)]`` applies.

    The slide() formulation in ricu is mathematically equivalent to
    our cross-product + window filter + cum_max approach; we keep the
    polars expression style for performance, not because the
    semantics differ.

    Args:
        abx_duration_df: Frame of ``[stay_id, starttime, endtime]``
            from the new ``abx_duration`` canonical table. May have
            null endtime for raw-data clamps (e.g. eICU medication's
            ``stopoffset < startoffset``).
        base_cohort: Base cohort with ``mortality_in_icu`` +
            ``discharge_time`` for the death_icu truncation.

    Returns:
        ``[stay_id, episode_start_time]``. One row per qualifying
        anchor (ricu callback-sepsis.R::abx_cont preserves all
        surviving slide() rows). A stay may appear multiple times.
    """
    if abx_duration_df.height == 0:
        return pl.DataFrame(schema=_ABX_CONT_SCHEMA)

    has_mort_cols = (
        "mortality_in_icu" in base_cohort.columns and "discharge_time" in base_cohort.columns
    )
    if has_mort_cols:
        death_lookup = base_cohort.filter(pl.col("mortality_in_icu").fill_null(False)).select(
            "stay_id", pl.col("discharge_time").alias("_death_time")
        )
    else:
        death_lookup = pl.DataFrame(
            schema={
                "stay_id": base_cohort.schema.get("stay_id", pl.Utf8()),
                "_death_time": pl.Datetime("us", "UTC"),
            }
        )

    abx = (
        abx_duration_df.group_by("stay_id", "starttime")
        .agg(pl.col("endtime").max().alias("_endtime_max"))
        .with_columns(
            pl.when(
                pl.col("_endtime_max").is_null() | (pl.col("_endtime_max") < pl.col("starttime"))
            )
            .then(pl.col("starttime"))
            .otherwise(pl.col("_endtime_max"))
            .alias("_endtime_clean"),
        )
        .select("stay_id", "starttime", "_endtime_clean")
    )

    abx_with_death = abx.join(death_lookup, on="stay_id", how="left").filter(
        pl.col("_death_time").is_null() | (pl.col("starttime") <= pl.col("_death_time"))
    )
    if abx_with_death.height == 0:
        return pl.DataFrame(schema=_ABX_CONT_SCHEMA)
    abx_clean = abx_with_death.select("stay_id", "starttime", "_endtime_clean").sort(
        "stay_id", "starttime"
    )

    anchors = (
        abx_clean.with_row_index("_anchor_idx")
        .rename({"starttime": "anchor_start"})
        .select("_anchor_idx", "stay_id", "anchor_start")
    )
    candidates = abx_clean.rename(
        {"starttime": "candidate_start", "_endtime_clean": "candidate_end"}
    )
    joined = anchors.join(candidates, on="stay_id", how="inner").filter(
        (pl.col("candidate_start") >= pl.col("anchor_start"))
        & (
            (pl.col("candidate_start") - pl.col("anchor_start")).dt.total_microseconds()
            <= _ABX_WIN_US
        )
    )

    per_anchor = (
        joined.join(death_lookup, on="stay_id", how="left")
        .sort("_anchor_idx", "candidate_start")
        .with_columns(
            pl.col("candidate_end").cum_max().over("_anchor_idx").alias("_cum_end"),
            pl.min_horizontal(
                pl.col("anchor_start").dt.offset_by(f"{_ABX_WIN_HOURS}h"),
                pl.col("_death_time"),
            ).alias("_window_end"),
        )
        .with_columns(
            pl.coalesce(
                [pl.col("candidate_start").shift(-1).over("_anchor_idx"), pl.col("_window_end")]
            ).alias("_after_time"),
        )
        .with_columns(
            (pl.col("_after_time") - pl.col("_cum_end")).dt.total_microseconds().alias("_gap_us"),
        )
        .group_by("_anchor_idx")
        .agg(
            pl.col("stay_id").first(),
            pl.col("anchor_start").first(),
            pl.col("_gap_us").max().alias("_max_gap_us"),
        )
    )
    qualifying = per_anchor.filter(pl.col("_max_gap_us") <= _ABX_MAX_GAP_US)
    if qualifying.height == 0:
        return pl.DataFrame(schema=_ABX_CONT_SCHEMA)
    return qualifying.sort("stay_id", "anchor_start").select(
        pl.col("stay_id"),
        pl.col("anchor_start").alias("episode_start_time"),
    )

def abx_cont(meds: pl.DataFrame, base_cohort: pl.DataFrame) -> pl.DataFrame:
    """Per-stay continuous antibiotic episode start time.

    A stay has a continuous abx episode iff there is some 72h sliding
    window starting at an abx admin where the maximum gap is ≤ 24h.
    Gaps are measured END-of-previous to START-of-next (audit round 10e),
    AND the trailing-edge gap from the last admin's cumulative max end to
    the window end is also bounded by the same threshold (audit round
    10f). Output = the START time of the earliest such window per stay.

    The trailing-edge gap mirrors ricu's
    `min(c(get(dind), get(aind)[1] + abx_win)) - cummax_difftime(...)`
    in callback-sepsis.R:50-52. This lets a SINGLE long-duration admin
    qualify if its duration ≥ 48h (i.e. trailing gap = 72h - 48h = 24h).
    Audit round 10f drops the prior `≥2 admins` filter — that was a CM-
    specific interpretation that excluded ricu's single-long-admin case
    and was a major driver of the eICU sepsis-3 recall gap (12% → 18%
    after the 10e end-to-start fix; remaining gap to ~50% targeted by 10f).

    Audit round 10l (2026-05-20): death_icu truncation now wired in.
    The per-anchor window is `min(anchor + 72h, death_time)` where
    `death_time = discharge_time` for stays with `mortality_in_icu==True`,
    else infinity (no truncation). Matches ricu's
    `min(death, anchor+72h)` boundary in callback-sepsis.R:50-52.

    Pre-fix, a patient who died at hour 36 within a 72h window had
    trailing gap = (anchor+72h - last_endtime) ≈ 36h+, which spuriously
    failed the ≤24h gate even when ricu's truncated `min(death,
    anchor+72h)` window passed.

    Implementation note: the per-stay cross-product is O(n²) where n is
    admins-per-stay. Test fixtures use n ≤ 4 so this is trivial; heavy
    real-data runs (n in the hundreds for sustained-abx ICU stays) are
    a candidate for the audit pass — replace with a streaming
    sliding-window if profiling flags this hot.

    Returns [stay_id, episode_start_time]. One row per qualifying
    anchor (a stay may appear multiple times — see ricu-faithful
    `abx_cont_ricu` for the rationale).
    Stays with no qualifying episode are absent.
    """
    if meds.height == 0 or "drug_class" not in meds.columns:
        return pl.DataFrame(schema=_ABX_CONT_SCHEMA)

    has_mort_cols = (
        "mortality_in_icu" in base_cohort.columns and "discharge_time" in base_cohort.columns
    )
    if has_mort_cols:
        death_lookup = base_cohort.filter(pl.col("mortality_in_icu").fill_null(False)).select(
            "stay_id", pl.col("discharge_time").alias("_death_time")
        )
    else:
        death_lookup = pl.DataFrame(
            schema={
                "stay_id": base_cohort.schema.get("stay_id", pl.Utf8()),
                "_death_time": pl.Datetime("us", "UTC"),
            }
        )

    abx = (
        meds.filter(pl.col("drug_class") == "antibiotic")
        .with_columns(
            pl.when(pl.col("endtime").is_null() | (pl.col("endtime") < pl.col("starttime")))
            .then(pl.col("starttime"))
            .otherwise(pl.col("endtime"))
            .alias("_endtime_clean"),
        )
        .select("stay_id", "starttime", "_endtime_clean")
        .sort("stay_id", "starttime")
    )
    if abx.height == 0:
        return pl.DataFrame(schema=_ABX_CONT_SCHEMA)

    anchors = (
        abx.with_row_index("_anchor_idx")
        .rename({"starttime": "anchor_start"})
        .select("_anchor_idx", "stay_id", "anchor_start")
    )
    candidates = abx.rename({"starttime": "candidate_start", "_endtime_clean": "candidate_end"})

    joined = anchors.join(candidates, on="stay_id", how="inner").filter(
        (pl.col("candidate_start") >= pl.col("anchor_start"))
        & (
            (pl.col("candidate_start") - pl.col("anchor_start")).dt.total_microseconds()
            <= _ABX_WIN_US
        )
    )

    per_anchor = (
        joined.join(death_lookup, on="stay_id", how="left")
        .sort("_anchor_idx", "candidate_start")
        .with_columns(
            pl.col("candidate_end").cum_max().over("_anchor_idx").alias("_cum_end"),
            pl.min_horizontal(
                pl.col("anchor_start").dt.offset_by(f"{_ABX_WIN_HOURS}h"),
                pl.col("_death_time"),
            ).alias("_window_end"),
        )
        .with_columns(
            pl.coalesce(
                [pl.col("candidate_start").shift(-1).over("_anchor_idx"), pl.col("_window_end")]
            ).alias("_after_time"),
        )
        .with_columns(
            (pl.col("_after_time") - pl.col("_cum_end")).dt.total_microseconds().alias("_gap_us"),
        )
        .group_by("_anchor_idx")
        .agg(
            pl.col("stay_id").first(),
            pl.col("anchor_start").first(),
            pl.col("_gap_us").max().alias("_max_gap_us"),
            pl.len().alias("_n_in_window"),
        )
    )

    qualifying = per_anchor.filter(pl.col("_max_gap_us") <= _ABX_MAX_GAP_US)
    if qualifying.height == 0:
        return pl.DataFrame(schema=_ABX_CONT_SCHEMA)

    return qualifying.sort("stay_id", "anchor_start").select(
        pl.col("stay_id"),
        pl.col("anchor_start").alias("episode_start_time"),
    )

def susp_inf_alt(abx_cont_df: pl.DataFrame, microbio: pl.DataFrame) -> pl.DataFrame:
    """Per-stay first susp_inf time.

    A stay is positive iff its abx_cont episode_start_time and a
    microbio.charttime fall in ricu's asymmetric (abx_win, samp_win)
    pair:

        samp_time - abx_time ∈ [-samp_win, +abx_win] = [-72h, +24h]

    Equivalent forms:
      - samp may occur up to 72h BEFORE abx (culture → empiric abx)
      - samp may occur up to 24h AFTER abx (empiric abx → culture)

    SI time = `min(episode_start_time, charttime)` for the qualifying
    pair (SEP-3 convention: the earlier of the two events; Singer 2016).

    When a stay has multiple qualifying (abx, samp) pairs, the SI
    time is the earliest min(abx, samp) across all qualifying pairs.

    Audit round 10z (2026-05-20): replaced the symmetric ±48h window
    with ricu's two-window asymmetric scheme. Pre-fix we missed
    48-72h pre-abx samp pairs (dominant under-detection on miiv;
    miiv recall 59 % vs ricu) and over-called 24-48h post-abx samps.

    Returns [stay_id, susp_inf_time]. One row per qualifying stay.
    """
    if abx_cont_df.height == 0 or microbio.height == 0:
        return pl.DataFrame(schema=_SUSP_INF_SCHEMA)
    samp = microbio.select("stay_id", "charttime")
    joined = abx_cont_df.join(samp, on="stay_id", how="inner")
    if joined.height == 0:
        return pl.DataFrame(schema=_SUSP_INF_SCHEMA)
    delta_us = (pl.col("charttime") - pl.col("episode_start_time")).dt.total_microseconds()
    qualifying = joined.with_columns(delta_us.alias("_delta_us")).filter(
        (pl.col("_delta_us") >= -_SI_SAMP_BEFORE_US) & (pl.col("_delta_us") <= _SI_ABX_AFTER_US)
    )
    if qualifying.height == 0:
        return pl.DataFrame(schema=_SUSP_INF_SCHEMA)
    qualifying = qualifying.with_columns(
        pl.min_horizontal(pl.col("episode_start_time"), pl.col("charttime")).alias("_si_time")
    )
    return qualifying.group_by("stay_id").agg(pl.col("_si_time").min().alias("susp_inf_time"))

def sep3_alt(
    sofa_per_hour: pl.DataFrame,
    susp_inf_alt_df: pl.DataFrame,
    base_cohort: pl.DataFrame,
) -> pl.DataFrame:
    """Per-stay SEP-3 onset.

    For each stay with a susp_inf time t:
      baseline = min(sofa) over hours in [t-48h, t]
      peak = max(sofa) over hours in [t-48h, t+24h]
      positive iff (peak - baseline) >= 2

    Onset time = susp_inf_time (= the SI time per ricu sep3.R:174-188).
    `base_cohort` provides admit_time so we can map susp_inf_time
    (Datetime) → si_hour (Int32 relative to admit_time + h*1h), matching
    the hour axis of `sofa_per_hour` produced by compute_sofa_per_hour.

    Audit round 10e (2026-05-19): switched from
    `max(peak window) - min(baseline window)` to ricu's element-wise
    `delta_cummin` (sep3.R:142-149). At each hour t in the window, the
    baseline is the running min over [si-48h, t] and delta_t is
    SOFA[t] - baseline_t. Positive iff max(delta_t over window) >= 2.
    The prior global-min/max form spuriously flagged stays where SOFA was
    high BEFORE susp_inf and then recovered — those collapse to negative
    under the cummin formulation because SOFA never RISES above its
    running min (it only fell).

    Returns [stay_id, onset_time]. One row per positive stay.
    """
    if susp_inf_alt_df.height == 0 or sofa_per_hour.height == 0:
        return pl.DataFrame(schema=_SEP3_SCHEMA)
    si_with_hour = (
        susp_inf_alt_df.join(base_cohort.select("stay_id", "admit_time"), on="stay_id", how="inner")
        .with_columns(
            ((pl.col("susp_inf_time") - pl.col("admit_time")).dt.total_seconds() / 3600.0)
            .floor()
            .cast(pl.Int32)
            .alias("si_hour"),
        )
        .select("stay_id", "susp_inf_time", "si_hour")
    )
    joined = (
        si_with_hour.join(sofa_per_hour, on="stay_id", how="inner")
        .filter(
            (pl.col("hour") >= pl.col("si_hour") - _SOFA_BASELINE_HOURS)
            & (pl.col("hour") <= pl.col("si_hour") + _SOFA_LOOKAHEAD_HOURS)
        )
        .sort("stay_id", "susp_inf_time", "hour")
    )
    windowed = joined.with_columns(
        pl.col("sofa").cum_min().over("stay_id", "susp_inf_time").alias("_cummin"),
    ).with_columns(
        (pl.col("sofa") - pl.col("_cummin")).alias("_delta"),
    )
    per_stay_max_delta = windowed.group_by("stay_id", "susp_inf_time", "si_hour").agg(
        pl.col("_delta").max().alias("_max_delta"),
    )
    positive = per_stay_max_delta.filter(pl.col("_max_delta") >= _SOFA_THRESHOLD)
    return (
        positive.group_by("stay_id")
        .agg(pl.col("susp_inf_time").min().alias("onset_time"))
        .select("stay_id", "onset_time")
    )
