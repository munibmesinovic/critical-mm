"""AKI — KDIGO 2012 acute kidney injury, with a creatinine-only variant for
datasets lacking usable hourly urine (nwicu, omix).

Standard definition (used for eICU, MIMIC-IV, HiRID):
- Creatinine arm: ≥0.3 mg/dL increase OR ≥1.5× baseline.
- Urine arm: <0.5 mL/kg/h for ≥6 hours.
- Onset time = charttime of qualifying peak (crea arm) or first low-flow
  hour boundary (urine arm); the earliest across arms.

Creatinine-only variant (datasets without usable hourly urine: nwicu, omix):
- Urine arm dropped — NWICU v0.1.0 has no outputevents.csv.gz; OMIX lacks
  recorded weight and charts urine as per-void / per-shift volumes that
  bin to spurious anuric low-flow when hourly-bucketed.
- Deviation `{dataset}_aki_crea_only` recorded in the per-task deviations.csv
  for each crea-only dataset (nwicu, omix).
- `supports_urine_arm(dataset) == False`; the urine branch is never queried.

Cohort exclusions (re-verified against YAIB-cohorts/R/aki.R lines 95-125):
- excl7: AKI onset within the first 6h of ICU → stay excluded entirely.
- excl8: baseline creatinine > 4 mg/dL → stay excluded. Baseline =
  last cummin(crea) over (pre-ICU OR first-in-ICU). CM has no pre-ICU
  events in the canonical schema, so the cummin collapses to "first
  in-ICU crea" — that is the available proxy. Restoring this filter
  reverts (commit 8b228aa), which removed it citing "aki.R
  lines 50-75" but missed the actual exclusions block at lines 95-125
  (`exclude(patients, mget(paste0("excl", 6:8)))`).
- excl6: for eICU only, drop stays whose hospital_id has zero AKI cases
  (multi-hospital prevalence filter).

Per-hour outc shape (YAIB-cohorts parity, audit round 2026-05-15):
- One row per (stay, hour) for hour in 0..floor(los_hours). Cumulative
  semantics: `label_value` flips from 0 to 1 at the hour bucket
  containing onset_time, and stays 1 for the rest of the stay. Stays
  with no onset carry label_value = 0 throughout. `label_time` at hour t
  is `admit_time + t*1h`. The YAIB exporter casts label_value (Int8)
  to Boolean and emits [stay_id, time, label] — matching
  reproductions/yaib_cohorts/outputs/aki/eicu/outc.parquet exactly.

v1 simplifications (deferred to ricu-parity calibration):
- Urine "≥6 hours" is summed total hours below threshold, not a strict
  rolling-window consecutive-run check.
- HiRID urine arm: YAIB paper App D.2 notes HiRID records urine rate
  DIRECTLY (other datasets are computed as output_ml / hours_since_last).
  The current `_aki_urine_arm` algorithm sums urine values per (stay, hour)
  and divides by weight — equivalent to "rate" only if HiRID's urine
  entries are stored as hourly volumes (the assumption from `test_hirid.py:54`).
  Heavy-run verification on `data/raw/hirid` will confirm; deviation flag
  `hirid_aki_urine_assumed_hourly_volume` is emitted to surface this.

Task 16 (2026-05-16): crea arm now matches YAIB-cohorts kdigo_crea — rolling
min over 48h (delta arm) and 168h (ratio arm) windows per stay (closed='right',
i.e. (t-period, t]). Stage cascade collapses to: positive ⟺ delta ≥ 0.3 mg/dL
vs crea_48hr OR crea ≥ 1.5 × crea_168hr. The urine arm remains the v1
"total low hours ≥ 6" surrogate.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar, Literal

import polars as pl

from critical_mm.registry import register_task
from critical_mm.tasks.base import LOS_CAP_HOURS, Task, TaskBuildResult

_CREA_DELTA_THRESHOLD_MGDL: float = 0.3
_CREA_RATIO_THRESHOLD: float = 1.5
_URINE_THRESHOLD_ML_KG_HR: float = 0.5
_URINE_MIN_HOURS: int = 6
_WEIGHT_FALLBACK_KG: float = 75.0
_ONSET_GRACE_HOURS: int = 6
_BASELINE_CREA_EXCLUSION_MGDL: float = 4.0


@register_task
class AKI(Task):
    """KDIGO AKI within a 6h prediction horizon."""

    task_name: ClassVar[str] = "aki"
    task_type: ClassVar[Literal["classification", "regression"]] = "classification"
    outcome_min = None
    outcome_max = None
    prediction_horizon_hours: ClassVar[int] = 6

    def supports_urine_arm(self, dataset: str) -> bool:
        return dataset not in {"nwicu", "omix"}

    def _dyn_max_hour_per_stay(self, cohort: pl.DataFrame) -> pl.DataFrame:
        return cohort.select(
            "stay_id",
            pl.col("los_hours")
            .clip(0.0, float(LOS_CAP_HOURS))
            .floor()
            .cast(pl.Int32)
            .alias("max_hour"),
        )

    def build_labels(
        self,
        base_cohort: pl.DataFrame,
        events_long: pl.DataFrame | pl.LazyFrame,
        meds: pl.DataFrame,
        dataset: str,
        microbio: pl.DataFrame | None = None,
        interventions: pl.DataFrame | None = None,
        abx_duration: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        del meds, microbio, interventions
        if base_cohort.height == 0:
            return _empty_aki_labels()
        events_lf = events_long.lazy() if isinstance(events_long, pl.DataFrame) else events_long
        _aki_concepts = ["crea", "urine", "urine_rate"]
        try:
            events_long = events_lf.filter(pl.col("concept").is_in(_aki_concepts)).collect(
                engine="streaming"
            )
        except (TypeError, ValueError):
            events_long = events_lf.filter(pl.col("concept").is_in(_aki_concepts)).collect()
        base_cohort = _exclude_high_baseline_crea(base_cohort, events_long)
        if base_cohort.height == 0:
            return _empty_aki_labels()
        crea_onsets = _aki_creatinine_arm(base_cohort, events_long)
        if self.supports_urine_arm(dataset):
            urine_onsets = _aki_urine_arm(base_cohort, events_long, dataset=dataset)
            onsets = _earliest_per_stay(crea_onsets, urine_onsets)
        else:
            onsets = crea_onsets
        base_cohort = _filter_eicu_hospitals_without_cases(base_cohort, onsets, dataset)
        base_cohort, onsets = _exclude_early_onset_stays(base_cohort, onsets, _ONSET_GRACE_HOURS)
        return _per_hour_outc_from_onsets(base_cohort, onsets)

    def build(self, **kwargs: object) -> TaskBuildResult:
        result = super().build(**kwargs)  # type: ignore[arg-type]
        dataset = str(kwargs["dataset"])
        deviations_path = result["sta_path"].parent / "deviations.csv"
        deviations: list[tuple[str, str]] = []
        if not self.supports_urine_arm(dataset):
            crea_only_reason = {
                "nwicu": "NWICU v0.1.0 has no urine output → KDIGO urine arm dropped",
                "omix": (
                    "OMIX has no recorded weight (the 75kg ricu default would "
                    "apply to every stay) and urine is charted as per-void / "
                    "per-shift volumes (median ~400 mL) not hourly rates; binned "
                    "hourly the no-void hours read as spurious anuric low-flow, so "
                    "the KDIGO urine arm over-fires. AKI determined by the "
                    "creatinine arm alone (NWICU precedent)."
                ),
            }.get(dataset, f"{dataset}: KDIGO urine arm dropped (crea arm only)")
            deviations.append((f"{dataset}_aki_crea_only", crea_only_reason))
        if dataset == "hirid":
            deviations.append(
                (
                    "hirid_aki_urine_assumed_hourly_volume",
                    "YAIB paper App D.2 notes HiRID records urine RATE directly, "
                    "other datasets compute rate from output_ml / hours_since_last. "
                    "CRITICAL-MM's _aki_urine_arm sums urine per (stay, hour) and "
                    "divides by weight; correct iff HiRID's urine concept is stored "
                    "as hourly VOLUME (test_hirid.py:54 assumption). Heavy-run on "
                    "data/raw/hirid will confirm; if HiRID's raw schema is rate-direct "
                    "a dataset-specific branch will be needed.",
                )
            )
        if self.supports_urine_arm(dataset):
            cohort_path = Path(str(kwargs["processed_root"])) / "base_cohort" / dataset
            stays_parquet = cohort_path / "stays.parquet"
            if stays_parquet.exists():
                cohort = pl.read_parquet(stays_parquet)
                n_fallback = int(cohort.filter(pl.col("weight").is_null()).height)
                n_total = int(cohort.height)
                if n_fallback > 0:
                    deviations.append(
                        (
                            "weight_fallback_75kg_used",
                            (
                                f"{n_fallback}/{n_total} stays had weight=null; "
                                f"75kg ricu-default applied for urine arm "
                                f"(callback-kdigo.R:76)"
                            ),
                        )
                    )
        _write_deviations(deviations_path, deviations)
        return result


_ONSET_SCHEMA: dict[str, pl.DataType] = {
    "stay_id": pl.Utf8(),
    "onset_time": pl.Datetime("us", "UTC"),
}


def _exclude_high_baseline_crea(
    base_cohort: pl.DataFrame, events_long: pl.DataFrame
) -> pl.DataFrame:
    """Drop stays whose baseline crea > 4 mg/dL (ricu aki.R excl8).

    ricu definition (aki.R lines 104-121): baseline = last cummin(crea) over
    (pre-ICU OR first-in-ICU rows). CRITICAL-MM has no pre-ICU events in the
    canonical schema, so the cummin collapses to "first in-ICU crea". Stays
    with no crea measurement are kept (no exclusion fires).
    """
    if base_cohort.height == 0:
        return base_cohort
    crea = events_long.filter((pl.col("concept") == "crea") & pl.col("value").is_not_null())
    if crea.height == 0:
        return base_cohort
    baseline = (
        crea.sort("stay_id", "charttime")
        .group_by("stay_id", maintain_order=True)
        .agg(pl.col("value").cast(pl.Float64).first().alias("_baseline_crea"))
    )
    excluded = baseline.filter(pl.col("_baseline_crea") > _BASELINE_CREA_EXCLUSION_MGDL).select(
        "stay_id"
    )
    return base_cohort.join(excluded, on="stay_id", how="anti")


def _exclude_early_onset_stays(
    base_cohort: pl.DataFrame, onsets: pl.DataFrame, grace_hours: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Drop stays whose onset is within the first `grace_hours` of ICU admission.

    Returns the filtered (base_cohort, onsets) pair. Stays with no onset
    are kept (still emit as negatives in the per-hour outc).
    """
    if onsets.height == 0:
        return base_cohort, onsets
    admits = base_cohort.select("stay_id", "admit_time")
    annotated = onsets.join(admits, on="stay_id", how="inner").with_columns(
        ((pl.col("onset_time") - pl.col("admit_time")).dt.total_seconds() / 3600.0).alias(
            "_onset_hours_into_stay"
        ),
    )
    excluded_stays = annotated.filter(pl.col("_onset_hours_into_stay") < float(grace_hours)).select(
        "stay_id"
    )
    new_cohort = base_cohort.join(excluded_stays, on="stay_id", how="anti")
    new_onsets = annotated.filter(pl.col("_onset_hours_into_stay") >= float(grace_hours)).select(
        "stay_id", "onset_time"
    )
    return new_cohort, new_onsets


def _filter_eicu_hospitals_without_cases(
    base_cohort: pl.DataFrame, onsets: pl.DataFrame, dataset: str
) -> pl.DataFrame:
    """For eICU only, drop stays whose hospital_id has zero positive cases.

    Paper App C.2: "we further excluded hospitals that did not have a single
    patient with AKI or sepsis to exclude hospitals with an insufficient
    recording of features necessary to define the outcome." Mirrors the
    YAIB-cohorts `prevalence` step at sepsis.R:77-97. Only applied for
    eICU (other datasets are single-site so the filter is a no-op).
    """
    if dataset != "eicu":
        return base_cohort
    if "hospital_id" not in base_cohort.columns or base_cohort.height == 0:
        return base_cohort
    if onsets.height == 0:
        return base_cohort.head(0)
    positive_hospitals = (
        base_cohort.join(onsets.select("stay_id"), on="stay_id", how="inner")
        .filter(pl.col("hospital_id").is_not_null())
        .select("hospital_id")
        .unique()
    )
    if positive_hospitals.height == 0:
        return base_cohort.head(0)
    return base_cohort.join(positive_hospitals, on="hospital_id", how="inner")


def _aki_creatinine_arm(base_cohort: pl.DataFrame, events_long: pl.DataFrame) -> pl.DataFrame:
    """Return positive stays + onset time (charttime of first qualifying reading).

    KDIGO depth (Task 16, 2026-05-16) — YAIB-cohorts parity (callback-kdigo.R:2-39):
    - ``crea_48hr`` = rolling min of crea over (t-48h, t] per stay
    - ``crea_168hr`` = rolling min of crea over (t-168h, t] per stay
    - Stage cascade (fcase) at each timestamp:
        - stage 3 if crea >= 3 * crea_168hr
        - stage 3 if crea >= 4 AND (crea - crea_48hr >= 0.3 OR crea >= 1.5 * crea_168hr)
        - stage 2 if crea >= 2 * crea_168hr
        - stage 1 if crea - crea_48hr >= 0.3
        - stage 1 if crea >= 1.5 * crea_168hr
    The binary label is `stage >= 1`. Since 2× and 3× ratios both imply ≥ 1.5×,
    and the stage-3 ≥4mg/dL pathway gates on the same delta/ratio predicates,
    the cascade collapses to: stage >= 1 ⟺ (crea - crea_48hr >= 0.3) OR
    (crea >= 1.5 * crea_168hr). Onset = first charttime where this holds.

    Before Task 16, this function used a fixed first-24h crea-min baseline.
    That under-detects AKI in stays whose admission baseline was already
    elevated and then recovered before deteriorating again (test fixture
    `test_aki_creatinine_arm_uses_rolling_48h_baseline_not_fixed_first_24h`).
    """
    crea = events_long.filter((pl.col("concept") == "crea") & pl.col("value").is_not_null())
    if crea.height == 0:
        return pl.DataFrame(schema=_ONSET_SCHEMA)
    joined = crea.join(
        base_cohort.select("stay_id"),
        on="stay_id",
        how="inner",
    ).with_columns(pl.col("value").cast(pl.Float64))
    sorted_lf = joined.sort("stay_id", "charttime").with_columns(
        pl.col("value")
        .rolling_min_by(by="charttime", window_size="48h", closed="right")
        .over("stay_id")
        .alias("_crea_48hr"),
        pl.col("value")
        .rolling_min_by(by="charttime", window_size="168h", closed="right")
        .over("stay_id")
        .alias("_crea_168hr"),
    )
    val = pl.col("value")
    c48 = pl.col("_crea_48hr")
    c168 = pl.col("_crea_168hr")
    qualifying = sorted_lf.filter(
        ((val - c48) >= _CREA_DELTA_THRESHOLD_MGDL) | (val >= _CREA_RATIO_THRESHOLD * c168)
    )
    return (
        qualifying.group_by("stay_id")
        .agg(pl.col("charttime").min().alias("onset_time"))
        .select("stay_id", "onset_time")
    )


def _aki_urine_arm(
    base_cohort: pl.DataFrame,
    events_long: pl.DataFrame,
    dataset: str = "",
) -> pl.DataFrame:
    """Return positive stays + onset time (first qualifying urine measurement).

    Faithful port of ricu's `kdigo_urine` callback (audit round 10s,
    2026-05-20) — verbatim from
    `reproductions/yaib_cohorts_pinned/ricu-extensions/callbacks/
    callback-kdigo.R::kdigo_urine` + `urine_rate`.

    Algorithm:

    1. **`urine_rate` per measurement.** For each urine event at
       charttime t:
       - `tm = t - prev_t` (hours since previous urine event in same
         stay). For the first event per stay OR when `tm > 24h`,
         `tm = 1h`.
       - `urine_rate = urine_vol / tm` (mL/h)

    2. **Windowed rate per timestamp.** For each window
       dur ∈ {6h, 12h, 24h}: at each urine event time t, sum
       `urine_rate` over the half-open window `(t-dur, t]`, then
       divide by patient weight (75 kg fallback per
       callback-kdigo.R:76 `ifelse(is.na(weight), 75, weight)`). This
       gives an effective time-weighted average urine output in
       mL/kg/h over each window.

    3. **Stage cascade** (ricu callback-kdigo.R:108-113):
       - stage 3 if hour ≥ 24h AND urine_rate_24hr < 0.3
       - stage 3 if hour ≥ 12h AND urine_rate_12hr == 0
       - stage 2 if hour ≥ 12h AND urine_rate_12hr < 0.5
       - stage 1 if hour ≥ 6h AND urine_rate_6hr < 0.5
       - default 0

       The hour gates prevent firing on short stays — a 4h stay
       cannot have stage 1 urine AKI.

    Onset = charttime of first urine event where stage ≥ 1.

    Pre-fix ( and earlier): the urine arm used a simpler
    "hours where instantaneous urine ÷ weight < 0.5 ≥ 6" heuristic
    that bucketed urine to hour grid then counted low-flow hours
    independently. That under-fired vs ricu's sliding-window
    semantics; eICU AKI rate was 16.97 % vs ricu reference 37.79 %
    after . This faithful port targets the residual gap.
    """
    urine_rate_events = events_long.filter(
        (pl.col("concept") == "urine_rate")
        & pl.col("value").is_not_null()
        & (pl.col("value") >= 0)
        & (pl.col("value") <= 2000)
    )
    use_urine_rate_concept = urine_rate_events.height > 0
    if use_urine_rate_concept:
        urine = urine_rate_events
    else:
        urine = events_long.filter(
            (pl.col("concept") == "urine")
            & pl.col("value").is_not_null()
            & (pl.col("value") >= 0)
            & (pl.col("value") <= 5000)
        )
    if urine.height == 0:
        return pl.DataFrame(schema=_ONSET_SCHEMA)
    weights = base_cohort.select(
        "stay_id",
        pl.col("weight").cast(pl.Float64).fill_null(_WEIGHT_FALLBACK_KG).alias("weight"),
    )
    admits = base_cohort.select("stay_id", "admit_time")

    if use_urine_rate_concept:
        rate_expr = pl.col("value").cast(pl.Float64)
    else:
        rate_expr = pl.col("value").cast(pl.Float64) / pl.col("_tm_h")
    _ = dataset
    urine_rated = (
        urine.join(admits, on="stay_id", how="inner")
        .sort("stay_id", "charttime")
        .with_columns(
            (
                (
                    pl.col("charttime") - pl.col("charttime").shift(1).over("stay_id")
                ).dt.total_seconds()
                / 3600.0
            ).alias("_tm_h")
        )
        .with_columns(
            pl.when(pl.col("_tm_h").is_null() | (pl.col("_tm_h") > 24.0))
            .then(pl.lit(1.0))
            .otherwise(pl.col("_tm_h"))
            .alias("_tm_h"),
        )
        .with_columns(
            rate_expr.alias("_urate"),
            ((pl.col("charttime") - pl.col("admit_time")).dt.total_seconds() / 3600.0).alias(
                "_hour_since_admit"
            ),
        )
        .join(weights, on="stay_id", how="inner")
    )

    sorted_lf = urine_rated.sort("stay_id", "charttime").with_columns(
        (
            pl.col("_urate")
            .rolling_sum_by(by="charttime", window_size="6h", closed="right")
            .over("stay_id")
            / pl.col("weight")
        ).alias("_rate_6hr"),
        (
            pl.col("_urate")
            .rolling_sum_by(by="charttime", window_size="12h", closed="right")
            .over("stay_id")
            / pl.col("weight")
        ).alias("_rate_12hr"),
        (
            pl.col("_urate")
            .rolling_sum_by(by="charttime", window_size="24h", closed="right")
            .over("stay_id")
            / pl.col("weight")
        ).alias("_rate_24hr"),
    )

    h = pl.col("_hour_since_admit")
    r6 = pl.col("_rate_6hr")
    r12 = pl.col("_rate_12hr")
    r24 = pl.col("_rate_24hr")
    qualifying = sorted_lf.filter(
        ((h >= 24.0) & (r24 < 0.3))
        | ((h >= 12.0) & (r12 == 0.0))
        | ((h >= 12.0) & (r12 < 0.5))
        | ((h >= 6.0) & (r6 < _URINE_THRESHOLD_ML_KG_HR))
    )
    if qualifying.height == 0:
        return pl.DataFrame(schema=_ONSET_SCHEMA)
    return (
        qualifying.group_by("stay_id")
        .agg(pl.col("charttime").min().alias("onset_time"))
        .select("stay_id", "onset_time")
    )


def _earliest_per_stay(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    """Union two onset frames; keep the earliest onset_time per stay_id."""
    if a.height == 0:
        return b
    if b.height == 0:
        return a
    return (
        pl.concat([a, b], how="vertical_relaxed")
        .group_by("stay_id")
        .agg(pl.col("onset_time").min())
    )


_PREDICTION_HORIZON_HOURS: int = 6


def _per_hour_outc_from_onsets(base_cohort: pl.DataFrame, onsets: pl.DataFrame) -> pl.DataFrame:
    """Expand per-stay onsets into per-hour windowed binary labels.

    Each stay contributes ``floor(min(los_hours, LOS_CAP_HOURS)) + 1`` rows
    for hour in [0, max_hour] INCLUSIVE — matches YAIB-cohorts dyn/outc
    alignment (reproductions/yaib_cohorts/outputs/aki/eicu/outc.parquet has
    dyn_n == outc_n per stay).

    ``label_time = admit_time + hour*1h`` and (YAIB ``outcome_window(c(6,6))``):
    ``label_value = 1`` iff the stay has a non-null onset AND
    ``abs(hour - onset_hour) <= 6`` (13-hour window centred on onset).
    Stays with no onset carry label_value = 0 throughout.
    """
    if base_cohort.height == 0:
        return _empty_aki_labels()
    stay_hours = (
        base_cohort.select("patient_id", "stay_id", "admit_time", "los_hours")
        .with_columns(
            pl.int_ranges(
                0,
                pl.col("los_hours").clip(0.0, float(LOS_CAP_HOURS)).floor().cast(pl.Int32) + 1,
            ).alias("hour"),
        )
        .explode("hour")
        .with_columns(pl.col("hour").cast(pl.Int32))
    )
    with_onsets = stay_hours.join(onsets, on="stay_id", how="left").with_columns(
        (pl.col("admit_time") + pl.duration(hours=pl.col("hour"))).alias("label_time"),
        ((pl.col("onset_time") - pl.col("admit_time")).dt.total_seconds() / 3600.0)
        .floor()
        .cast(pl.Int32)
        .alias("onset_hour"),
    )
    return with_onsets.with_columns(
        (
            pl.col("onset_hour").is_not_null()
            & ((pl.col("hour") - pl.col("onset_hour")).abs() <= _PREDICTION_HORIZON_HOURS)
        )
        .cast(pl.Int8)
        .alias("label_value")
    ).select("patient_id", "stay_id", "hour", "label_time", "label_value")


def _empty_aki_labels() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "patient_id": pl.Utf8(),
            "stay_id": pl.Utf8(),
            "hour": pl.Int32(),
            "label_time": pl.Datetime("us", "UTC"),
            "label_value": pl.Int8(),
        }
    )


def _write_deviations(path: Path, deviations: list[tuple[str, str]]) -> None:
    """CSV with header + one row per applied deviation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["deviation", "reason"])
        for name, reason in deviations:
            writer.writerow([name, reason])
