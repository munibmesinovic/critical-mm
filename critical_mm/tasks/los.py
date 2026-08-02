"""LengthOfStay — hourly regression on remaining stay length."""

from __future__ import annotations

from typing import ClassVar, Literal

import polars as pl

from critical_mm.registry import register_task
from critical_mm.tasks.base import LOS_CAP_HOURS, Task

@register_task
class LengthOfStay(Task):
    """Predict remaining length of stay at every hour from h=0 onwards.

    outcome_max = 168 (7-day cap in hours) per the documented
    correction to the YAIB-models checkpoints — NOT the 15-day default
    in YAIB's Regression.gin (that value targets creatinine, not los).

    Per-stay hour grid is [0, floor(min(los, 168))] INCLUSIVE — matches
    YAIB-cohorts (reproductions/yaib_cohorts/outputs/los/eicu/outc.parquet:
    min(time)=0ms, max rows/stay=169 for stays with los≥7d).

    Cohort restriction: NONE beyond the base cohort. pre-fix this task applied `los_hours >= 48h` citing
    "YAIB paper App C.2 + Figure 8", but the paper's App C.2 only
    applies a 48h filter to KF (kidney function), NOT LoS. Table 14
    confirms LoS task n = base cohort n (e.g. eICU LoS = 182,774 =
    eICU base). ricu's `R/los.R` confirms: comment reads "Exclusions
    1.-5. are defined in base_cohort.R" with no task-specific filters.

    Pre-fix CM LoS cohort was 58-74 % under-coverage vs paper
    (eicu 76,140 vs 182,774; hirid 8,527 vs 32,338).
    """

    task_name: ClassVar[str] = "los"
    task_type: ClassVar[Literal["classification", "regression"]] = "regression"
    outcome_min: ClassVar[float | None] = 0.0
    outcome_max: ClassVar[float | None] = 168.0
    prediction_horizon_hours: ClassVar[int] = 1

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
        del events_long, meds, dataset, microbio, interventions, abx_duration
        if base_cohort.height == 0:
            return _empty_labels()
        eligible = base_cohort.with_columns(
            pl.col("los_hours")
            .clip(0.0, float(LOS_CAP_HOURS))
            .floor()
            .cast(pl.Int32)
            .alias("max_hour")
        )
        expanded = eligible.with_columns(
            pl.int_ranges(start=pl.lit(0), end=pl.col("max_hour") + 1).alias("hour")
        ).explode("hour")
        expanded = expanded.filter(pl.col("hour").is_not_null())
        return expanded.with_columns(
            pl.col("hour").cast(pl.Int32),
            pl.col("admit_time")
            .dt.offset_by(pl.col("hour").cast(pl.Utf8) + pl.lit("h"))
            .alias("label_time"),
            (pl.col("los_hours") - pl.col("hour").cast(pl.Float32))
            .clip(0.0, float(LOS_CAP_HOURS))
            .cast(pl.Float32)
            .alias("label_value"),
        ).select("patient_id", "stay_id", "hour", "label_time", "label_value")

def _empty_labels() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "patient_id": pl.Utf8(),
            "stay_id": pl.Utf8(),
            "hour": pl.Int32(),
            "label_time": pl.Datetime("us", "UTC"),
            "label_value": pl.Float32(),
        }
    )

