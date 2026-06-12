"""KidneyFunction — regression on MEDIAN creatinine in the (24h, 48h] post-admit window.

Matches YAIB-cohorts `kidney_function.R:60`:
    outc <- function_step(outc, function(df) df[, .(crea = median(crea)), by=id_var])
with window `. > hours(24) & . <= hours(48)` — i.e. half-open (24h, 48h].
"""

from __future__ import annotations

from typing import ClassVar, Literal

import polars as pl

from critical_mm.registry import register_task
from critical_mm.tasks.base import Task

@register_task
class KidneyFunction(Task):
    """Predict MEDIAN creatinine in hours (24, 48] post-admit (YAIB parity)."""

    task_name: ClassVar[str] = "kidney_function"
    task_type: ClassVar[Literal["classification", "regression"]] = "regression"
    outcome_min: ClassVar[float | None] = 0.0
    outcome_max: ClassVar[float | None] = 15.0
    prediction_horizon_hours: ClassVar[int] = 24

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
        del meds, dataset, microbio, interventions
        eligible = base_cohort.filter(pl.col("los_hours") >= 48.0)
        if eligible.height == 0:
            return _empty_labels()

        events_lf = events_long.lazy() if isinstance(events_long, pl.DataFrame) else events_long
        try:
            crea = events_lf.filter(pl.col("concept") == "crea").collect(engine="streaming")
        except (TypeError, ValueError):
            crea = events_lf.filter(pl.col("concept") == "crea").collect()
        if crea.height == 0:
            return _empty_labels()

        joined = crea.join(
            eligible.select("stay_id", "admit_time"), on="stay_id", how="inner"
        ).filter(
            (pl.col("charttime") > pl.col("admit_time").dt.offset_by("24h"))
            & (pl.col("charttime") <= pl.col("admit_time").dt.offset_by("48h"))
        )
        medians = joined.group_by("stay_id").agg(
            pl.col("value").cast(pl.Float64).median().alias("median_crea")
        )
        out = eligible.join(medians, on="stay_id", how="inner")
        label = pl.col("median_crea").clip(0.0, 15.0).cast(pl.Float32).alias("label_value")
        return out.select(
            "patient_id",
            "stay_id",
            pl.col("admit_time")
            .dt.offset_by(f"{self.prediction_horizon_hours}h")
            .alias("label_time"),
            label,
        )

def _empty_labels() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "patient_id": pl.Utf8(),
            "stay_id": pl.Utf8(),
            "label_time": pl.Datetime("us", "UTC"),
            "label_value": pl.Float32(),
        }
    )
