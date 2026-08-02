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

    def _dyn_max_hour_per_stay(self, cohort: pl.DataFrame) -> pl.DataFrame:
        """Stop the dyn grid at hour 23 so features never reach the label window.

        The base default is ``prediction_horizon_hours`` INCLUSIVE, which for KF
        builds hours 0..24 and — because ``_build_dyn`` filters on
        ``charttime < admit + (max_hour + 1)h`` — collects events in
        ``[admit+24h, admit+25h)``. The label is the median ``crea`` over
        ``(admit+24h, admit+48h]`` (see ``build_labels``), so the two windows
        intersect, and ``crea`` is one of the 48 DYNAMIC concepts. For any stay with
        a creatinine drawn in that hour the label constituent was handed to the model
        as the final timestep, which is exactly what the DL last hidden state and the
        ML ``groupby.last()`` read.

        Measured before this fix, with hour 23 as a persistence control. Among the
        stays that HAVE a creatinine in the bucket (a conditional rate — the two
        hours have different denominators), the fraction whose value equals the label
        to 1e-4 jumps from hour 23 to hour 24:

            MIMIC-IV 21.6% -> 43.0%    eICU   7.0% -> 41.3%
            HiRID     2.0% -> 78.0%    SICdb  8.8% -> 60.0%
            Zigong    0.0% -> 76.9%    OMIX   0.0% -> 55.4%
            NWICU     3.2% -> 25.7%

        Unconditionally, 2.0%-6.9% of labelled stays have a creatinine in bucket 24
        at all, so that is the share of stays actually exposed. A label-permutation
        null gives 0.6%-4.4% and hour 22 behaves like hour 23 on every dataset, which
        is what rules out benign creatinine persistence as the explanation.
        Reproduce with ``scripts/measure_kf_window_leak.py``; see
        ``the project notes`` section B2.

        Mortality24 shares ``prediction_horizon_hours=24`` but is NOT affected and
        keeps the base default: its label is eventual in-ICU death gated by
        ``los>=30h``, so hour 24 predates any label event. That is why this override
        lives here rather than in ``Task``.
        """
        return cohort.select(
            "stay_id",
            pl.lit(self.prediction_horizon_hours - 1, dtype=pl.Int32).alias("max_hour"),
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
        del meds, dataset, microbio, interventions, abx_duration
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

