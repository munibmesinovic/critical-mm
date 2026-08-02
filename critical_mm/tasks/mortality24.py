"""Mortality24 — binary classification of ICU mortality at 24h."""

from __future__ import annotations

from typing import ClassVar, Literal

import polars as pl

from critical_mm.registry import register_task
from critical_mm.tasks.base import Task

@register_task
class Mortality24(Task):
    """Predict in-ICU mortality at h=24 after admission.

    Inclusion (YAIB mortality.R excl6+excl7 parity):
    - excl6: drop stays where mortality_in_icu AND death within 30h.
    - excl7: drop stays with los_hours < 30.
    Combined, these collapse to ``los_hours >= 30`` in CRITICAL-MM's
    schema because discharge_time = death_time for in-ICU deaths, so
    los_hours < 30 catches early deaths automatically. The base cohort
    already filtered los_hours ≥ 6; here we tighten to 30.

    CAVEAT (OMIX): omix ``discharge_time`` is a reconstructed monitoring-window
    span, NOT the death time, so the "los_hours<30 catches early deaths"
    shortcut does NOT hold for omix — its label is eventual in-ICU mortality
    (``StatusOnDischarge == 'Dead'``), a documented harmonisation compromise.
    """

    task_name: ClassVar[str] = "mortality24"
    task_type: ClassVar[Literal["classification", "regression"]] = "classification"
    outcome_min = None
    outcome_max = None
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
        del events_long, meds, dataset, microbio, interventions
        eligible = base_cohort.filter(pl.col("los_hours") >= 30.0)
        return eligible.select(
            "patient_id",
            "stay_id",
            pl.col("admit_time")
            .dt.offset_by(f"{self.prediction_horizon_hours}h")
            .alias("label_time"),
            pl.col("mortality_in_icu").cast(pl.Int8).alias("label_value"),
        )

