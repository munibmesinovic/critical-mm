"""Template: a minimal new Task.

This file is a SCAFFOLD -- it is intentionally NOT auto-registered
(``critical_mm.registry._ensure_contrib_loaded`` skips the
``_examples/`` subpackage). To activate your task:

  1. Copy this file to ``critical_mm/contrib/<your_task>.py`` (or to
     your own pip package).
  2. Rename ``_ExampleMortality48`` to something meaningful.
  3. Change ``task_name`` from the underscore-prefixed scaffold name to
     your real name (e.g. ``"mortality48"``).
  4. Uncomment the ``@register_task`` decorator.
  5. Implement ``build_labels`` to emit the outcome frame.

Once that's done, ``python scripts/train.py --tasks mortality48`` works.

The full Task contract is documented in ``docs/extending/tasks.md``.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import polars as pl

from critical_mm.api import Task

class _ExampleMortality48(Task):
    """Predict in-ICU mortality at h=48 after admission (worked example).

    This mirrors the built-in ``critical_mm.tasks.mortality24.Mortality24``
    but with a 48-hour horizon. Real implementations should:

      - Set ``task_name`` to a string unique across the registry.
      - Set ``task_type`` to ``"classification"`` or ``"regression"``.
      - Set ``prediction_horizon_hours`` to the cohort's eligibility cap.
      - For regression: set ``outcome_min`` / ``outcome_max`` to the
        clinical valid range (used by the YAIB-equivalent scaled MAE
        conversion if your task ships in CM_REFERENCE).
      - Implement ``build_labels`` to return a polars DataFrame with
        columns ``[patient_id, stay_id, label_time, label_value]``.
    """

    task_name: ClassVar[str] = "_example_mortality48"
    task_type: ClassVar[Literal["classification", "regression"]] = "classification"
    prediction_horizon_hours: ClassVar[int] = 48

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
        """Return the outc-shaped label frame for this task.

        The scaffold returns an empty frame so the file imports cleanly.
        A real implementation derives labels from ``base_cohort`` (stays,
        with intime / outtime / mortality_in_icu columns) plus optional
        signal frames (events_long for vitals/labs, meds for drugs,
        microbio for cultures, interventions for procedures, abx_duration
        for ricu-faithful antibiotic episodes).
        """
        del base_cohort, events_long, meds, dataset
        del microbio, interventions, abx_duration
        return pl.DataFrame(
            schema={
                "patient_id": pl.Int64,
                "stay_id": pl.Int64,
                "label_time": pl.Datetime("us", "UTC"),
                "label_value": pl.Int8,
            }
        )
