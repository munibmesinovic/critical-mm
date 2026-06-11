# Extending CRITICAL-MM — Tasks

A "task" in CRITICAL-MM is a prediction target with a defined cohort and label-generation procedure. The 5 built-in tasks live under `critical_mm/tasks/`. This document covers what you need to implement to add one.

## Contract

Every task subclasses `critical_mm.api.Task` and declares four ClassVars + implements one method.

```python
from critical_mm.api import Task, register_task
from typing import ClassVar, Literal
import polars as pl

@register_task
class Mortality48(Task):
    task_name: ClassVar[str] = "mortality48"
    task_type: ClassVar[Literal["classification", "regression"]] = "classification"
    prediction_horizon_hours: ClassVar[int] = 48

    # Optional, for regression tasks only:
    # outcome_min: ClassVar[float | None] = 0.0
    # outcome_max: ClassVar[float | None] = 168.0

    def build_labels(self, base_cohort, events_long, meds, dataset,
                     microbio=None, interventions=None, abx_duration=None):
        # Return a polars DataFrame with columns:
        # patient_id, stay_id, label_time, label_value
        # plus `hour` for per-hour tasks.
        ...
```

### Required ClassVars

| Name | Type | Purpose |
|---|---|---|
| `task_name` | `str` | Unique identifier across the registry. Used as the CLI flag value (`--tasks mortality48`), as the directory name in `data/processed/<task>/`, and as the key in CM_REFERENCE. |
| `task_type` | `Literal["classification", "regression"]` | Routes to AUROC vs MAE metrics + selects the model head's output dimension. |
| `prediction_horizon_hours` | `int` | Cohort inclusion cap. Used by `Task._dyn_max_hour_per_stay` to bound the dyn grid (overridable for per-hour tasks). |

### Optional ClassVars (regression only)

| Name | Purpose |
|---|---|
| `outcome_min` / `outcome_max` | Clinical valid range. If you ship in CM_REFERENCE, these feed the YAIB-equivalent scaled-MAE conversion (`raw_mae / outcome_max`). |

### Required method

`build_labels(self, base_cohort, events_long, meds, dataset, microbio=None, interventions=None, abx_duration=None) -> pl.DataFrame`

Inputs:

- `base_cohort` (DataFrame): one row per ICU stay with at least `[patient_id, stay_id, intime, outtime, los_hours, mortality_in_icu, age, sex, weight, height]`. This is the eligibility-filtered cohort for the (task, dataset) cell.
- `events_long` (DataFrame or LazyFrame): vital + lab events, long-format `[stay_id, charttime, concept_name, value,...]`. Pass-through accepts both — use `_to_events_lf` in `tasks/base.py` for lazy-friendly handling.
- `meds` (DataFrame): per-stay medication admins `[stay_id, starttime, drug_name, drug_class,...]`. Empty if the dataset's harmonise pipeline didn't produce a meds table.
- `dataset` (str): the dataset name (e.g. "eicu", "miiv"). Use this to branch on per-dataset cohort variants (see AKI's NWICU urine-arm-dead branch).
- `microbio`, `interventions`, `abx_duration` (DataFrame, optional): only Sepsis-3 uses them; safe to ignore for other tasks.

Output: a polars DataFrame with columns:

- `patient_id` (Int64 or Utf8 — must match `base_cohort["patient_id"]`)
- `stay_id` (Int64 or Utf8)
- `label_time` (Datetime µs UTC) — when the label was determined
- `label_value` (Int8 for classification, Float32 for regression)
- `hour` (Int32, **only for per-hour tasks**) — the hour bucket within the stay

## Worked example: built-in Mortality24

See `critical_mm/tasks/mortality24.py`. Skeleton:

```python
@register_task
class Mortality24(Task):
    task_name = "mortality24"
    task_type = "classification"
    prediction_horizon_hours = 24

    def build_labels(self, base_cohort, events_long, meds, dataset, **_):
        # excl6 + excl7 from YAIB mortality.R: drop stays with los < 30h.
        eligible = base_cohort.filter(pl.col("los_hours") >= 30)
        return eligible.select(
            "patient_id",
            "stay_id",
            pl.col("intime").alias("label_time"),
            pl.col("mortality_in_icu").cast(pl.Int8).alias("label_value"),
        )
```

That's it. The base `Task.build` orchestrator handles loading `events_long` / `meds` / `microbio` from disk, applying the cohort filter, building `sta` and `dyn` parquets, and writing all three to `data/processed/mortality24/<dataset>/`.

## Per-hour tasks

If your task emits one row per `(stay, hour)` (like AKI / Sepsis / LoS), override `_dyn_max_hour_per_stay` so the dyn grid spans the whole stay rather than just `0..prediction_horizon_hours`:

```python
def _dyn_max_hour_per_stay(self, cohort: pl.DataFrame) -> pl.DataFrame:
    return cohort.select(
        "stay_id",
        pl.col("los_hours").clip(0.0, float(LOS_CAP_HOURS)).floor.cast(pl.Int32).alias("max_hour"),
    )
```

See `critical_mm/tasks/los.py` for the simplest example.

## Capability requirements

If your task structurally requires a dataset feature (e.g. microbio cultures for a Sepsis-3 strict variant), declare it via `required_dataset_capabilities`:

```python
@register_task
class StrictSepsis3(Task):
    task_name = "sepsis3_strict"
    task_type = "classification"
    prediction_horizon_hours = 6

    def required_dataset_capabilities(self, dataset: str) -> frozenset[str]:
        # Always need microbio + abx_duration -- no surrogate fallback.
        return frozenset({"microbio", "abx_duration"})

    def build_labels(self,...):...
```

`train_one` reads the dataset's `CAPABILITIES` ClassVar (see [datasets.md](datasets.md#per-dataset-capability-flags)) and raises `ValueError` at dispatch time when your task asks for capabilities the dataset doesn't declare. The error message names the missing capabilities and the dataset's declared set:

```
task 'sepsis3_strict' requires dataset capabilities ['microbio']
that 'nwicu' does not declare.
Dataset CAPABILITIES: ['abx_duration'].
Add the capability to the dataset class or remove the requirement from the task.
```

Default `required_dataset_capabilities` returns `frozenset` — no requirement, no check. The 5 built-in tasks (Mortality24, AKI, Sepsis, LoS, KidneyFunction) keep their existing per-dataset hardcoded dispatch for backward compatibility; the capability mechanism is opt-in for contrib tasks.

## Activation checklist

1. Save the file under `critical_mm/contrib/<your_task>.py`.
2. Add `@register_task` above the class.
3. Use a `task_name` that doesn't collide with built-ins or other contrib (decorator raises `ValueError` on collision).
4. Test: `python -c "from critical_mm.registry import discover_tasks; print('<your_task>' in discover_tasks)"` → True.
5. Run: `python scripts/train.py --tasks <your_task> --datasets <ds> --models GRU --seeds 42 --dry-run`.

## See also

- `critical_mm/tasks/base.py` — full Task base class with `build` orchestrator
- `critical_mm/contrib/_examples/example_task.py` — copy-paste template
- [datasets.md](datasets.md) — how to add a new dataset (most new tasks compose with existing datasets)
