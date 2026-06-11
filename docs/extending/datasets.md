# Extending CRITICAL-MM — Datasets

A "dataset" in CRITICAL-MM is one ICU source (eICU, MIMIC-IV, HiRID, NWICU, or a new one you bring). Each dataset has a reader class that produces eight canonical interim parquets — these are the substrate every task builds on top of.

## Contract

```python
from critical_mm.api import DatasetReader, register_dataset
from critical_mm.schema import TABLES
from typing import ClassVar
import polars as pl

@register_dataset
class MyHospitalReader(DatasetReader):
    DATASET_NAME: ClassVar[str] = "myhosp" # registry key, dir name, stay_id prefix

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    def read_stays(self) -> pl.LazyFrame:...
    def read_events_long(self, concepts: list[str]) -> pl.LazyFrame:...
    def read_meds(self) -> pl.LazyFrame:...
    def read_interventions(self) -> pl.LazyFrame:...
    def read_notes(self) -> pl.LazyFrame:...
    def read_diagnoses(self) -> pl.LazyFrame:...
    def read_microbio(self) -> pl.LazyFrame:... # empty if none
    # read_abx_duration has a default-empty implementation; override only if your
    # ricu concept-dict entry exists.
```

The base `DatasetReader.harmonise_all` runs each `read_*` in turn, validates the output against `critical_mm.schema.TABLES`, and writes through the content-hash cache to `data/interim/<dataset_name>/<table>.parquet`.

## The eight canonical tables

| Table | One row per | Required columns | Notes |
|---|---|---|---|
| `stays` | ICU stay | `patient_id, stay_id, intime, outtime, los_hours, mortality_in_icu, age, sex, weight, height, hospital_id` | The cohort foundation. Every other table joins on `stay_id`. |
| `events_long` | (stay, charttime, concept) | `patient_id, stay_id, charttime, concept_name, value, unit` | Vitals + labs in long format. ~50M rows for eICU. Use polars lazy scan. |
| `meds` | medication admin | `stay_id, starttime, endtime, drug_name, drug_class, dose, dose_unit` | drug_class is the harmonised ATC-like category ("antibiotic", "vasopressor",...). |
| `interventions` | procedure / event | `stay_id, charttime, intervention_name, value` | E.g. mechanical ventilation start/stop, RRT initiation. Used by SOFA respiration arm. |
| `notes` | clinical note | `stay_id, charttime, note_text, note_type` | Currently unused by tasks but kept for future text-modality work. Empty for datasets without notes. |
| `diagnoses` | diagnosis code | `stay_id, code, code_system, time` | E.g. ICD-10. Used by cohort filtering for some research tasks. |
| `microbio` | culture sample | `patient_id, stay_id, charttime, specimen_type, organism` | Sepsis-3 `susp_inf_alt`. `organism=NULL` for culture-negative samples (the sampling itself is the signal). |
| `abx_duration` | antibiotic episode | `stay_id, starttime, endtime` | ricu-faithful antibiotic intervals. Default-empty if your dataset has no ricu concept-dict abx_duration source. |

Exact schemas live in `critical_mm/schema.py::TABLES`. Use `TABLES["<name>"][0]` as the schema dict for `pl.LazyFrame(schema=...)` to produce an empty frame with the right columns.

## Worked example: SyntheticReader

`critical_mm/datasets/synthetic.py` is the simplest concrete reader — it generates a 1,000-stay deterministic synthetic dataset with no I/O. Read it for the minimum-viable shape of each `read_*` method.

For a real CSV-backed dataset, look at `critical_mm/datasets/eicu.py` (simplest mature reader) or `critical_mm/datasets/mimic_iv.py` (largest, with the v3.1 vs v2.2 reviewer-defense documented inline).

## Per-dataset capability flags

Different datasets have different table coverage. formalizes this with a `CAPABILITIES` ClassVar:

```python
@register_dataset
class MyHospitalReader(DatasetReader):
    DATASET_NAME = "myhosp"
    # Declare what structural features this dataset supports.
    CAPABILITIES = frozenset({"urine", "abx_duration"})
    #... read_* methods...
```

Current capability vocabulary (open — add new strings as needed):

| Capability | Meaning |
|---|---|
| `microbio` | Non-empty `microbio.parquet` with reliable culture data |
| `urine` | Urine output events present in `events_long` |
| `abx_duration` | `read_abx_duration` returns ricu-faithful antibiotic episodes (not the default empty frame) |
| `notes` | Non-empty `notes.parquet` |

Built-in declarations:

| Dataset | CAPABILITIES |
|---|---|
| eicu | `{urine, abx_duration}` — no microbio per YAIB App D.3 |
| hirid | `{urine, abx_duration}` — no microbio (v1.1.1 lacks table) |
| miiv | `{microbio, urine, abx_duration, notes}` — full set |
| nwicu | `{abx_duration}` — no urine (no `outputevents.csv.gz`), no microbio, no notes |
| synthetic | `frozenset` — zero-capability sandbox |

Tasks declare requirements by overriding `Task.required_dataset_capabilities(dataset: str) -> frozenset[str]`. `train_one` raises `ValueError` with a precise "missing X" message when a (task, dataset) combo is incompatible. See [tasks.md](tasks.md#capability-requirements) for the task-side pattern.

## Per-dataset capability flags — legacy notes

The built-in tasks (Mortality24, AKI, Sepsis, LoS, KidneyFunction) do NOT declare requirements — their existing hardcoded per-dataset dispatch (e.g. `Sepsis.supports_sep3_arm("nwicu") == False` triggers an abx-only surrogate) stays in place for backward compatibility. is opt-in: new contrib tasks should use the capability mechanism instead of hardcoding dataset names.

## cache_source_paths

The base class defaults to "every file under raw_root is a dependency of every interim table." That's correct but coarse — any change anywhere in your dataset invalidates every table. To narrow:

```python
def cache_source_paths(self, table: str) -> list[Path]:
    if table == "stays":
        return [self.raw_root / "icustays.csv.gz"]
    if table == "events_long":
        return [
            self.raw_root / "chartevents.csv.gz",
            self.raw_root / "labevents.csv.gz",
        ]
    return super.cache_source_paths(table)
```

## Activation checklist

1. Save the file under `critical_mm/contrib/<your_dataset>.py`.
2. Add `@register_dataset` above the class.
3. Choose a `DATASET_NAME` that doesn't collide (decorator raises `ValueError` on collision).
4. Test: `python -c "from critical_mm.registry import discover_datasets; print('<your_ds>' in discover_datasets)"` → True.
5. Build interim: `python -c "from <your_module> import MyReader; from pathlib import Path; MyReader(raw_root=Path('...'), interim_root=Path('data/interim'), repo_root=Path('.')).harmonise_all"`.
6. Then any existing task should run against your dataset: `python scripts/train.py --tasks mortality24 --datasets <your_ds> --models GRU`.

## See also

- `critical_mm/datasets/base.py` — full DatasetReader contract
- `critical_mm/datasets/synthetic.py` — simplest worked example
- `critical_mm/datasets/eicu.py` — mature CSV-backed reader
- `critical_mm/contrib/_examples/example_dataset.py` — copy-paste template
