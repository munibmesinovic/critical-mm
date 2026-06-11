# Extending CRITICAL-MM

CRITICAL-MM is designed to be extended by external researchers. There are four extension surfaces:

| Surface | What it adds | When to use it |
|---|---|---|
| **Task** | A new prediction target with its own cohort builder + label semantics | Adding "ICU mortality at 48h" or "30-day readmission" |
| **Dataset** | A new ICU source (raw → 8 canonical interim parquets) | Adding a regional ICU you've collected, or a future MIMIC release |
| **Model** | A new DL or ML architecture wrapped for the training loop | Adding a Mamba block, a foundation model fine-tune, etc. |
| **Concept** | A new vital/lab feature in the canonical 48-feature schema | Adding a 49th dyn feature (e.g. troponin-T) — see WARNING below |

## How extensions get discovered

Each surface has a class decorator from `critical_mm.api`:

```python
from critical_mm.api import Task, register_task

@register_task
class Mortality48(Task):
    task_name = "mortality48"
    task_type = "classification"
    prediction_horizon_hours = 48
    def build_labels(self, base_cohort, events_long, meds, dataset, **_):
        ...
```

Two ways to make the decorator fire:

1. **Drop into `critical_mm/contrib/<name>.py`** — `critical_mm.registry._ensure_contrib_loaded` walks the contrib package on first `discover_*` call and imports every module. Your decorated class lands in the registry. `scripts/train.py` then accepts `--tasks mortality48` automatically.

2. **Pip-publish your own package** — put the decorated class in `my_extension_pkg/tasks/foo.py`, then `pip install -e.` your package and `import my_extension_pkg.tasks.foo` before running `scripts/train.py`. The decorator runs at import; the registry picks it up. (Entry-points-based auto-discovery is on the + roadmap but not yet implemented.)

The `critical_mm/contrib/_examples/` subpackage is **never** auto-loaded — it carries copy-paste templates only. To activate a template, copy it out of `_examples/`.

## Decision tree

```
What are you adding?
├── A new prediction target → see tasks.md
├── A new ICU data source → see datasets.md
├── A new neural network architecture or classical ML estimator → see models.md
└── A new vital/lab feature → see concepts.md ⚠ (invalidates CM_REFERENCE_v1!)
```

## Important: the CM_REFERENCE_v1 invariant

CM_REFERENCE (in `critical_mm/validation/oracle.py`) is the locked 80-cell ship gate that justifies the published benchmark numbers. It was trained on:

- The 48 features in `DYNAMIC_CONCEPTS` (sourced from `configs/concepts_loinc.csv`).
- The 5 canonical tasks (mortality24, AKI, sepsis, LoS, kidney_function).
- 4 datasets × 4 DL models × 1 seed.

You can **add new** tasks, datasets, and models without touching CM_REFERENCE — they simply produce *new* cells, not replacements for existing ones. Adding a **new feature concept** changes the architecture's `input_dim` so it can't share `CM_REFERENCE_v1`; instead, register a new version side-by-side via 's multi-version registry. See [concepts.md](concepts.md) for the workflow.

## See also

- [tasks.md](tasks.md) — full Task contract + worked example
- [datasets.md](datasets.md) — DatasetReader contract + worked example
- [models.md](models.md) — DL + ML wrapper contracts + worked examples
- [concepts.md](concepts.md) — concept CSV format + CM_REFERENCE versioning
