# Extending CRITICAL-MM — Concepts (vital/lab features)

> ⚠️ **WARNING:** adding a new concept invalidates the locked `CM_REFERENCE_v1` oracle. Read the bottom section before touching `configs/concepts_loinc.csv`.

A "concept" in CRITICAL-MM is a clinical feature (vital sign, lab test, etc.) with a canonical name, unit, and valid range. The 48 dyn features used by the v1 oracle are derived from a single CSV: `configs/concepts_loinc.csv`.

## Concept CSV format

`configs/concepts_loinc.csv` has columns:

| Column | Type | Purpose |
|---|---|---|
| `name` | str | Canonical short name (`hr`, `sbp`, `crea`,...). Used as dyn column name. |
| `category` | str | One of `vital`, `lab`, `outcome`. Only `vital` + `lab` enter `DYNAMIC_CONCEPTS`. |
| `loinc_code` | str | LOINC code for the concept (used by external mapping tools). |
| `canonical_unit` | str | Unit after harmonisation. **Required for inclusion in `DYNAMIC_CONCEPTS`.** |
| `valid_range_low` | float | Clinical lower bound. Values outside are clamped/dropped. **Required.** |
| `valid_range_high` | float | Clinical upper bound. **Required.** |
| (per-dataset itemid columns) | str/list | Mapping from this concept to each dataset's source itemids/varnames. Per-dataset readers consume these. |

The auto-discovery in `critical_mm/tasks/base.py`:

```python
DYNAMIC_CONCEPTS = sorted(
    name for name, cc in CONCEPTS_BY_NAME.items
    if cc.category in ("vital", "lab")
    and cc.canonical_unit is not None
    and cc.valid_range is not None
    and name not in _DYNAMIC_CONCEPT_EXCLUSIONS
)
```

So adding a row to the CSV with category + unit + valid_range automatically lands it in `DYNAMIC_CONCEPTS`. **That's exactly why this is dangerous** — see below.

## Excluded concepts

`_DYNAMIC_CONCEPT_EXCLUSIONS` in `tasks/base.py` carries concepts that are registered (have unit + valid_range) but intentionally NOT in the dyn schema:

- `gcs` — used internally by the Phase 2 SOFA engine (CNS sub-score), not as a direct dyn feature.
- `urine_rate` — used internally by AKI's urine arm, not as a direct dyn feature.

If your new concept is "internal use only," add it to this exclusion set.

## ⚠ CM_REFERENCE_v1 invariant

`CM_REFERENCE` (in `critical_mm/validation/oracle.py`) is the locked 80-cell ship gate. The DL models that produced those numbers have `input_dim = len(DYNAMIC_CONCEPTS) = 48`. Adding a 49th concept means:

1. Trained checkpoints fail to load with `input_dim mismatch` errors.
2. CM_REFERENCE numbers become meaningless for the new schema (they describe a 48-feature model, you're now training a 49-feature one).
3. Any oracle comparison against CM_REFERENCE produces `outside_tolerance` even on identical training because the architecture is different.

This isn't a bug — it's the locked-reference contract. Once we publish CM_REFERENCE_v1, **everyone training against v1 uses 48 features**. 's multi-version registry (see Option A below) is the mechanism for adding new concepts without invalidating v1: publish a `CM_REFERENCE_v2` alongside.

## How to actually add a concept

Two options:

### Option A — publish a CM_REFERENCE_v2

 introduced multi-version coexistence: `CM_REFERENCE_V1` stays locked at 48 features for paper reproducibility; new versions register alongside via `register_cm_reference`. Workflow:

1. Add the row to `configs/concepts_loinc.csv` with `canonical_unit` + `valid_range`.
2. Add per-dataset itemid mappings for every dataset you train on.
3. Re-run the full grid against the expanded schema: `bash scripts/run_cm_grid.sh`. The new 49-feature checkpoints land under `data/checkpoints/` alongside the v1 ones (different `input_dim`, so they won't collide with v1 cells).
4. Aggregate into a v2 reference JSON, then register it:

   ```python
   from critical_mm.validation.oracle import register_cm_reference

   register_cm_reference(
       "v2",
       cells={...}, # your 80+ cells with the 49-feature architecture
       manifest={
           "source_json": "data/cm_reference/v2/cm_grid_metrics.json",
           "git_sha": "<your-grid-run-sha>",
           "n_seeds_per_cell": 3,
           "schema_notes": "Adds <your-concept> as 49th dyn feature",
       },
   )
   ```

5. Query the oracle: `run_cm_oracle_cell(..., version="v2")`. v1 cells still validate via `version="v1"` (default).

This is the right path for any concept addition we expect to publish. Reviewers can verify v1 numbers (locked-paper reproducibility) AND v2 numbers (your contribution) side-by-side.

### Option B — fork your own benchmark

For internal-only research where you don't need to compare against the published v1 numbers:

1. Add the row to `configs/concepts_loinc.csv` with canonical_unit + valid_range.
2. Add per-dataset itemid mappings for every dataset you train on.
3. Re-run the full grid: `bash scripts/run_cm_grid.sh`. New checkpoints with 49-feature input.
4. Re-aggregate: `python scripts/aggregate_cm_grid.py && python scripts/lift_cm_reference.py`. CM_REFERENCE is now your fork's 49-feature reference.
5. Document your fork's schema bump in a `docs/sessions/<date>-schema-v2.md`.

This invalidates forever for your fork — you've baked in your custom schema. That's a defensible choice for internal projects; less so for publishable benchmarks.

## Suggested workflow for "I want to try a new feature"

If you're exploring whether a new feature improves performance (the most common case):

1. Don't touch `concepts_loinc.csv`. Instead, build your feature into a **per-task** processor.
2. Subclass an existing task, add your feature inside `build_labels` as part of the outc frame or as a custom dyn-side processor.
3. Train and measure on your task — you're now comparing CM_REFERENCE_v1 (48-feature baseline) against your_task (48-feature + your engineered feature). That's a clean ablation.

Most "new feature" exploration is feature engineering, not schema extension — Option B above is rarely actually needed.

## See also

- `configs/concepts_loinc.csv` — the canonical concept registry
- `critical_mm/concepts.py` — Python wrapper around the CSV
- `critical_mm/tasks/base.py::DYNAMIC_CONCEPTS` — the auto-derived 48-feature list
- `critical_mm/validation/oracle.py::CM_REFERENCE` — the locked 80-cell ship gate
- [overview.md](overview.md) — full extension surface map
