# CRITICAL-MM

A multi-dataset benchmark for clinical prediction in the ICU. CRITICAL-MM
harmonizes seven intensive-care databases into a single schema and a fixed grid
of tasks, datasets, and models, so deep-learning and machine-learning baselines
are trained and evaluated under identical conditions — and new models (including
multimodal ones) drop in next to them.

- **Datasets:** eICU, MIMIC-IV, HiRID, NWICU, OMIX, SICdb, and Zigong.
- **Tasks:** mortality (24h), acute kidney injury, sepsis, length of stay,
  kidney function.
- **Models:** GRU, LSTM, TCN, Transformer (deep learning) and LightGBM plus
  scikit-learn baselines (machine learning).
- **Multimodal:** diagnosis (ICD), treatment, and clinical-notes surfaces with a
  swappable fusion interface.

## Install

```bash
pip install -e .[dev]
# optional, only to rebuild note embeddings from raw text:
pip install -e .[multimodal]
```

Requires Python 3.11+. The training stack (PyTorch, Lightning, LightGBM) is part
of the core dependencies because the baselines cannot run without it.
`environment.yml` and `requirements-lock.txt` pin the exact versions we ran.

## Data

Processed tensors and trained checkpoints are delivered out-of-band as a bundle
(they are not in this repository, owing to size and data-use agreements). Unpack
the bundle and point the code at it:

```bash
tar -xf critical-mm-bundle.tar -C /path/to/data-root
export CRITICAL_MM_DATA_ROOT=/path/to/data-root
```

`CRITICAL_MM_DATA_ROOT` governs the cohort and modality paths resolved by
`critical_mm/io/paths.py` — `data/processed/...` and
`data/processed/_modalities/...`. It does **not** govern where splits and
checkpoints are read from; those are passed explicitly, as below. With no bundle
you can still smoke-test on the bundled `synthetic` dataset.

> **Data-use note:** these databases are separately credentialed — MIMIC-IV and
> eICU via PhysioNet, and HiRID, NWICU, OMIX, SICdb and Zigong under their own
> agreements. Confirm your access terms before redistributing any derived tensors.

## Splits are patient-grouped

Splits group by patient, so no patient appears in more than one fold. The locked
tree is `data/processed/splits_patient/`, with checkpoints written to a parallel
`data/checkpoints_patient/`. Reported results use seeds `{42, 1337, 2024}`.

`critical_mm/patient_splits.py` implements the grouping and
`scripts/lock_splits_patient.py` writes the locked folds.
`scripts/verify_patient_splits.py` checks a tree against its manifests.

The 42 `manifest.json` files under `data/processed/splits_patient/` are tracked
in this repository. They carry per-cohort content hashes and fold row counts —
no identifiers — so you can confirm a split tree you regenerated matches ours:

```bash
python scripts/verify_patient_splits.py --repo /abs/path/to/repo
```

> `critical_mm/splits.py` predates this and is no longer used by any code path.
> `critical_mm/patient_splits.py` is the one to read.

## Run a baseline

```bash
python scripts/train.py --models GRU --tasks sepsis --datasets miiv --seeds 42 \
    --splits-root     /abs/path/to/data/processed/splits_patient \
    --checkpoint-root /abs/path/to/data/checkpoints_patient
```

**Both roots must be absolute** — the run metadata records them relative to the
repository, and a relative root fails.

This trains one cell and writes its metrics to
`<checkpoint-root>/<task>/<dataset>/<model>/seed_42/metadata.json`.
`scripts/aggregate_cm_grid.py --ckpt-root <checkpoint-root> --out <file>`
collates many cells into a comparison table.

Reuse is decided by fingerprint, not by file existence: a checkpoint on disk
records only that a cell trained at some point, not what it trained on. See
`critical_mm.validation.cohort_fingerprint.checkpoint_is_current`, which compares
cohort hashes, split hashes and a builder-code hash. Re-lock splits after
rebuilding a cohort and *before* retraining.

## Add your own model

Two routes, both via the auto-discovery registry — no edits to the core package.

**Unimodal.** Copy `critical_mm/contrib/_examples/example_model.py` to
`critical_mm/contrib/my_model.py`, adjust it, then:

```bash
python scripts/train.py --models ExampleNet --tasks mortality24 \
    --datasets synthetic --seeds 42
```

The model contract (instantiation signature, per-timestep `forward`) is in
[`docs/extending/models.md`](docs/extending/models.md).

**Multimodal.** Copy `critical_mm/contrib/_examples/example_fusion.py` to
`critical_mm/contrib/my_fusion.py` to add a `@register_fusion` strategy that
consumes the ICD / treatment / notes blocks alongside the structured stream,
then run it via `scripts/fusion_grid.py --strategy <name> ...`.

**Load a checkpoint.** Each checkpoint is a standard Lightning checkpoint of the
`DLPredictionWrapper` subclass; load it with the model class's
`load_from_checkpoint`.

## Compare against the reference

`baseline_metrics.json` ships the across-seed mean (and seed count) of every
baseline and fusion cell in the reference grid, keyed
`{task: {dataset: {model: {metric, seed_mean, n_seeds}}}}`. Put your numbers next
to ours without re-running the grid. It is generated by
`scripts/export_baseline_metrics.py` from the patient-grouped reference grid; the
provenance is recorded in the file's `_meta` block.

## Scope

This repository is the benchmark's source and extension surface: harmonisation,
tasks, models, the fusion interface, and the reference numbers to compare
against. It is not a one-command reproduction of every published table — the
cross-dataset transfer and foundation-model arms are not included here.

## License

MIT. See [LICENSE](LICENSE).
