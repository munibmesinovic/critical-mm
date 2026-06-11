# CRITICAL-MM

A multi-dataset benchmark for clinical prediction in the ICU. CRITICAL-MM
harmonizes five intensive-care databases into a single schema and a fixed grid
of tasks, datasets, and models, so deep-learning and machine-learning baselines
are trained and evaluated under identical conditions — and new models (including
multimodal ones) drop in next to them.

- **Datasets:** eICU, MIMIC-IV, HiRID, NWICU, OMIX.
- **Tasks:** mortality (24h), acute kidney injury, sepsis, length of stay,
  kidney function.
- **Models:** GRU, LSTM, TCN, Transformer (deep learning) and LightGBM plus
  scikit-learn baselines (machine learning).
- **Multimodal:** diagnosis (ICD) and clinical-notes surfaces with a swappable
  fusion interface.

## Install

```bash
pip install -e .[dev]
# optional, only to rebuild note embeddings from raw text:
pip install -e .[multimodal]
```

Requires Python 3.11+. The training stack (PyTorch, Lightning, LightGBM) is part
of the core dependencies because the baselines cannot run without it.

## Data

Processed tensors and trained checkpoints are delivered out-of-band as a bundle
(they are not in this repository, owing to size and data-use agreements). Unpack
the bundle and point the code at it:

```bash
tar -xf critical-mm-bundle.tar -C /path/to/data-root
export CRITICAL_MM_DATA_ROOT=/path/to/data-root
```

Every path the code reads (`data/processed/...`, `data/checkpoints/...`,
`data/processed/_modalities/...`) resolves under `CRITICAL_MM_DATA_ROOT`. With
no bundle you can still smoke-test on the bundled `synthetic` dataset.

> **OMIX note:** the OMIX dataset is governed by a data-use agreement. Confirm
> your access terms before redistributing any OMIX-derived tensors.

## Run a baseline

```bash
python scripts/train.py --models GRU --tasks sepsis --datasets miiv --seeds 42
```

This trains one cell and writes its metrics to
`data/checkpoints/<task>/<dataset>/<model>/seed_42/metadata.json`.
`scripts/aggregate_cm_grid.py` collates many cells into a comparison table.

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
consumes the ICD / notes blocks alongside the structured stream, then run it via
`scripts/fusion_grid.py --strategy <name> ...`.

**Load a checkpoint.** Each checkpoint is a standard Lightning checkpoint of the
`DLPredictionWrapper` subclass; load it with the model class's
`load_from_checkpoint`.

## Compare against the reference

`baseline_metrics.json` ships the across-seed mean (and seed count) of every
baseline and fusion cell in the reference grid, keyed
`{task: {dataset: {model: {metric, seed_mean, n_seeds}}}}`. Put your numbers next
to ours without re-running the grid.

## Extending further

See [`docs/extending/`](docs/extending/) for the task, dataset, concept, and
model contracts.

## License

MIT. See [LICENSE](LICENSE).
