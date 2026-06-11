"""Export the reference grid metrics to a flat, public ``baseline_metrics.json``.

Reads the locked reference grid JSON (and, optionally, a fusion-ladder grid
JSON) and emits the across-seed mean for every baseline and fusion cell, so a
collaborator can compare their numbers against ours without the locked oracle or
any checkpoints. JSON in, JSON out -- no model imports, no filesystem walking of
``data/checkpoints``.

Output schema::

    {task: {dataset: {model: {"metric": str, "seed_mean": float, "n_seeds": int}}}}

where ``model`` carries the fusion rung as a ``+icd`` / ``+notes`` / ``+icd_notes``
suffix. Foundation-model, transfer, and ablation cells are filtered out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = ("eicu", "miiv", "hirid", "nwicu", "omix")
CLASSIFICATION_TASKS = frozenset({"sepsis", "aki", "mortality24"})
KEEP_RUNGS = frozenset({None, "icd", "notes", "icd_notes"})


def _is_fm(model: str) -> bool:
    return (
        model.startswith(("MOMENT", "Mantis", "Chronos", "Toto"))
        or "-probe" in model
        or "-ft" in model
    )


def _split_cell(key: str) -> tuple[str, str, str | None] | None:
    """Parse ``"<dataset>_<model>[__rung]"`` into (dataset, model, rung)."""
    dataset = next((d for d in DATASETS if key.startswith(d + "_")), None)
    if dataset is None:
        return None
    rest = key[len(dataset) + 1 :]
    if "__" in rest:
        model, rung = rest.split("__", 1)
    else:
        model, rung = rest, None
    return dataset, model, rung


def _metric_name(task: str) -> str:
    return "AUROC" if task in CLASSIFICATION_TASKS else "MAE"


def _ingest(reference: dict, out: dict) -> None:
    for task, cells in reference.items():
        metric = _metric_name(task)
        task_out = out.setdefault(task, {})
        for key, stats in cells.items():
            parsed = _split_cell(key)
            if parsed is None:
                continue
            dataset, model, rung = parsed
            if _is_fm(model):
                continue
            if rung is not None and "abl" in rung:
                continue
            if rung not in KEEP_RUNGS:
                continue
            label = model if rung is None else f"{model}+{rung}"
            ds_out = task_out.setdefault(dataset, {})
            if label in ds_out:
                continue
            ds_out[label] = {
                "metric": metric,
                "seed_mean": stats["mean"],
                "n_seeds": stats["n"],
            }


def build_baseline_metrics(src_root: Path) -> dict:
    """Build the flat baseline-metrics dict from the reference grid JSON(s)."""
    src_root = Path(src_root)
    primary = src_root / "data" / "cm_reference" / "v9" / "cm_grid_metrics.json"
    extra = src_root / "data" / "cm_reference" / "v4" / "cm_grid_metrics.json"

    out: dict = {}
    with primary.open() as fh:
        _ingest(json.load(fh)["reference"], out)
    if extra.exists():
        with extra.open() as fh:
            _ingest(json.load(fh)["reference"], out)
    return {
        task: {ds: dict(sorted(models.items())) for ds, models in sorted(dsmap.items())}
        for task, dsmap in sorted(out.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repo root containing data/cm_reference/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "baseline_metrics.json",
        help="output JSON path",
    )
    args = parser.parse_args()
    metrics = build_baseline_metrics(args.src)
    args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    n_cells = sum(len(m) for t in metrics.values() for m in t.values())
    print(f"wrote {args.out} ({len(metrics)} tasks, {n_cells} cells)")


if __name__ == "__main__":
    main()
