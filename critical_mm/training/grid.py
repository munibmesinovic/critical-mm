"""train_grid: orchestrator over (task x dataset x model x seed) cells.

Calls critical_mm.training.train.train_one in-process for each cell,
collects per-cell metadata into a single grid summary at
data/checkpoints/grid_summary.json.

Usage:
    from critical_mm.training.grid import run_grid
    run_grid(tasks=['sepsis'], datasets=['eicu'], models=['GRU'], seeds=[42])
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from critical_mm.training.config import REPO, TrainConfig
from critical_mm.training.train import train_one

_DEFAULT_TASKS: tuple[str, ...] = ("sepsis", "aki", "mortality24", "los", "kidney_function")
_DEFAULT_DATASETS: tuple[str, ...] = ("eicu", "miiv", "hirid", "omix")
_DEFAULT_MODELS: tuple[str, ...] = ("GRU", "LSTM", "TCN", "Transformer")
_DEFAULT_SEEDS: tuple[int, ...] = (42, 1337, 2024)

def run_grid(
    tasks: Sequence[str] = _DEFAULT_TASKS,
    datasets: Sequence[str] = _DEFAULT_DATASETS,
    models: Sequence[str] = _DEFAULT_MODELS,
    seeds: Sequence[int] = _DEFAULT_SEEDS,
    cv_repetitions: int = 5,
    cv_folds: int = 5,
    repetition_index: int = 0,
    fold_index: int = 0,
    cpu: bool = True,
    debug: bool = False,
    skip_existing: bool = True,
    summary_path: Path | None = None,
    data_root: Path | None = None,
    splits_root: Path | None = None,
    checkpoint_root: Path | None = None,
    extra_hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Iterate over (task x dataset x model x seed) cells, calling train_one.

    When ``skip_existing`` is True (default), cells whose checkpoint dir already
    contains a ``metadata.json`` are recorded as ``skipped_existing`` and not
    retrained. This protects the bit-exact preservation contract for cells in
    a prior run (e.g. the seed=42 CM_REFERENCE_V1 baseline). Pass
    ``skip_existing=False`` to force a full retrain.

    ``extra_hyperparams`` is an optional dict of trainer / model overrides
    injected into every constructed ``TrainConfig``.  When ``None`` or empty
    (the default), behaviour is identical to prior runs (preservation-safe).
    Typical use: cap FM probe cells via ``{"max_epochs": 5}``.
    """
    data_root = data_root if data_root is not None else REPO
    _summary_base = checkpoint_root or (data_root / "data" / "checkpoints")
    summary_path = summary_path or (_summary_base / "grid_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _extra: dict[str, Any] = dict(extra_hyperparams) if extra_hyperparams else {}

    total = len(tasks) * len(datasets) * len(models) * len(seeds)
    print(
        f"=== train_grid: {len(tasks)} tasks x {len(datasets)} ds x "
        f"{len(models)} models x {len(seeds)} seeds = {total} cells "
        f"(skip_existing={skip_existing}) ==="
    )

    total_start = time.perf_counter()
    cells: list[dict[str, Any]] = []
    n_ok = 0
    n_fail = 0
    n_skip = 0
    n_skipped_existing = 0
    for task in tasks:
        for dataset in datasets:
            for model in models:
                for seed in seeds:
                    label = f"{task}/{dataset}/{model}/seed_{seed}"
                    t0 = time.perf_counter()
                    cfg = TrainConfig(
                        task=task,
                        dataset=dataset,
                        model=model,
                        seed=seed,
                        cv_repetitions=cv_repetitions,
                        cv_folds=cv_folds,
                        repetition_index=repetition_index,
                        fold_index=fold_index,
                        cpu=cpu,
                        debug=debug,
                        data_root=data_root,
                        splits_root=splits_root,
                        checkpoint_root=checkpoint_root,
                        extra_hyperparams=_extra,
                    )
                    existing_meta_path = cfg.checkpoint_dir / "metadata.json"
                    existing_meta = None
                    existing_complete = False
                    if existing_meta_path.exists():
                        try:
                            existing_meta = json.loads(existing_meta_path.read_text())
                            existing_complete = bool(existing_meta.get("test_metrics"))
                        except (json.JSONDecodeError, OSError):
                            existing_complete = False
                    if skip_existing and existing_complete:
                        status = "skipped_existing"
                        cell_record = {
                            "cell": label,
                            "status": status,
                            "duration_s": 0.0,
                            "metadata": existing_meta,
                            "note": "preserved bit-exact from prior run",
                        }
                    else:
                        try:
                            result = train_one(cfg)
                            status = result["status"]
                            cell_record = {
                                "cell": label,
                                "status": status,
                                "duration_s": round(time.perf_counter() - t0, 1),
                                "metadata": result["metadata"],
                            }
                        except Exception as exc:
                            status = "error"
                            cell_record = {
                                "cell": label,
                                "status": status,
                                "duration_s": round(time.perf_counter() - t0, 1),
                                "error": repr(exc),
                            }

                    cells.append(cell_record)
                    if status == "ok":
                        n_ok += 1
                    elif status == "error":
                        n_fail += 1
                    elif status == "skipped_existing":
                        n_skipped_existing += 1
                    else:
                        n_skip += 1

                    print(f"  [{status:18s}] {label:50s} ({cell_record['duration_s']:.1f}s)")

                    summary_path.write_text(
                        json.dumps(
                            {
                                "n_cells_total": total,
                                "n_ok": n_ok,
                                "n_fail": n_fail,
                                "n_skip": n_skip,
                                "n_skipped_existing": n_skipped_existing,
                                "skip_existing": skip_existing,
                                "cells": cells,
                            },
                            indent=2,
                            default=str,
                        )
                    )

    elapsed = time.perf_counter() - total_start
    print(
        f"\n=== Done in {elapsed:.1f}s ({n_ok} ok, {n_fail} fail, "
        f"{n_skip} skip, {n_skipped_existing} skipped_existing) ==="
    )
    return {
        "n_cells_total": total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_skip": n_skip,
        "n_skipped_existing": n_skipped_existing,
        "elapsed_s": round(elapsed, 1),
        "summary_path": str(summary_path),
    }

__all__ = ["run_grid"]

