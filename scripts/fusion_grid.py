"""Sidecar grid runner for the fusion ablation ladder (CM_REFERENCE_V4).

Trains the 180 NEW cells (rungs icd + icd_notes; rung-1 reuses locked V3) under
an ISOLATED checkpoint root data/checkpoints/_fusion/, never touching the locked
tree. Both backbones call the REAL training functions from critical_mm.training
.train via an explicit ``checkpoint_dir`` override — no training logic is copied
here (single source of truth):

  * DL (LSTM): _train_one_preamble(fusion=...) builds the augmented context once
    per (task,dataset,rung); _train_one_dl(..., checkpoint_dir=ckpt) re-fits per
    seed and writes into the isolated _fusion dir.
  * LGBM: the augmented arrays are materialized per cell (no on-disk ML cache:
    the augmented arrays must not pollute the locked _ml_array_cache, keyed only
    by (task,dataset)+splits fp), then _fit_ml_inprocess(..., checkpoint_dir=ckpt)
    fits + writes.

Fusion provenance (rung + block fingerprint) is patched onto the written
metadata.json AFTER the locked funcs write it, so the locked training functions
stay unaware of fusion.

Run (training env, mem-capped): bash scripts/run_fusion_grid.sh

This module stays torch-free at import time: every heavy import lives inside a
function body. FusionConfig (torch-free) is imported at module top.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from critical_mm.fusion.config import FusionConfig
from critical_mm.training.config import REPO

TASKS = ("mortality24", "aki", "sepsis", "los", "kidney_function")
DATASETS = ("miiv", "eicu", "omix", "sicdb")
RUNGS = ("icd", "icd_notes")
_NOTES_DATASETS = frozenset({"miiv", "eicu", "omix"})
SEEDS = (42, 1337, 2024)
FUSION_ROOT = REPO / "data" / "checkpoints" / "_fusion"
CLASSIFICATION = frozenset({"mortality24", "aki", "sepsis"})

@dataclasses.dataclass(frozen=True)
class Cell:
    task: str
    dataset: str
    backbone: str
    rung: str
    seed: int

    def ml_model(self) -> str:
        if self.backbone != "LGBM":
            return self.backbone
        return "LGBMClassifier" if self.task in CLASSIFICATION else "LGBMRegressor"

    def cell_model_key(self) -> str:
        return f"{self.ml_model()}{FusionConfig(rung=self.rung).model_suffix()}"

def enumerate_cells() -> list[Cell]:
    cells = []
    for task in TASKS:
        for ds in DATASETS:
            for bb in ("LSTM", "LGBM"):
                for rung in RUNGS:
                    if rung == "icd_notes" and ds not in _NOTES_DATASETS:
                        continue
                    for seed in SEEDS:
                        cells.append(Cell(task, ds, bb, rung, seed))
    return cells

def _checkpoint_dir(cell: Cell) -> Path:
    return FUSION_ROOT / cell.task / cell.dataset / cell.cell_model_key() / f"seed_{cell.seed}"

def _patch_fusion_provenance(ckpt: Path, fusion: FusionConfig, rung: str) -> dict:
    """Read the metadata.json the locked training func just wrote, add fusion
    provenance (block fingerprint + rung), and rewrite. Keeps the locked
    training functions unaware of fusion."""
    import json

    meta = json.loads((ckpt / "metadata.json").read_text())
    meta["fusion"] = fusion.to_metadata()
    meta["config"]["fusion_rung"] = rung
    (ckpt / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta

def _train_lgbm(cell: Cell, fusion: FusionConfig, ckpt: Path) -> dict:
    """In-process LGBM fit on the AUGMENTED arrays for one fusion cell.

    Calls the REAL critical_mm.training.train._fit_ml_inprocess (single source
    of truth) on arrays materialized via the augmented preamble. No on-disk ML
    array cache: the augmented arrays must not pollute the locked
    _ml_array_cache (keyed only by (task, dataset)+splits fp). Heavy imports are
    local so this module stays torch-free at import time.
    """
    import time
    from dataclasses import asdict

    from pytorch_lightning import seed_everything

    from critical_mm.models._runmode import RunMode
    from critical_mm.registry import discover_models
    from critical_mm.training.config import TrainConfig
    from critical_mm.training.train import _fit_ml_inprocess, _train_one_preamble

    t0 = time.perf_counter()
    seed_everything(cell.seed, workers=True)

    cfg = TrainConfig(task=cell.task, dataset=cell.dataset, model=cell.ml_model(), seed=cell.seed)
    ctx = _train_one_preamble(
        cfg,
        generate_features=True,
        ram_cache=False,
        write_metadata=False,
        fusion=fusion,
    )
    Xtr, ytr = ctx.train_dataset.get_data_and_labels()
    Xva, yva = ctx.val_dataset.get_data_and_labels()
    Xte, yte = ctx.test_dataset.get_data_and_labels()
    feature_names = list(ctx.train_dataset.get_feature_names())

    runmode = RunMode.classification if cfg.is_classification else RunMode.regression

    metadata: dict[str, object] = {
        "config": asdict(cfg),
        "data_git_sha_at_train": ctx.metadata["data_git_sha_at_train"],
        "splits_fingerprint": ctx.metadata["splits_fingerprint"],
        "runmode": ctx.metadata["runmode"],
        "duration_load_s": ctx.metadata["duration_load_s"],
        "splits": ctx.metadata["splits"],
        "model_kind": "ml",
    }
    metadata["config"].pop("data_root", None) # type: ignore[attr-defined]

    model_class = discover_models()[cfg.model]
    res = _fit_ml_inprocess(
        cfg,
        Xtr=Xtr,
        ytr=ytr,
        Xva=Xva,
        yva=yva,
        Xte=Xte,
        yte=yte,
        feature_names=feature_names,
        metadata=metadata,
        runmode=runmode,
        model_class=model_class,
        t0=t0,
        checkpoint_dir=ckpt,
    )
    res["metadata"] = _patch_fusion_provenance(ckpt, fusion, cell.rung)
    return res

def _train_cell(cell: Cell, ctx: object | None = None) -> dict:
    """Train one fusion cell. Resumable: skips if ckpt/metadata.json exists.

    LSTM: build (or reuse the passed-in) augmented _TrainContext via
    _train_one_preamble(generate_features=False, fusion=...), then call the REAL
    _train_one_dl with checkpoint_dir=ckpt.
    LGBM: _train_lgbm (materializes its own augmented arrays per cell).
    """
    ckpt = _checkpoint_dir(cell)
    if (ckpt / "metadata.json").exists():
        return {"status": "skipped", "ckpt": str(ckpt)}

    fusion = FusionConfig(rung=cell.rung)

    if cell.backbone == "LGBM":
        return _train_lgbm(cell, fusion, ckpt)

    from critical_mm.registry import discover_models
    from critical_mm.training.config import TrainConfig
    from critical_mm.training.train import _train_one_dl, _train_one_preamble

    cfg = TrainConfig(task=cell.task, dataset=cell.dataset, model="LSTM", seed=cell.seed)
    if ctx is None:
        ctx = _train_one_preamble(
            cfg, generate_features=False, write_metadata=False, fusion=fusion, dynamic_pad=True
        )
    model_class = discover_models()["LSTM"]
    res = _train_one_dl(cfg, ctx, model_class, checkpoint_dir=ckpt, dynamic_pad=True) # type: ignore[arg-type]
    res["metadata"] = _patch_fusion_provenance(ckpt, fusion, cell.rung)
    return res

def main() -> None:
    """Run the 180-cell fusion grid, resumable + mem-friendly.

    LSTM cells of a (task, dataset, rung) share ONE augmented _TrainContext's
    EXPENSIVE datasets (built/tensorized once); each seed gets a cheap per-seed
    _TrainContext (same dataset refs, fresh t0 + deep-copied metadata with the
    cell's seed) so model randomness, durations, AND provenance are all per-seed.
    LGBM cells materialize their own augmented arrays per cell. Progress is
    printed with flush=True.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Fusion ablation ladder grid runner")
    ap.add_argument("--task", choices=TASKS, help="restrict to one task")
    ap.add_argument("--dataset", choices=DATASETS, help="restrict to one dataset")
    ap.add_argument("--backbone", choices=("LSTM", "LGBM"), help="restrict to one backbone")
    ap.add_argument("--rung", choices=RUNGS, help="restrict to one rung")
    ap.add_argument("--seed", type=int, choices=SEEDS, help="restrict to one seed")
    args = ap.parse_args()

    total = len(enumerate_cells())
    cells = enumerate_cells()
    for attr in ("task", "dataset", "backbone", "rung", "seed"):
        val = getattr(args, attr)
        if val is not None:
            cells = [c for c in cells if getattr(c, attr) == val]
    print(f"[fusion] {len(cells)} cells selected (of {total})", flush=True)

    import copy
    import time

    from pytorch_lightning import seed_everything

    from critical_mm.training.config import TrainConfig
    from critical_mm.training.train import _train_one_preamble

    done = 0
    for task in TASKS:
        for ds in DATASETS:
            for rung in RUNGS:
                lstm_cells = [
                    c
                    for c in cells
                    if c.task == task
                    and c.dataset == ds
                    and c.rung == rung
                    and c.backbone == "LSTM"
                ]
                pending = [
                    c for c in lstm_cells if not (_checkpoint_dir(c) / "metadata.json").exists()
                ]
                if pending:
                    fusion = FusionConfig(rung=rung)
                    cfg = TrainConfig(task=task, dataset=ds, model="LSTM", seed=pending[0].seed)
                    print(f"[fusion] preamble {task}/{ds}/LSTM__{rung}", flush=True)
                    ctx = _train_one_preamble(
                        cfg,
                        generate_features=False,
                        write_metadata=False,
                        fusion=fusion,
                        dynamic_pad=True,
                    )
                    for c in pending:
                        seed_everything(c.seed, workers=True)
                        print(
                            f"[fusion] train {task}/{ds}/{c.cell_model_key()}/seed_{c.seed}",
                            flush=True,
                        )
                        seed_meta = copy.deepcopy(ctx.metadata)
                        seed_meta["config"]["seed"] = c.seed
                        seed_ctx = dataclasses.replace(
                            ctx, t0=time.perf_counter(), metadata=seed_meta
                        )
                        _train_cell(c, ctx=seed_ctx)
                        done += 1
                        print(f"[fusion] progress {done}", flush=True)
                for c in lstm_cells:
                    if (_checkpoint_dir(c) / "metadata.json").exists() and c not in pending:
                        done += 1

                lgbm_cells = [
                    c
                    for c in cells
                    if c.task == task
                    and c.dataset == ds
                    and c.rung == rung
                    and c.backbone == "LGBM"
                ]
                for c in lgbm_cells:
                    if (_checkpoint_dir(c) / "metadata.json").exists():
                        done += 1
                        continue
                    print(
                        f"[fusion] train {task}/{ds}/{c.cell_model_key()}/seed_{c.seed}",
                        flush=True,
                    )
                    _train_cell(c)
                    done += 1
                    print(f"[fusion] progress {done}", flush=True)

    print(f"[fusion] grid complete ({done} cells)", flush=True)

if __name__ == "__main__":
    main()
