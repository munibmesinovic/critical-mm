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

Run (training env, mem-capped):  bash scripts/run_fusion_grid.sh

This module stays torch-free at import time: every heavy import lives inside a
function body. FusionConfig (torch-free) is imported at module top.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from critical_mm.fusion.config import FusionConfig
from critical_mm.training.config import REPO, TrainConfig
from critical_mm.validation.cohort_fingerprint import code_fingerprint

TASKS = ("mortality24", "aki", "sepsis", "los", "kidney_function")
DATASETS = ("miiv", "eicu", "omix", "sicdb", "zigong", "nwicu")
RUNGS = ("icd", "icd_notes")

ISO_DATA_ROOTS: dict[str, Path] = {
    "nwicu": REPO / "data_relock_v6",
    "omix": REPO / "data_relock_v3",
}
_NOTES_DATASETS = frozenset({"miiv", "eicu", "omix"})
SEEDS = (42, 1337, 2024)
FUSION_ROOT = REPO / "data" / "checkpoints" / "_fusion"

_FUSION_CODE_FP = code_fingerprint([
    REPO / "critical_mm" / "fusion" / "blocks.py",
    REPO / "critical_mm" / "fusion" / "strategy.py",
    REPO / "critical_mm" / "fusion" / "leakage_matrix.py",
])
CLASSIFICATION = frozenset({"mortality24", "aki", "sepsis"})

def _cohort_data_root(dataset: str, splits_root: Path | None = None) -> Path:
    """Structured-cohort read root for ``dataset``.

    LOCKED stay-level grid (``splits_root=None``): nwicu/omix live off the main
    tree in their alternate processed-data root roots (v6/v3) — byte-identical to history.

    PATIENT splits (``splits_root`` set): ALL cohorts come from the MAIN tree
    (REPO). the initial split work generated the patient splits against the main-tree cohorts; the
    v6/v3 ISO cohorts carry a DISJOINT stay_id namespace (verified: 0% overlap
    with splits_patient/nwicu train ids, and a different outc schema), so ISO
    routing under patient splits joins to an EMPTY train set (the observed
    `weight tensor shape [0]` crash on mortality24/nwicu/LSTM__icd)."""
    if splits_root is not None:
        return REPO
    return ISO_DATA_ROOTS.get(dataset, REPO)

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

def _fusion_root(checkpoint_root: Path | None = None) -> Path:
    """Isolated _fusion write-root, honoring the the patient-grouped split work ``checkpoint_root``.

    Default (None) -> FUSION_ROOT (REPO/data/checkpoints/_fusion), byte-identical
    to history. When the patient-grouped split work supplies ``checkpoint_root`` (e.g. data/checkpoints_patient)
    the isolated _fusion subtree is redirected under it. ``data_root`` (the
    nwicu/omix ISO cohort routing) is orthogonal and untouched.
    """
    if checkpoint_root is None:
        return FUSION_ROOT
    return checkpoint_root / "_fusion"

def _checkpoint_dir(cell: Cell, checkpoint_root: Path | None = None) -> Path:
    return (
        _fusion_root(checkpoint_root)
        / cell.task
        / cell.dataset
        / cell.cell_model_key()
        / f"seed_{cell.seed}"
    )

def _cell_is_current(
    ckpt: Path, task: str, dataset: str, splits_root: Path | None = None
) -> tuple[bool, str]:
    """Fingerprint-aware replacement for a bare existence check on metadata.json.

    Existence says a cell trained, not what on. Checks cohort bytes, split hashes
    and the fusion builder-code hash; a missing hash is STALE, not a pass. See
    critical_mm/validation/cohort_fingerprint.py for why each axis matters.

    ``splits_root`` MUST be the same value the cell was (or would be) trained
    with. It mirrors ``TrainConfig.splits_dir`` exactly (splits_root when given,
    else data_root/data/processed/splits -- see scripts/treatments_grid.py's
    identical resolution), and the cohort root is resolved via
    ``_cohort_data_root`` (alternate processed-data root for nwicu/omix on the LOCKED grid, REPO
    for every dataset once patient splits are set). Hardcoding
    ``splits_patient/`` here compared every cell trained in the DEFAULT mode
    (``splits_root=None`` -> locked stay-level splits) against a manifest it was
    never trained on -- retrain forever.
    """
    from critical_mm.validation.cohort_fingerprint import (
        checkpoint_is_current,
        split_fingerprint,
    )

    cohort_root = _cohort_data_root(dataset, splits_root)
    cohort_dir = cohort_root / "data" / "processed" / task / dataset
    splits_base = (
        splits_root
        if splits_root is not None
        else cohort_root / "data" / "processed" / "splits"
    )
    splits_dir = splits_base / task / dataset
    return checkpoint_is_current(
        ckpt / "metadata.json",
        cohort_dir,
        split_expectation=split_fingerprint(splits_dir),
        code_expectation=_FUSION_CODE_FP,
        require_split=True,
        require_code=True,
    )

def _build_cfg(
    task: str,
    dataset: str,
    model: str,
    seed: int,
    *,
    data_root: Path | None = None,
    splits_root: Path | None = None,
    checkpoint_root: Path | None = None,
) -> TrainConfig:
    """Build one fusion cell's TrainConfig, threading the the patient-grouped split work split/checkpoint
    roots. ``data_root`` routes the STRUCTURED cohort read (alternate processed-data root for
    nwicu/omix; REPO otherwise). ``splits_root``/``checkpoint_root`` are
    ORTHOGONAL: they redirect the splits read (leakage-critical: patient splits)
    and the _ml_array_cache write into the parallel patient tree. All three
    default None -> byte-identical locked REPO paths."""
    return TrainConfig(
        task=task,
        dataset=dataset,
        model=model,
        seed=seed,
        data_root=data_root or REPO,
        splits_root=splits_root,
        checkpoint_root=checkpoint_root,
    )

def _patch_fusion_provenance(ckpt: Path, fusion: FusionConfig, rung: str) -> dict:
    """Read the metadata.json the locked training func just wrote, add fusion
    provenance (block fingerprint + rung), and rewrite. Keeps the locked
    training functions unaware of fusion."""
    import json

    meta = json.loads((ckpt / "metadata.json").read_text())
    meta["fusion"] = fusion.to_metadata()
    meta["config"]["fusion_rung"] = rung
    meta["code_fingerprint"] = _FUSION_CODE_FP
    (ckpt / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta

def _train_lgbm(
    cell: Cell,
    fusion: FusionConfig,
    ckpt: Path,
    data_root: Path | None = None,
    splits_root: Path | None = None,
    checkpoint_root: Path | None = None,
    apply_leakage_matrix: bool = True,
) -> dict:
    """In-process LGBM fit on the AUGMENTED arrays for one fusion cell.

    Calls the REAL critical_mm.training.train._fit_ml_inprocess (single source
    of truth) on arrays materialized via the augmented preamble. No on-disk ML
    array cache: the augmented arrays must not pollute the locked
    _ml_array_cache (keyed only by (task, dataset)+splits fp). Heavy imports are
    local so this module stays torch-free at import time.

    ``apply_leakage_matrix`` (default ``True``) is forwarded to
    ``_train_one_preamble`` -> the treatments block. ``True`` is the gated
    (locked) behavior, byte-identical to prior runs; ``False`` is the naive
    leakage-audit arm. It affects ONLY the treatments rung.
    """
    import time
    from dataclasses import asdict

    from pytorch_lightning import seed_everything

    from critical_mm.models._runmode import RunMode
    from critical_mm.registry import discover_models
    from critical_mm.training.train import _fit_ml_inprocess, _train_one_preamble

    t0 = time.perf_counter()
    seed_everything(cell.seed, workers=True)

    cfg = _build_cfg(
        cell.task,
        cell.dataset,
        cell.ml_model(),
        cell.seed,
        data_root=data_root,
        splits_root=splits_root,
        checkpoint_root=checkpoint_root,
    )
    ctx = _train_one_preamble(
        cfg,
        generate_features=True,
        ram_cache=False,
        write_metadata=False,
        fusion=fusion,
        apply_leakage_matrix=apply_leakage_matrix,
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
    metadata["config"].pop("data_root", None)

    import gc

    del ctx
    gc.collect()

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
        low_memory=True,
    )
    res["metadata"] = _patch_fusion_provenance(ckpt, fusion, cell.rung)
    return res

def _train_cell(
    cell: Cell,
    ctx: object | None = None,
    data_root: Path | None = None,
    splits_root: Path | None = None,
    checkpoint_root: Path | None = None,
    apply_leakage_matrix: bool = True,
    force: bool = False,
) -> dict:
    """Train one fusion cell. Resumable: skips if ckpt/metadata.json exists.

    ``force=True`` overrides that skip. It exists because the existence test here is
    the INNERMOST gate: a caller that has already decided a cell is stale — because
    its cohort moved, or because the block builder changed — still gets "skipped"
    from this line, silently. That made a treatments re-run print "retrain", call
    this function, get skipped, and report 210/210 complete having trained nothing:
    strictly worse than the old behaviour, which at least said "skip". Callers with
    their own staleness gate MUST pass force=True.

    LSTM: build (or reuse the passed-in) augmented _TrainContext via
    _train_one_preamble(generate_features=False, fusion=...), then call the REAL
    _train_one_dl with checkpoint_dir=ckpt.
    LGBM: _train_lgbm (materializes its own augmented arrays per cell).

    ``data_root`` routes the STRUCTURED data read to an alternate root (e.g. an
    alternate processed-data root tree for a dataset whose main processed tree is stale). It is the
    only routing knob: ``None`` -> REPO is byte-identical to prior behavior (the
    TrainConfig.data_root default is also REPO). The treatment-block surfaces
    keep coming from the main tree (loader.MODALITIES_ROOT is untouched).

    ``apply_leakage_matrix`` (default ``True``) is forwarded to the LGBM /
    LSTM training path -> the treatments block. ``True`` is the gated (locked)
    behavior, byte-identical to prior runs; ``False`` is the naive leakage-audit
    arm. A naive cell MUST be written to a DISTINCT checkpoint dir (the
    fingerprint does not encode the flag) — the audit driver supplies that.
    """
    ckpt = _checkpoint_dir(cell, checkpoint_root)
    is_current, _why = _cell_is_current(ckpt, cell.task, cell.dataset, splits_root=splits_root)
    if is_current and not force:
        return {"status": "skipped", "ckpt": str(ckpt)}

    fusion = FusionConfig(rung=cell.rung)

    if cell.backbone == "LGBM":
        return _train_lgbm(
            cell,
            fusion,
            ckpt,
            data_root=data_root,
            splits_root=splits_root,
            checkpoint_root=checkpoint_root,
            apply_leakage_matrix=apply_leakage_matrix,
        )

    from critical_mm.registry import discover_models
    from critical_mm.training.train import _train_one_dl, _train_one_preamble

    cfg = _build_cfg(
        cell.task,
        cell.dataset,
        "LSTM",
        cell.seed,
        data_root=data_root,
        splits_root=splits_root,
        checkpoint_root=checkpoint_root,
    )
    if ctx is None:
        ctx = _train_one_preamble(
            cfg,
            generate_features=False,
            write_metadata=False,
            fusion=fusion,
            dynamic_pad=True,
            apply_leakage_matrix=apply_leakage_matrix,
        )
    model_class = discover_models()["LSTM"]
    res = _train_one_dl(cfg, ctx, model_class, checkpoint_dir=ckpt, dynamic_pad=True)
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
    ap.add_argument(
        "--splits-root", type=Path, default=None,
        help="the patient-grouped split work: read splits from this root (e.g. data/processed/splits_patient). "
        "Default (None) -> locked stay-level splits.",
    )
    ap.add_argument(
        "--checkpoint-root", type=Path, default=None,
        help="the patient-grouped split work: redirect the isolated _fusion write-root under this "
        "(e.g. data/checkpoints_patient). Default (None) -> locked REPO tree.",
    )
    args = ap.parse_args()
    splits_root = args.splits_root
    checkpoint_root = args.checkpoint_root

    total = len(enumerate_cells())
    cells = enumerate_cells()
    for attr in ("task", "dataset", "backbone", "rung", "seed"):
        val = getattr(args, attr)
        if val is not None:
            cells = [c for c in cells if getattr(c, attr) == val]
    print(f"[fusion] {len(cells)} cells selected (of {total})", flush=True)
    print(
        f"[fusion] splits_root={splits_root or '<locked>'} "
        f"checkpoint_root={checkpoint_root or '<locked>'}",
        flush=True,
    )

    import copy
    import gc
    import time

    from pytorch_lightning import seed_everything

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
                    c
                    for c in lstm_cells
                    if not _cell_is_current(
                        _checkpoint_dir(c, checkpoint_root), c.task, c.dataset, splits_root=splits_root
                    )[0]
                ]
                if pending:
                    fusion = FusionConfig(rung=rung)
                    ds_root = _cohort_data_root(ds, splits_root)
                    cfg = _build_cfg(
                        task,
                        ds,
                        "LSTM",
                        pending[0].seed,
                        data_root=ds_root,
                        splits_root=splits_root,
                        checkpoint_root=checkpoint_root,
                    )
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
                        _train_cell(
                            c,
                            ctx=seed_ctx,
                            data_root=_cohort_data_root(ds, splits_root),
                            splits_root=splits_root,
                            checkpoint_root=checkpoint_root,
                        )
                        done += 1
                        print(f"[fusion] progress {done}", flush=True)
                for c in lstm_cells:
                    if (
                        _checkpoint_dir(c, checkpoint_root) / "metadata.json"
                    ).exists() and c not in pending:
                        done += 1

                ctx = None
                gc.collect()

                lgbm_cells = [
                    c
                    for c in cells
                    if c.task == task
                    and c.dataset == ds
                    and c.rung == rung
                    and c.backbone == "LGBM"
                ]
                for c in lgbm_cells:
                    ckpt_c = _checkpoint_dir(c, checkpoint_root)
                    is_current, why = _cell_is_current(ckpt_c, c.task, c.dataset, splits_root=splits_root)
                    if is_current:
                        done += 1
                        continue
                    if (ckpt_c / "metadata.json").is_file():
                        print(f"[fusion] RETRAIN {ckpt_c.relative_to(REPO)}: {why}", flush=True)
                    print(
                        f"[fusion] train {task}/{ds}/{c.cell_model_key()}/seed_{c.seed}",
                        flush=True,
                    )
                    _train_cell(
                        c,
                        data_root=_cohort_data_root(ds, splits_root),
                        splits_root=splits_root,
                        checkpoint_root=checkpoint_root,
                    )
                    done += 1
                    print(f"[fusion] progress {done}", flush=True)

    print(f"[fusion] grid complete ({done} cells)", flush=True)

if __name__ == "__main__":
    main()

