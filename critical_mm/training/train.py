"""CM-native training entry: one (task, dataset, model, seed) cell.

Uses CRITICAL-MM's locked splits + vendored YAIB model architectures,
but owns the entry point, checkpoint format, and metadata layer.

Runs in the critical-mm-train conda env (torch + pytorch-lightning;
no gin, no recipys, no icu_benchmarks).

Flow:
    1. Load CM dyn / sta / outc parquets for (task, dataset).
    2. Load the locked split parquet for (cv_rep, fold).
    3. Apply CM-native preprocessor (preprocess.py) per split.
    4. Dispatch on the registered model class:
         - DL wrapper (needs_training=True): instantiate from
           model_defaults_for(task)[model] + cfg.extra_hyperparams,
           wrap in pytorch_lightning.Trainer with seed_everything(cfg.seed).
         - ML wrapper (needs_fit=True): instantiate with cfg.extra_hyperparams
           (sklearn-style filter), call model.fit(train_dataset, val_dataset),
           compute test metric with sklearn.
    5. Save checkpoint + metadata.json (data git SHA, splits fingerprint,
       test metrics) to data/checkpoints/<task>/<ds>/<model>/seed_<n>/.

Public:
    train_one(cfg: TrainConfig) -> dict
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

from critical_mm.training.config import REPO, TrainConfig, resolve_trainer_overrides

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from critical_mm.fusion.config import FusionConfig
    from critical_mm.models._data.loader import PredictionDataset


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _splits_fingerprint(cfg: TrainConfig) -> dict[str, Any]:
    manifest_path = cfg.splits_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"locked-split manifest missing: {manifest_path}. Run scripts/lock_splits.py first."
        )
    manifest = json.loads(manifest_path.read_text())
    h = hashlib.sha256()
    for fold in manifest["folds"]:
        h.update(fold["content_sha256"].encode())
    return {
        "manifest_path": str(manifest_path.relative_to(REPO)),
        "data_git_sha_when_locked": manifest["data_git_sha"],
        "outc_content_sha256": manifest["outc_content_sha256"],
        "folds_combined_sha256": h.hexdigest(),
        "n_stays_total": manifest["n_stays_total"],
        "n_folds": len(manifest["folds"]),
    }


def _load_split_parquet(cfg: TrainConfig) -> pd.DataFrame:
    """Load locked split as pandas (CM-native trainer uses pandas-shaped frames)."""
    import pandas as pd

    p = cfg.splits_dir / f"cv_rep_{cfg.repetition_index}_fold_{cfg.fold_index}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"locked split missing: {p}. Run scripts/lock_splits.py first.")
    return pd.read_parquet(p)


def _load_data_parquets(cfg: TrainConfig) -> dict[str, pd.DataFrame]:
    """Load CM sta / dyn / outc parquets as pandas DataFrames."""
    import pandas as pd

    out: dict[str, pd.DataFrame] = {}
    for seg, fname in [
        ("STATIC", "sta.parquet"),
        ("DYNAMIC", "dyn.parquet"),
        ("OUTCOME", "outc.parquet"),
    ]:
        path = cfg.data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"missing {seg} parquet at {path}")
        out[seg] = pd.read_parquet(path)
    return out


def hash_stay_id(s: object) -> int:
    """Deterministic int32 hash of a stay_id (suffix-int where possible, else hash).

    Shared by the fusion block builders so modality frames join to the
    preamble's int-hashed stay_id identically.
    """
    text = str(s)
    if "_" in text:
        text = text.rsplit("_", 1)[-1]
    try:
        return int(text) % (2**31)
    except ValueError:
        return abs(hash(text)) % (2**31)


def _apply_split(
    data: dict[str, pd.DataFrame],
    split_df: pd.DataFrame,
    id_col: str = "stay_id",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Partition each segment by the loaded split into train / val / test."""
    sample = next(iter(data.values()))
    stay_dtype = sample[id_col].dtype
    split_pd = split_df.astype({id_col: stay_dtype})

    result: dict[str, dict[str, Any]] = {}
    for fold in ("train", "val", "test"):
        fold_ids = split_pd.loc[split_pd["split"] == fold, [id_col]]
        result[fold] = {
            seg: data[seg].merge(fold_ids, on=id_col, how="right", sort=True) for seg in data
        }
    return result


@dataclass
class _TrainContext:
    """State produced by the shared preamble; consumed by the DL/ML branches."""

    t0: float
    train_dataset: PredictionDataset
    val_dataset: PredictionDataset
    test_dataset: PredictionDataset
    runmode: Any
    vars: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class _BasePreamble:
    """Structured `cm_preprocess` prefix of the preamble, reusable across fusion
    variants. Carries every value the post-prefix body reads from the prefix."""

    t0: float
    preprocessed: dict[Any, dict[Any, Any]]
    vars: dict[str, Any]
    runmode: Any
    splits_fp: dict[str, Any]


def _build_base_preamble(cfg: TrainConfig, generate_features: bool = False) -> _BasePreamble:
    """Structured prefix of the preamble (no RNG after cm_preprocess) — reusable
    across fusion variants. Identical statements to the inline prefix it replaces,
    moved verbatim from the head of `_train_one_preamble`.
    """
    from pytorch_lightning import seed_everything

    from critical_mm.models._data.constants import DataSegment as Segment
    from critical_mm.models._data.constants import DataSplit as Split
    from critical_mm.models._data.loader import PredictionDataset  # noqa: F401
    from critical_mm.models._runmode import RunMode
    from critical_mm.training.preprocess import preprocess as cm_preprocess

    t0 = time.perf_counter()

    seed_everything(cfg.seed, workers=True)

    splits_fp = _splits_fingerprint(cfg)
    split_df = _load_split_parquet(cfg)
    data = _load_data_parquets(cfg)

    segment_data = {
        Segment.static: data["STATIC"],
        Segment.dynamic: data["DYNAMIC"],
        Segment.outcome: data["OUTCOME"],
    }

    partitioned = _apply_split(segment_data, split_df)
    import numpy as np

    _drop_cols = {"patient_id"}
    for fold in partitioned:
        for seg in (Segment.static, Segment.dynamic, Segment.outcome):
            df = partitioned[fold][seg]
            df = df.drop(columns=[c for c in _drop_cols if c in df.columns], errors="ignore")
            if df["stay_id"].dtype == object:
                df = df.assign(stay_id=df["stay_id"].map(hash_stay_id).astype(np.int64))
            partitioned[fold][seg] = df

    yaib_partitioned: dict[Any, dict[Any, Any]] = {
        Split.train: partitioned["train"],
        Split.val: partitioned["val"],
        Split.test: partitioned["test"],
    }

    runmode = RunMode.classification if cfg.is_classification else RunMode.regression

    sta_df = data["STATIC"]
    dyn_df = data["DYNAMIC"]
    outc_df = data["OUTCOME"]
    label_col = "label_value" if "label_value" in outc_df.columns else "label"
    seq_col = "time" if "time" in dyn_df.columns else "hour"
    _id_cols = {"stay_id", "patient_id", "time", "hour", "_label"}
    sta_cols = [c for c in sta_df.columns if c not in _id_cols]
    dyn_cols = [c for c in dyn_df.columns if c not in _id_cols]
    outc_cols = [c for c in outc_df.columns if c not in _id_cols]
    vars = {
        "GROUP": "stay_id",
        "LABEL": label_col,
        "SEQUENCE": seq_col,
        Segment.dynamic: dyn_cols,
        Segment.static: sta_cols,
        Segment.outcome: outc_cols,
    }
    mask_output_root = cfg.data_dir / "masks"
    preprocessed = cm_preprocess(
        yaib_partitioned,
        vars,
        mask_output_root=mask_output_root,
        generate_features=generate_features,
    )

    return _BasePreamble(
        t0=t0,
        preprocessed=preprocessed,
        vars=vars,
        runmode=runmode,
        splits_fp=splits_fp,
    )


def _train_one_preamble(
    cfg: TrainConfig,
    generate_features: bool = False,
    ram_cache: bool = True,
    write_metadata: bool = True,
    build_splits: tuple[str, ...] = ("train", "val", "test"),
    fusion: object | None = None,
    dynamic_pad: bool = False,
    base: _BasePreamble | None = None,
) -> _TrainContext:
    """Shared steps for both DL and ML training paths.

    Loads parquets, applies locked split, runs CM preprocess, builds the three
    PredictionDataset objects, and seeds the metadata dict with the data
    provenance block. Returns a _TrainContext consumed by the DL/ML branches.

    Args:
        cfg: training config (task, dataset, model, seed, paths).
        generate_features: forward to `preprocess()`; ML path passes True so
            each stay's last row carries per-stay running aggregates rather
            than a single time-step observation.
    """
    from critical_mm.models._data.constants import DataSegment as Segment
    from critical_mm.models._data.constants import DataSplit as Split
    from critical_mm.models._data.loader import PredictionDataset

    if base is None:
        base = _build_base_preamble(cfg, generate_features=generate_features)
    t0, preprocessed, vars, runmode, splits_fp = (
        base.t0,
        base.preprocessed,
        base.vars,
        base.runmode,
        base.splits_fp,
    )

    metadata_fusion = None
    if fusion is not None and getattr(fusion, "rung", "structured") != "structured":
        from critical_mm.fusion.loader import build_blocks_for_preamble
        from critical_mm.fusion.strategy import augment_preprocessed

        fusion_cfg = cast("FusionConfig", fusion)
        train_ids = set(
            preprocessed[Split.train][Segment.features][vars["GROUP"]].unique().tolist()
        )
        icd_block, notes_block = build_blocks_for_preamble(
            cfg=cfg,
            fusion=fusion_cfg,
            vars=vars,
            preprocessed=preprocessed,
            train_stay_ids=train_ids,
        )
        preprocessed = augment_preprocessed(
            preprocessed,
            fusion=fusion_cfg,
            vars=vars,
            icd_block=icd_block,
            notes_block=notes_block,
            strategy=getattr(fusion_cfg, "icd_strategy", "feature_augmentation"),
        )
        metadata_fusion = fusion_cfg.to_metadata()

    _cfg_dict = asdict(cfg)
    _cfg_dict.pop("data_root", None)
    metadata: dict[str, Any] = {
        "config": _cfg_dict,
        "data_git_sha_at_train": _git_sha(),
        "splits_fingerprint": splits_fp,
        "runmode": runmode.name,
    }

    metadata["duration_load_s"] = round(time.perf_counter() - t0, 2)
    metadata["splits"] = {
        fold: {seg: len(df) for seg, df in segs.items()} for fold, segs in preprocessed.items()
    }
    if metadata_fusion is not None:
        metadata["fusion"] = metadata_fusion
    if write_metadata:
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (cfg.checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    train_dataset = PredictionDataset(
        preprocessed,
        vars=vars,
        split=Split.train,
        ram_cache=ram_cache and "train" in build_splits,
        dynamic_pad=dynamic_pad,
    )
    val_dataset = PredictionDataset(
        preprocessed,
        vars=vars,
        split=Split.val,
        ram_cache=ram_cache and "val" in build_splits,
        dynamic_pad=dynamic_pad,
    )
    test_dataset = PredictionDataset(
        preprocessed,
        vars=vars,
        split=Split.test,
        ram_cache=ram_cache and "test" in build_splits,
        dynamic_pad=dynamic_pad,
    )

    return _TrainContext(
        t0=t0,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        runmode=runmode,
        vars=vars,
        metadata=metadata,
    )


def _train_one_dl(
    cfg: TrainConfig,
    ctx: _TrainContext,
    model_class: type,
    checkpoint_dir: Path | None = None,
    dynamic_pad: bool = False,
) -> dict[str, Any]:
    """DL training path: PyTorch Lightning Trainer fit + test.

    ``checkpoint_dir`` overrides ``cfg.checkpoint_dir`` for ALL write/log
    targets (the fusion sidecar grid passes its isolated _fusion ckpt). When
    None (the locked train_one path), it falls back to cfg.checkpoint_dir so
    behavior is byte-identical.
    """
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.optim import Adam
    from torch.utils.data import DataLoader

    from critical_mm.training.config import model_defaults_for

    collate = None
    if dynamic_pad:
        from critical_mm.models._data.loader import dynamic_pad_collate

        collate = dynamic_pad_collate

    ckpt = checkpoint_dir if checkpoint_dir is not None else cfg.checkpoint_dir
    metadata = ctx.metadata
    train_dataset, val_dataset, test_dataset = (
        ctx.train_dataset,
        ctx.val_dataset,
        ctx.test_dataset,
    )

    use_cuda = (not cfg.cpu) and torch.cuda.is_available()
    if use_cuda:
        batch_size = min(512, len(train_dataset), len(val_dataset))
        num_workers = min(8, len(train_dataset) // batch_size)
        pin_memory = True
        persistent_workers = num_workers > 0
    else:
        batch_size = min(64, len(train_dataset), len(val_dataset))
        num_workers = 0
        pin_memory = False
        persistent_workers = False
    extra_hp, epochs, precision = resolve_trainer_overrides(cfg.extra_hyperparams, use_cuda)
    batch_size_override = extra_hp.pop("batch_size", None)
    if batch_size_override is not None:
        bs = int(batch_size_override)  # type: ignore[call-overload]
        batch_size = min(bs, len(train_dataset), len(val_dataset))
        print(f" [batch_size override] using batch_size={batch_size}")
    patience = 10

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate,
    )

    data_shape = next(iter(train_loader))[0].shape

    hparams: dict[str, object] = dict(model_defaults_for(cfg.task)[cfg.model])
    hparams.update(extra_hp)
    model = model_class(
        input_size=data_shape,
        optimizer=Adam,
        epochs=epochs,
        run_mode=ctx.runmode,
        **hparams,
    )
    model.set_weight("balanced" if cfg.is_classification else None, train_dataset)
    model.set_trained_columns(train_dataset.get_feature_names())  # type: ignore[no-untyped-call]

    tb_logger = TensorBoardLogger(str(ckpt))
    callbacks = [
        EarlyStopping(monitor="val/loss", patience=patience, strict=False),
        ModelCheckpoint(
            dirpath=str(ckpt),
            filename="model",
            save_top_k=1,
            save_last=True,
        ),
    ]
    trainer = Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        accelerator="cuda" if use_cuda else "cpu",
        devices=1,
        precision=precision,
        deterministic="warn" if use_cuda else True,
        enable_progress_bar=False,
        logger=tb_logger,
        num_sanity_val_steps=0,
    )
    metadata["compute"] = {
        "device": "cuda" if use_cuda else "cpu",
        "device_name": (torch.cuda.get_device_name(0) if use_cuda else "cpu"),
        "precision": precision,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    metadata["model_kind"] = "dl"

    fit_start = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    metadata["duration_fit_s"] = round(time.perf_counter() - fit_start, 2)

    test_metrics_list = trainer.test(model, dataloaders=test_loader, verbose=False)
    metadata["test_metrics"] = test_metrics_list[0] if test_metrics_list else {}
    metadata["duration_total_s"] = round(time.perf_counter() - ctx.t0, 2)

    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    return {"status": "ok", "metadata": metadata}


_CUML_GPU_MODELS = frozenset(
    {"LogisticRegression", "RFClassifier", "ElasticNet", "LinearRegression"}
)


def _ml_cache_root(cfg: TrainConfig) -> Path:
    """Root of the ML array cache, under cfg.data_root so an isolated re-lock
    does not write into the locked tree's _ml_array_cache."""
    return cfg.data_root / "data" / "checkpoints" / "_ml_array_cache"


def _evict_other_ml_caches(keep_dir: Path, cache_root: Path) -> None:
    """Keep only the current cohort's cache (arrays are multi-GB; the box is
    disk-tight). The grid processes a (task, dataset)'s cells consecutively,
    so a cohort is never revisited within a run."""
    if not cache_root.exists():
        return
    for task_dir in cache_root.iterdir():
        if not task_dir.is_dir():
            continue
        for ds_dir in task_dir.iterdir():
            if ds_dir.is_dir() and ds_dir.resolve() != keep_dir.resolve():
                shutil.rmtree(ds_dir, ignore_errors=True)


def _ml_array_cache(cfg: TrainConfig) -> tuple[Any, dict[str, Any]]:
    """Return (cache_dir, meta) for cfg's (task, dataset), materializing the
    arrays on a cache miss. cache_dir holds arrays.npz (Xtr/ytr/Xva/yva/Xte/
    yte) + meta.json, reused across all model x seed cells of the cohort."""
    import numpy as np

    splits_fp = _splits_fingerprint(cfg)
    fp_key = splits_fp["folds_combined_sha256"]
    cache_dir = _ml_cache_root(cfg) / cfg.task / cfg.dataset
    meta_path = cache_dir / "meta.json"
    npz_path = cache_dir / "arrays.npz"

    if meta_path.exists() and npz_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fp_key") == fp_key:
            return cache_dir, meta

    _evict_other_ml_caches(cache_dir, _ml_cache_root(cfg))
    ctx = _train_one_preamble(cfg, generate_features=True, ram_cache=False)
    Xtr, ytr = ctx.train_dataset.get_data_and_labels()
    Xva, yva = ctx.val_dataset.get_data_and_labels()
    Xte, yte = ctx.test_dataset.get_data_and_labels()
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)
    meta = {
        "fp_key": fp_key,
        "splits_fingerprint": splits_fp,
        "data_git_sha_at_train": ctx.metadata["data_git_sha_at_train"],
        "splits": ctx.metadata["splits"],
        "duration_load_s": ctx.metadata["duration_load_s"],
        "runmode": ctx.metadata["runmode"],
        "feature_names": list(ctx.train_dataset.get_feature_names()),  # type: ignore[no-untyped-call]
    }
    meta_path.write_text(json.dumps(meta, default=str))
    return cache_dir, meta


def _fit_ml_inprocess(
    cfg: TrainConfig,
    *,
    Xtr: Any,
    ytr: Any,
    Xva: Any,
    yva: Any,
    Xte: Any,
    yte: Any,
    feature_names: list[str],
    metadata: dict[str, Any],
    runmode: Any,
    model_class: type,
    t0: float,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """In-process LGBM/CPU fit on already-materialized arrays.

    Single source of truth for the non-GPU ML path: builds the model,
    set_trained_columns, the is_lgbm/device_type/backend logic, fit_model,
    predict, the AUC/MAE metric, save_model, and the metadata write. Writes to
    ``checkpoint_dir`` when provided (the fusion sidecar's isolated dir), else
    ``cfg.checkpoint_dir`` (the locked train_one path — byte-identical).

    ``metadata`` is the partially-built dict the caller assembled (config /
    runmode / splits / etc.); this helper fills in fit/metric/compute/duration
    keys and writes it out. ``t0`` is the caller's wall-clock start (used for
    duration_total_s — the locked path measures it from the ML entry t0).
    """
    import lightgbm
    import numpy as np
    import sklearn
    import sklearn.metrics as skm

    ckpt = checkpoint_dir if checkpoint_dir is not None else cfg.checkpoint_dir
    ckpt.mkdir(parents=True, exist_ok=True)

    model = model_class(run_mode=runmode, **cfg.extra_hyperparams)
    model.set_trained_columns(feature_names)

    gpu = os.environ.get("CMM_ML_GPU") == "1"
    is_lgbm = cfg.model in ("LGBMClassifier", "LGBMRegressor")
    device, device_name = "cpu", "cpu"
    if gpu and is_lgbm:
        model.model.set_params(device_type="cuda")
        device = "cuda"
        try:
            import torch

            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "GPU"
    backend = f"lightgbm-{lightgbm.__version__}" if is_lgbm else f"sklearn-{sklearn.__version__}"

    fit_start = time.perf_counter()
    model.fit_model(Xtr, ytr, Xva, yva)
    metadata["duration_fit_s"] = round(time.perf_counter() - fit_start, 2)

    test_pred = model.predict(Xte)
    if cfg.is_classification:
        if hasattr(model.model, "predict_proba"):
            proba = model.model.predict_proba(Xte)
            test_pred_pos = (
                proba[:, 1]
                if proba.ndim == 2 and proba.shape[1] == 2
                else np.asarray(test_pred, dtype=float)
            )
        else:
            test_pred_pos = np.asarray(test_pred, dtype=float)
        test_metrics: dict[str, float] = {"test/AUC": float(skm.roc_auc_score(yte, test_pred_pos))}
    else:
        test_metrics = {"test/MAE": float(skm.mean_absolute_error(yte, test_pred))}

    metadata["test_metrics"] = test_metrics
    metadata["compute"] = {"device": device, "device_name": device_name, "backend": backend}
    metadata["duration_total_s"] = round(time.perf_counter() - t0, 2)
    model.save_model(ckpt, "model", ".joblib")
    (ckpt / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return {"status": "ok", "metadata": metadata}


def _train_one_ml(cfg: TrainConfig, model_class: type) -> dict[str, Any]:
    """ML training path (cached). Materializes the cohort's (X, y) arrays once
    via ``_ml_array_cache``, then fits this cell's model:

      * GPU + cuML model: subprocess in the cuml-gpu env reads arrays.npz.
      * otherwise (LGBM, or CPU mode): in-process fit on the loaded arrays;
        LGBM gets device_type='cuda' when CMM_ML_GPU=1.

    Writes the same metadata schema as the DL path so the aggregator ingests
    ML and DL cells uniformly into one CM_REFERENCE table.
    """
    import numpy as np
    from pytorch_lightning import seed_everything

    from critical_mm.models._runmode import RunMode

    t0 = time.perf_counter()
    seed_everything(cfg.seed, workers=True)

    cache_dir, meta = _ml_array_cache(cfg)
    runmode = RunMode.classification if cfg.is_classification else RunMode.regression
    runmode_str = "classification" if cfg.is_classification else "regression"

    metadata: dict[str, Any] = {
        "config": asdict(cfg),
        "data_git_sha_at_train": meta["data_git_sha_at_train"],
        "splits_fingerprint": meta["splits_fingerprint"],
        "runmode": meta["runmode"],
        "duration_load_s": meta["duration_load_s"],
        "splits": meta["splits"],
        "model_kind": "ml",
    }
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get("CMM_ML_GPU") == "1"

    if gpu and cfg.model in _CUML_GPU_MODELS:
        metrics_out = cfg.checkpoint_dir / "_cuml_metrics.json"
        model_out = cfg.checkpoint_dir / "model.joblib"
        cuml_py = os.environ.get(
            "CMM_CUML_PYTHON",
            "python",
        )
        runner = REPO / "scripts" / "cuml_runner.py"
        wall_start = time.perf_counter()
        subprocess.run(
            [
                cuml_py,
                str(runner),
                "--npz",
                str(cache_dir / "arrays.npz"),
                "--model",
                cfg.model,
                "--runmode",
                runmode_str,
                "--hparams",
                json.dumps(cfg.extra_hyperparams),
                "--metrics-out",
                str(metrics_out),
                "--model-out",
                str(model_out),
            ],
            check=True,
        )
        res = json.loads(metrics_out.read_text())
        metrics_out.unlink(missing_ok=True)
        metadata["test_metrics"] = res["test_metrics"]
        metadata["duration_fit_s"] = res["duration_fit_s"]
        metadata["duration_wall_s"] = round(time.perf_counter() - wall_start, 2)
        metadata["compute"] = res["compute"]
        metadata["duration_total_s"] = round(time.perf_counter() - t0, 2)
        (cfg.checkpoint_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )
        return {"status": "ok", "metadata": metadata}

    data = np.load(cache_dir / "arrays.npz")
    return _fit_ml_inprocess(
        cfg,
        Xtr=data["Xtr"],
        ytr=data["ytr"],
        Xva=data["Xva"],
        yva=data["yva"],
        Xte=data["Xte"],
        yte=data["yte"],
        feature_names=meta["feature_names"],
        metadata=metadata,
        runmode=runmode,
        model_class=model_class,
        t0=t0,
        checkpoint_dir=None,
    )


def train_one(cfg: TrainConfig) -> dict[str, Any]:
    """Train one (task, dataset, model, seed) cell with locked splits.

    Dispatches to ``_train_one_dl`` or ``_train_one_ml`` based on the
    registered model class's ``needs_training`` / ``needs_fit`` flags.
    Before dispatching, checks that the dataset declares every capability
    the task's ``required_dataset_capabilities`` lists (); raises
    ValueError with a precise message when an external contributor tries
    to run a (task, dataset) combo that's structurally incompatible.
    """
    from critical_mm.registry import discover_datasets, discover_models, discover_tasks

    _models = discover_models()
    if cfg.model not in _models:
        raise ValueError(f"unknown model {cfg.model!r}; registered names: {sorted(_models)}")
    model_class = _models[cfg.model]

    _tasks = discover_tasks()
    _datasets = discover_datasets()
    task_class = _tasks.get(cfg.task)
    dataset_class = _datasets.get(cfg.dataset)
    if task_class is not None and dataset_class is not None:
        required = task_class().required_dataset_capabilities(cfg.dataset)
        have: frozenset[str] = getattr(dataset_class, "CAPABILITIES", frozenset())
        missing = required - have
        if missing:
            raise ValueError(
                f"task {cfg.task!r} requires dataset capabilities {sorted(missing)} "
                f"that {cfg.dataset!r} does not declare. "
                f"Dataset CAPABILITIES: {sorted(have)}. "
                f"Add the capability to the dataset class or remove the requirement "
                f"from the task."
            )

    needs_fit = getattr(model_class, "needs_fit", False)
    needs_training = getattr(model_class, "needs_training", False)

    if needs_fit and not needs_training:
        return _train_one_ml(cfg, model_class)
    if needs_training and not needs_fit:
        ctx = _train_one_preamble(cfg, generate_features=False)
        return _train_one_dl(cfg, ctx, model_class)
    raise ValueError(
        f"Model {cfg.model!r} ({model_class.__module__}.{model_class.__name__}) is neither "
        f"pure-ML (needs_fit=True, needs_training=False) nor pure-DL (needs_training=True, "
        f"needs_fit=False). Hybrid wrappers are not supported."
    )


__all__ = ["train_one"]
