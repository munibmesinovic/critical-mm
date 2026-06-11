"""Resolve per-cell modality artifacts and build the fusion blocks.

Reads the gitignored on-disk modality surfaces produced by increments 1+2:
  data/processed/_modalities/<ds>/diagnoses/aligned_<task>.parquet
  data/processed/_modalities/<ds>/notes/aligned_<task>.parquet
  data/processed/_modalities/<ds>/notes_emb/<encoder>@<rev>/part_*.parquet
"""

from __future__ import annotations

from typing import Any

import polars as pl

from critical_mm.fusion.blocks import build_icd_block, build_notes_block
from critical_mm.fusion.config import STAY_LEVEL_CUTOFF_H, FusionConfig
from critical_mm.training.config import REPO

MODALITIES_ROOT = REPO / "data" / "processed" / "_modalities"
PER_HOUR_TASKS = frozenset({"aki", "sepsis", "los"})


def _load_emb(dataset: str, encoder: str, note_ids: set[str] | None = None) -> pl.DataFrame:
    root = MODALITIES_ROOT / dataset / "notes_emb" / encoder
    parts = sorted(root.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no note embeddings under {root}")
    lf = pl.scan_parquet(parts)
    cols = lf.collect_schema().names()
    emb_col = next(
        (c for c in ("embedding", "emb") if c in cols),
        next(c for c in cols if c not in ("note_id", "n_tokens")),
    )
    lf = lf.select("note_id", pl.col(emb_col).alias("emb"))
    if note_ids is not None:
        lf = lf.filter(pl.col("note_id").is_in(list(note_ids)))
    return lf.collect()


def build_blocks_for_cell(
    *,
    task: str,
    dataset: str,
    fusion: FusionConfig,
    train_stay_ids: set[int],
    min_prevalence: int | None = None,
    pca_dim: int | None = None,
    dyn_grid: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Lower-level builder used by tests and the preamble adaptor below."""
    icd_block: pl.DataFrame | None = None
    notes_block: pl.DataFrame | None = None
    if fusion.uses_icd:
        aligned = pl.read_parquet(
            MODALITIES_ROOT / dataset / "diagnoses" / f"aligned_{task}.parquet"
        )
        icd_block, _ = build_icd_block(
            aligned.select(
                "stay_id", "code", "code_system", "origin", "delta_h", "prior_visit_idx"
            ),
            train_stay_ids=train_stay_ids,
            group=fusion.icd_group,
            min_prevalence=min_prevalence or fusion.icd_min_prevalence,
            top_k=fusion.icd_top_k,
            representation=fusion.icd_repr,
            pca_dim=fusion.icd_pca_dim,
        )
    if fusion.uses_notes:
        if dataset not in fusion.encoders:
            raise ValueError(
                f"notes rung requested for dataset {dataset!r} but no note encoder is "
                f"configured (notes modality exists only for miiv/eicu/omix)"
            )
        aligned = pl.read_parquet(MODALITIES_ROOT / dataset / "notes" / f"aligned_{task}.parquet")
        note_ids = set(aligned["note_id"].to_list())
        emb = _load_emb(dataset, fusion.encoders[dataset], note_ids=note_ids)
        per_hour = task in PER_HOUR_TASKS
        notes_block, _ = build_notes_block(
            aligned,
            emb,
            train_stay_ids=train_stay_ids,
            per_hour=per_hour,
            dyn_grid=dyn_grid,
            cutoff_h=None if per_hour else STAY_LEVEL_CUTOFF_H.get(task, 24.0),
            pca_dim=pca_dim if pca_dim is not None else fusion.pca_dim,
            half_life=fusion.notes_half_life,
        )
    return icd_block, notes_block


def build_blocks_for_preamble(
    *,
    cfg: Any,
    fusion: FusionConfig,
    vars: dict[str, Any],
    preprocessed: dict[Any, dict[Any, Any]],
    train_stay_ids: set[int],
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Adaptor for the train.py hook: derive the per-hour dyn_grid from the
    preprocessed features (stay_id + SEQUENCE) so the notes block aligns exactly."""
    import pandas as pd

    from critical_mm.models._data.constants import DataSegment as Segment
    from critical_mm.models._data.constants import DataSplit as Split

    group, seq = vars["GROUP"], vars["SEQUENCE"]
    dyn_grid: pl.DataFrame | None = None
    if fusion.uses_notes and cfg.task in PER_HOUR_TASKS:
        frames = [
            preprocessed[s][Segment.features][[group, seq]]
            for s in (Split.train, Split.val, Split.test)
        ]
        grid = pd.concat(frames, ignore_index=True).drop_duplicates()
        dyn_grid = pl.from_pandas(grid.rename(columns={group: "stay_id", seq: "hour"}))
    return build_blocks_for_cell(
        task=cfg.task,
        dataset=cfg.dataset,
        fusion=fusion,
        train_stay_ids=train_stay_ids,
        dyn_grid=dyn_grid,
    )
