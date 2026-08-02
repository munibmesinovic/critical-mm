"""Fusion strategies. Strategy #1 = feature-block augmentation on the fixed backbone.

The shared interface is the pre-built modality blocks (polars frames keyed on the
hashed stay_id; per-hour blocks also carry ``hour``). A strategy decides how to
consume them. ``FeatureAugmentationFusion`` left-joins them onto the preprocessed
``Segment.features`` frame and re-sorts ``[GROUP, SEQUENCE]`` so the DL loader's
positional pairing stays correct. Future Late/EndToEnd strategies consume the same
blocks differently -- no data re-plumbing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import polars as pl

from critical_mm.fusion.config import FusionConfig
from critical_mm.models._data.constants import DataSegment as Segment
from critical_mm.registry import register_fusion

class FusionStrategy(ABC):
    """Contract: given preprocessed splits + modality blocks, return augmented splits."""

    @abstractmethod
    def augment(
        self,
        preprocessed: dict[Any, dict[Any, pd.DataFrame]],
        *,
        fusion: FusionConfig,
        vars: dict[str, Any],
        blocks: dict[str, pl.DataFrame | None] | None = None,
        icd_block: pl.DataFrame | None = None,
        notes_block: pl.DataFrame | None = None,
    ) -> dict[Any, dict[Any, pd.DataFrame]]: ...

def _value_cols(block: pl.DataFrame, keys: list[str]) -> list[str]:
    return [c for c in block.columns if c not in keys]

def _coerce_blocks(
    blocks: dict[str, pl.DataFrame | None] | None,
    icd_block: pl.DataFrame | None,
    notes_block: pl.DataFrame | None,
) -> dict[str, pl.DataFrame | None]:
    """Accept either the new ``blocks`` dict or the legacy positional pair.

    When the legacy positional ``icd_block``/``notes_block`` args are supplied
    (``blocks is None``), this reconstructs ``{"icd": icd_block, "notes": notes_block}``
    so the existing locked-cell path is byte-identical to the dict path.
    """
    if blocks is not None:
        return blocks
    return {"icd": icd_block, "notes": notes_block}

@register_fusion("feature_augmentation")
class FeatureAugmentationFusion(FusionStrategy):
    def augment(
        self,
        preprocessed: dict[Any, dict[Any, pd.DataFrame]],
        *,
        fusion: FusionConfig,
        vars: dict[str, Any],
        blocks: dict[str, pl.DataFrame | None] | None = None,
        icd_block: pl.DataFrame | None = None,
        notes_block: pl.DataFrame | None = None,
    ) -> dict[Any, dict[Any, pd.DataFrame]]:
        if fusion.rung == "structured":
            return preprocessed
        blocks = _coerce_blocks(blocks, icd_block, notes_block)
        icd_block = blocks.get("icd")
        notes_block = blocks.get("notes")
        group, seq = vars["GROUP"], vars["SEQUENCE"]
        out: dict[Any, dict[Any, pd.DataFrame]] = {}
        for split, segs in preprocessed.items():
            feat = segs[Segment.features]
            if fusion.uses_icd and icd_block is not None:
                feat = _merge_block(feat, icd_block, on=[group], present="icd_present")
            if fusion.uses_notes and notes_block is not None:
                if "hour" in notes_block.columns:
                    note_keys = [group, seq]
                    right = notes_block.rename({"hour": seq})
                else:
                    note_keys = [group]
                    right = notes_block
                feat = _merge_block(feat, right, on=note_keys, present="notes_present")
            if fusion.uses_treatments and blocks.get("treatments") is not None:
                tblock = blocks["treatments"]
                if "hour" in tblock.columns:
                    tkeys = [group, seq]
                    tright = tblock.rename({"hour": seq})
                else:
                    tkeys = [group]
                    tright = tblock
                feat = _merge_block(feat, tright, on=tkeys, present="treatments_present")
            feat = feat.sort_values([group, seq], kind="stable").reset_index(drop=True)
            out[split] = {**segs, Segment.features: feat}
        return out

@register_fusion("icd_only")
class IcdOnlyFusion(FusionStrategy):
    """Replace the dynamic feature columns with the ICD block (LGBM-only baseline).

    Keeps GROUP + SEQUENCE so the per-timestep get_data_and_labels lexsort/alignment
    is preserved; drops every other (dynamic/static) predictor column.
    """

    def augment(
        self,
        preprocessed: dict[Any, dict[Any, pd.DataFrame]],
        *,
        fusion: FusionConfig,
        vars: dict[str, Any],
        blocks: dict[str, pl.DataFrame | None] | None = None,
        icd_block: pl.DataFrame | None = None,
        notes_block: pl.DataFrame | None = None,
    ) -> dict[Any, dict[Any, pd.DataFrame]]:
        blocks = _coerce_blocks(blocks, icd_block, notes_block)
        icd_block = blocks.get("icd")
        if icd_block is None:
            return preprocessed
        group, seq = vars["GROUP"], vars["SEQUENCE"]
        val_cols = [c for c in icd_block.columns if c != group]
        bpd = icd_block.to_pandas()
        out = {}
        for split, segs in preprocessed.items():
            feat = segs[Segment.features]
            keep = feat[[group, seq]].copy()
            keep[group] = keep[group].astype(feat[group].dtype)
            bpd[group] = bpd[group].astype(feat[group].dtype)
            merged = keep.merge(bpd, how="left", on=[group], validate="m:1")
            for c in val_cols:
                merged[c] = merged[c].fillna(0.0).astype("float32")
            merged = merged.sort_values([group, seq], kind="stable").reset_index(drop=True)
            out[split] = {**segs, Segment.features: merged}
        return out

def _merge_block(
    feat: pd.DataFrame, block: pl.DataFrame, *, on: list[str], present: str
) -> pd.DataFrame:
    """Left-merge a modality block; zero-fill value cols; set the present bit to 0 for misses."""
    keys = list(on)
    val_cols = _value_cols(block, keys)
    bpd = block.to_pandas()
    for k in keys:
        bpd[k] = bpd[k].astype(feat[k].dtype)
    merged = feat.merge(bpd, how="left", on=keys, validate="m:1")
    for c in val_cols:
        merged[c] = merged[c].fillna(0.0).astype("float32")
    if present in merged.columns:
        merged[present] = merged[present].fillna(0.0).astype("float32")
    return merged

def augment_preprocessed(
    preprocessed: dict[Any, dict[Any, pd.DataFrame]],
    *,
    fusion: FusionConfig,
    vars: dict[str, Any],
    blocks: dict[str, pl.DataFrame | None] | None = None,
    icd_block: pl.DataFrame | None = None,
    notes_block: pl.DataFrame | None = None,
    strategy: str = "feature_augmentation",
) -> dict[Any, dict[Any, pd.DataFrame]]:
    """Resolve the named strategy and apply it. Entry called from the training hook."""
    from critical_mm.registry import get_fusion

    blocks = _coerce_blocks(blocks, icd_block, notes_block)
    strat: FusionStrategy = get_fusion(strategy)()
    return strat.augment(preprocessed, fusion=fusion, vars=vars, blocks=blocks)

