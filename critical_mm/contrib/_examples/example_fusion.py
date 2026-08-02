"""Worked example: a minimal registered multimodal fusion strategy.

A fusion strategy decides how the pre-built modality blocks (ICD / notes) are
combined with the structured feature stream before the backbone trains. Copy
this file to ``critical_mm/contrib/my_fusion.py`` (OUT of ``_examples/``), then
select it by name from the fusion grid::

    python scripts/fusion_grid.py --strategy example_fusion --rung icd \
        --tasks mortality24 --datasets miiv --models LGBMClassifier --seeds 42

Contract (see ``critical_mm/fusion/strategy.py`` for the built-in strategies).
``augment`` receives the preprocessed per-split feature frames plus the modality
``blocks`` dict -- ``{"icd": ..., "notes": ..., "treatments": ...}`` keyed by
modality name, each value a polars frame keyed on the hashed ``GROUP`` id (per-hour
blocks also carry the ``SEQUENCE`` column) or ``None`` when absent -- and returns
augmented frames. The legacy positional ``icd_block``/``notes_block`` kwargs are
still accepted for back-compat; ``_coerce_blocks`` folds them into the dict. The
example left-joins the ICD block onto the dynamic features and re-sorts on
``[GROUP, SEQUENCE]`` so the loader's positional pairing stays correct.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl

from critical_mm.fusion.config import FusionConfig
from critical_mm.fusion.strategy import FusionStrategy, _coerce_blocks
from critical_mm.models._data.constants import DataSegment as Segment
from critical_mm.registry import register_fusion

@register_fusion("example_fusion")
class ExampleConcatFusion(FusionStrategy):
    """Concatenate the ICD block onto the structured features (zero-filled)."""

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
        block = icd_block.to_pandas()
        value_cols = [c for c in block.columns if c != group]
        out: dict[Any, dict[Any, pd.DataFrame]] = {}
        for split, segs in preprocessed.items():
            feat = segs[Segment.features]
            block[group] = block[group].astype(feat[group].dtype)
            merged = feat.merge(block, how="left", on=[group], validate="m:1")
            for col in value_cols:
                merged[col] = merged[col].fillna(0.0).astype("float32")
            merged = merged.sort_values([group, seq], kind="stable").reset_index(drop=True)
            out[split] = {**segs, Segment.features: merged}
        return out
