"""Overridable Tier-2 grouping for diagnosis codes.

rep="ccsr" (default): AHRQ CCSR v2026.1 (separate module) — ICD-10-CM codes map
to a clinical category; ICD-9 and non-CM (e.g. OMIX) codes fall back to the
3-char root (flagged `root_fallback`). rep="icd10_root": 3-char ICD category
(after stripping the dot), dependency-light. Non-ICD code systems (apache,
past_history) pass through ungrouped under both.

NOTE: ``rep="icd10_root"`` does NOT apply a GEM remap (an ICD-9 code groups by
its ICD-9 root). ``rep="ccsr"`` DOES apply the CMS/NBER ICD-9->ICD-10-CM GEM
*before* the CCSR lookup (see ``ccsr.add_ccsr_grouping`` ->
``gem.add_icd10_equivalent``), which lifts miiv CCSR coverage ~41%->99%;
unmapped ICD-9 and non-CM (e.g. OMIX WHO/GB) codes still fall to
``root_fallback``. Raw codes remain the source of truth.
"""

from __future__ import annotations

import polars as pl

_ICD_SYSTEMS = ("icd9", "icd10")


def add_grouping(df: pl.LazyFrame, *, rep: str = "ccsr") -> pl.LazyFrame:
    """Add `group` + `group_source` columns. df needs `code`, `code_system`."""
    if rep == "icd10_root":
        is_icd = pl.col("code_system").is_in(list(_ICD_SYSTEMS))
        root = pl.col("code").str.replace(r"\.", "").str.slice(0, 3)
        return df.with_columns(
            pl.when(is_icd).then(root).otherwise(pl.col("code")).alias("group"),
            pl.when(is_icd)
            .then(pl.lit("root"))
            .otherwise(pl.lit("ungrouped"))
            .alias("group_source"),
        )
    if rep == "ccsr":
        from critical_mm.modalities.ccsr import add_ccsr_grouping

        result: pl.LazyFrame = add_ccsr_grouping(df)
        return result
    raise ValueError(f"unknown rep {rep!r}; valid: icd10_root, ccsr")
