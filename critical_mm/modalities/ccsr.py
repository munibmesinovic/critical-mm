"""AHRQ CCSR v2026.1 grouping for ICD-10-CM diagnosis codes.

Uses a slim vendored mapping (dotless ICD-10-CM code -> Default Inpatient CCSR
category) derived from the AHRQ HCUP DXCCSR v2026.1 reference file. See
`_refs/README.md` + `_refs/CHECKSUMS.sha256` for provenance and the exact
extraction step (reproducible from the public AHRQ ZIP).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import polars as pl

_CCSR_PATH = Path(__file__).parent / "_refs" / "dxccsr_v2026-1_default_ccsr.csv.gz"


@lru_cache(maxsize=1)
def _load_ccsr() -> pl.DataFrame:
    """Load the slim `icd10cm_code -> ccsr_category` map (dotless codes)."""
    raw = pl.read_csv(_CCSR_PATH, infer_schema_length=0)
    cols = raw.columns
    code_col = next(c for c in cols if "code" in c.lower())
    cat_col = next(c for c in cols if "ccsr" in c.lower())
    return raw.select(
        pl.col(code_col).str.replace_all("'", "").str.strip_chars().alias("__code"),
        pl.col(cat_col).str.replace_all("'", "").str.strip_chars().alias("__ccsr"),
    ).unique(subset=["__code"])


def add_ccsr_grouping(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add `group`/`group_source`; CM codes -> CCSR, else 3-char root fallback.

    ICD-9 codes are first GEM-remapped to ICD-10-CM (`gem.add_icd10_equivalent`)
    so they land in the same CCSR space as ICD-10; unmapped ICD-9 and non-CM
    (e.g. OMIX WHO/GB) codes fall back to the 3-char root of their ICD-10
    equivalent. Non-ICD systems (apache, past_history) pass through ungrouped.
    """
    from critical_mm.modalities.gem import add_icd10_equivalent

    ccsr = _load_ccsr().lazy()
    is_icd = pl.col("code_system").is_in(["icd9", "icd10"])
    with_dotless = df.with_columns(pl.col("code").str.replace(r"\.", "").alias("__dotless"))
    remapped = add_icd10_equivalent(with_dotless)
    joined = remapped.join(ccsr, left_on="__cm10", right_on="__code", how="left")
    root = pl.col("__cm10").str.slice(0, 3)
    return joined.with_columns(
        pl.when(~is_icd)
        .then(pl.col("code"))
        .when(pl.col("__ccsr").is_not_null())
        .then(pl.col("__ccsr"))
        .otherwise(root)
        .alias("group"),
        pl.when(~is_icd)
        .then(pl.lit("ungrouped"))
        .when(pl.col("__ccsr").is_not_null())
        .then(pl.lit("ccsr"))
        .otherwise(pl.lit("root_fallback"))
        .alias("group_source"),
    ).drop("__dotless", "__cm10", "__ccsr")
