"""CMS/NBER ICD-9-CM -> ICD-10-CM General Equivalence Mapping (2018 forward GEM).

Slim vendored map (first ICD-10-CM per ICD-9-CM; `no_map` rows dropped) used to
remap ICD-9 diagnosis codes to their ICD-10-CM equivalent before CCSR grouping,
so a dataset's ICD-9 history lands in the same category space as its ICD-10.
Provenance + checksum in `_refs/README.md` / `_refs/CHECKSUMS.sha256`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import polars as pl

_GEM_PATH = Path(__file__).parent / "_refs" / "icd9to10cm_gem_2018_slim.csv.gz"

@lru_cache(maxsize=1)
def _load_gem() -> pl.DataFrame:
    """Load the slim ICD-9 -> ICD-10-CM map (dotless codes)."""
    return (
        pl.read_csv(_GEM_PATH, infer_schema_length=0)
        .select(
            pl.col("icd9").str.strip_chars().alias("__icd9"),
            pl.col("icd10").str.strip_chars().alias("__gem10"),
        )
        .unique(subset=["__icd9"])
    )

def add_icd10_equivalent(df: pl.LazyFrame, *, dotless_col: str = "__dotless") -> pl.LazyFrame:
    """Add a `__cm10` column: the ICD-10-CM-equivalent dotless code.

    For `code_system == "icd9"` rows with a GEM match, `__cm10` is the mapped
    ICD-10-CM code; otherwise (ICD-10 already, or an unmapped ICD-9) it is the
    original dotless code. Requires `code_system` and `dotless_col` columns.
    """
    gem = _load_gem().lazy()
    return (
        df.join(gem, left_on=dotless_col, right_on="__icd9", how="left")
        .with_columns(
            pl.when((pl.col("code_system") == "icd9") & pl.col("__gem10").is_not_null())
            .then(pl.col("__gem10"))
            .otherwise(pl.col(dotless_col))
            .alias("__cm10")
        )
        .drop("__gem10")
    )

