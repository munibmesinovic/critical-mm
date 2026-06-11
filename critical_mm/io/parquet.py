"""Thin parquet wrappers with schema-validated atomic writes.

`scan` / `sink` are direct passthroughs to polars with safe defaults.
`write_with_schema` adds an atomic ".tmp → rename" sequence guarded by
`critical_mm.schema.validate_frame`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from critical_mm.schema import TABLES, validate_frame


def scan(path: Path | str) -> pl.LazyFrame:
    """Lazily scan a parquet file with default statistics enabled."""
    return pl.scan_parquet(path)


def sink(
    df: pl.LazyFrame,
    path: Path | str,
    *,
    compression: str = "zstd",
    statistics: bool = True,
) -> None:
    """Materialise a LazyFrame to parquet with zstd compression and column statistics."""
    df.sink_parquet(path, compression=compression, statistics=statistics)


def write_with_schema(
    df: pl.LazyFrame,
    path: Path | str,
    table_name: str,
) -> None:
    """Cast `df` to the canonical schema for `table_name` then atomically sink it.

    Sequence:
      1. Cast every column declared in `TABLES[table_name].schema` to the
         canonical dtype (`pl.col(c).cast(dtype)`).
      2. Call `validate_frame` — raises `ValueError` on missing required
         columns, dtype mismatch, or null in a non-nullable column.
      3. Sink to a sibling `.tmp` file, then `Path.replace` it to the
         target path (POSIX-atomic on the same filesystem).
      4. On any failure between sink and rename, unlink the `.tmp` file
         so we never leave the final path partially written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / (target.name + ".tmp")

    schema, _ = TABLES[table_name]
    cast_exprs = [pl.col(col).cast(dtype) for col, dtype in schema.items()]
    df_cast = df.select(cast_exprs)
    validate_frame(df_cast, table_name)

    try:
        df_cast.sink_parquet(tmp, compression="zstd", statistics=True)
        tmp.replace(target)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
