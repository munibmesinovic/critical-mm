"""Per-dataset note cleaning: de-id normalisation, dedup, GB18030 decode, eICU serialization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl

_DEID_BRACKET = re.compile(r"\[\*\*.*?\*\*\]")
_DEID_OPEN = re.compile(r"\[\*\*")
_DEID_CLOSE = re.compile(r"\*\*\]")
_DEID_BLANK = re.compile(r"_{2,}")
_WS = re.compile(r"\s+")

def clean_clinical_text(text: str | None) -> str:
    """Normalise MIMIC de-id placeholders + whitespace. Empty string for null."""
    if text is None:
        return ""
    t = _DEID_BRACKET.sub(" ", text)
    t = _DEID_OPEN.sub(" ", t)
    t = _DEID_CLOSE.sub(" ", t)
    t = _DEID_BLANK.sub(" ", t)
    t = _WS.sub(" ", t)
    return t.strip()

def clean_text_expr(col: str = "text") -> pl.Expr:
    """Vectorised polars equivalent of clean_clinical_text (de-id placeholders + whitespace)."""
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .fill_null("")
        .str.replace_all(r"\[\*\*.*?\*\*\]", " ")
        .str.replace_all(r"\[\*\*", " ")
        .str.replace_all(r"\*\*\]", " ")
        .str.replace_all(r"_{2,}", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def decode_gb18030(
    path: str | Path,
    *,
    infer_schema_length: int = 100,
    ignore_errors: bool = False,
    schema_overrides: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Read a GB18030-encoded CSV into a polars DataFrame (file decoded in Python first).

    Extra kwargs are forwarded to ``pl.read_csv``. ``infer_schema_length`` defaults
    to 100 (sufficient for most OMIX tables); callers with heterogeneous columns
    (e.g. Hospital_ID written in scientific notation) should pass ``ignore_errors=True``
    or a ``schema_overrides`` dict.
    """
    import csv
    import io

    with open(path, encoding="gb18030", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return pl.DataFrame()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)
    return pl.read_csv(
        buf.getvalue().encode("utf-8"),
        infer_schema_length=infer_schema_length,
        ignore_errors=ignore_errors,
        schema_overrides=schema_overrides,
    )

def compose_omix_text(lf: pl.LazyFrame) -> pl.LazyFrame:
    """text = '<Category> | <item_Eng>: <DESC> <Finding>' (English header + Chinese body)."""
    return lf.with_columns(
        (
            pl.col("ExamReport_Category").fill_null("")
            + pl.lit(" | ")
            + pl.col("ExamReport_item_Eng").fill_null("")
            + pl.lit(": ")
            + pl.col("ExamReport_DESC").fill_null("")
            + pl.lit(" ")
            + pl.col("ExamReport_Finding").fill_null("")
        )
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("text")
    )

_EICU_DROP_LEAVES: frozenset[str] = frozenset(
    {"Obtain Options", "View Options", "Copies", "Print", "Save Options", "Performed - Structured"}
)
_MAX_PREADMIT_LOOKBACK_MIN = 30 * 24 * 60

def drop_implausible_offset_min(lf: pl.LazyFrame, *, col: str) -> pl.LazyFrame:
    """Drop rows whose minute-offset is below a 30-day pre-admission lookback (data artifacts)."""
    return lf.filter(pl.col(col) >= -_MAX_PREADMIT_LOOKBACK_MIN)

def serialize_eicu_rows(lf: pl.LazyFrame, *, note_type: str) -> pl.LazyFrame:
    """Render label->value rows into one '<label>: <value>; ...' pseudo-doc per (stay_id, hour).

    ``lf`` columns required: stay_id, hour, label, value. Drops UI leaves; sorts by label
    for determinism; emits (stay_id, hour, note_type, text).
    """
    _drop = list(_EICU_DROP_LEAVES)
    clean = (
        lf.filter(~pl.col("label").str.contains_any(_drop) & ~pl.col("value").is_in(_drop))
        .filter(pl.col("value").is_not_null() & (pl.col("value").str.len_chars() > 0))
        .with_columns((pl.col("label") + pl.lit(": ") + pl.col("value")).alias("__line"))
        .sort(["stay_id", "hour", "label"])
    )
    return (
        clean.group_by(["stay_id", "hour"])
        .agg(pl.col("__line").str.join("; ").alias("text"))
        .with_columns(pl.lit(note_type).alias("note_type"))
    )
