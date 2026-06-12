"""H1.2 invariants for events_long emitted by readers.

Two responsibilities:
1. `apply_canonical_units` — set `events.unit = CONCEPTS_BY_NAME[concept].canonical_unit`
   so every row carries the canonical SI string (H1.2 non-null + canonical-
   spelling invariant). `unit_source` is preserved as the pre-conversion
   audit column.
2. `drop_null_required` — drop rows with null in any non-nullable
   `events_long` column (patient_id, stay_id, charttime, concept, value,
   unit). Real data (e.g. NWICU labevents with text-only results) emits
   null `value` rows that would otherwise trip `validate_frame`. Silently
   dropping is the lesser evil; downstream cohort reporting picks up
   row-count diffs.
"""

from __future__ import annotations

import polars as pl

from critical_mm.concepts import CONCEPTS_BY_NAME

_CANONICAL_UNIT_LOOKUP: pl.LazyFrame = pl.LazyFrame(
    {
        "concept": [n for n, c in CONCEPTS_BY_NAME.items() if c.canonical_unit],
        "_cmm_canonical_unit": [
            c.canonical_unit for c in CONCEPTS_BY_NAME.values() if c.canonical_unit
        ],
    },
    schema={"concept": pl.Utf8(), "_cmm_canonical_unit": pl.Utf8()},
)

_VALID_RANGE_LOOKUP: pl.LazyFrame = pl.LazyFrame(
    {
        "concept": [n for n, c in CONCEPTS_BY_NAME.items() if c.valid_range is not None],
        "_cmm_range_lo": [
            c.valid_range[0] for c in CONCEPTS_BY_NAME.values() if c.valid_range is not None
        ],
        "_cmm_range_hi": [
            c.valid_range[1] for c in CONCEPTS_BY_NAME.values() if c.valid_range is not None
        ],
    },
    schema={"concept": pl.Utf8(), "_cmm_range_lo": pl.Float64(), "_cmm_range_hi": pl.Float64()},
)

def drop_null_required(events: pl.LazyFrame) -> pl.LazyFrame:
    """Drop rows with null in any non-nullable events_long column.

    Non-nullable per H1.2: patient_id, stay_id, charttime, concept, value, unit.
    `unit_source` is the only nullable column.
    """
    return events.filter(
        pl.col("patient_id").is_not_null()
        & pl.col("stay_id").is_not_null()
        & pl.col("charttime").is_not_null()
        & pl.col("concept").is_not_null()
        & pl.col("value").is_not_null()
        & pl.col("unit").is_not_null()
    )

def apply_canonical_units(
    events: pl.LazyFrame,
    out_of_range: str = "drop",
    sentinel_null_threshold: float | None = None,
) -> pl.LazyFrame:
    """Set `unit = canonical_unit[concept]`, handle valid_range, drop nulls.

    Concepts without a registered `canonical_unit` (sex, inr_pt, ph, aki,
    sepsis) keep the incoming `unit` value via `coalesce`. After unit
    canonicalisation, rows whose value falls outside the concept's registry
    valid_range are handled per `out_of_range` (audit 10ah, 2026-05-20).
    Finally, null-row cleanup drops schema-violating rows — typically
    text-only labevents (NWICU labs without a numeric value) and source rows
    where the timestamp failed to parse.

    Args:
        out_of_range: "drop" (default, historic) filters rows whose value is
            outside the concept valid_range; "clip" winsorises them to [lo, hi]
            instead, preserving the measurement-occurred signal (NWICU sentinel
            re-lock, 2026-06-04 — see project-cmm-nwicu-sentinel-corrupts-paper-tensors).
        sentinel_null_threshold: when set, values with |value| >= threshold are
            nulled BEFORE range handling, so a uniform missing-value sentinel
            (NWICU's 9999999) is treated as missing and dropped by
            `drop_null_required`, never clipped into a fabricated extreme.

    With the defaults this is byte-identical to the pre-2026-06-04 behaviour,
    so every reader except NWICU is unaffected.
    """
    if out_of_range not in ("drop", "clip"):
        raise ValueError(f"out_of_range must be 'drop' or 'clip', got {out_of_range!r}")
    joined = events.join(_CANONICAL_UNIT_LOOKUP, on="concept", how="left")
    canonicalised = joined.with_columns(
        pl.coalesce([pl.col("_cmm_canonical_unit"), pl.col("unit")]).alias("unit"),
    ).drop("_cmm_canonical_unit")
    if sentinel_null_threshold is not None:
        canonicalised = canonicalised.with_columns(
            pl.when(pl.col("value").abs() >= sentinel_null_threshold)
            .then(None)
            .otherwise(pl.col("value"))
            .alias("value")
        )
    ranged = canonicalised.join(_VALID_RANGE_LOOKUP, on="concept", how="left")
    if out_of_range == "drop":
        handled = ranged.filter(
            pl.col("_cmm_range_lo").is_null()
            | (
                (pl.col("value") >= pl.col("_cmm_range_lo"))
                & (pl.col("value") <= pl.col("_cmm_range_hi"))
            )
        )
    else:
        handled = ranged.with_columns(
            pl.when(pl.col("_cmm_range_lo").is_null())
            .then(pl.col("value"))
            .otherwise(pl.col("value").clip(pl.col("_cmm_range_lo"), pl.col("_cmm_range_hi")))
            .alias("value")
        )
    handled = handled.drop("_cmm_range_lo", "_cmm_range_hi")
    return drop_null_required(handled)
