"""Base cohort builder — apply 5 YAIB-paper-parity inclusion criteria.

Reads `data/interim/<dataset>/{stays,events_long}.parquet`, applies the
criteria in order, writes the surviving stays to
`data/processed/base_cohort/<dataset>/stays.parquet` plus an `attrition.csv`
documenting per-step drops. Writes are cached via the content-hash cache.

Criteria match YAIB paper App C.2 + Figure 6 + external/YAIB-cohorts/R/base_cohort.R:
  1. valid_times (excl1, "Invalid LoS / admit/discharge")
  2. los>=6h (excl2)
  3. ge_4_measured_bins (excl3, ≥4 hourly buckets with ≥1 dynamic measurement)
  4. no_12h_gap (excl4, no consecutive run of >12 measurement-free hours)
  5. age>=18 (excl5)

rewritten from the original 5-criterion set
(age, los>=6h, los<=28d, has_hr_in_24h, first_admission). The `has_hr_in_24h`
proxy was replaced with the proper paper-spec BASE-3/4; `los<=28d` was
dropped because YAIB has no upper LoS cap at base-cohort level (the 168h
LOS_CAP_HOURS in schema.py only governs per-hour outc grids, not cohort
membership); `first_admission` was dropped because YAIB-cohorts does NOT
deduplicate to first admission per patient — paper Table 14 shows MIMIC-IV
73k stays from 53k patients (1.42 stays/patient), eICU 183k stays from
161k patients (1.14). NOTE: the locked splits are STAY-level — `lock_splits.py`
uses StratifiedKFold on `stay_id` (matching YAIB's `make_single_split`), so a
patient with multiple stays CAN span train/test. This is a YAIB-faithful
limitation, not an active mitigation: `PatientGroupedKFold` exists in
`splits.py` but is NOT currently invoked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict

import polars as pl

from critical_mm.concepts import CONCEPTS_BY_NAME
from critical_mm.io.cache import build_cache_key, cache_lookup, cache_write
from critical_mm.io.parquet import scan
from critical_mm.schema import LOS_CAP_HOURS, TABLES

INCLUSION_CRITERIA: list[tuple[int, str, str]] = [
    (1, "valid_times", "non-null admit_time and discharge_time"),
    (2, "los>=6h", "ICU LoS ≥ 6h"),
    (3, "ge_4_measured_bins", "≥4 hourly bins with ≥1 dynamic-concept measurement"),
    (4, "no_12h_gap", "no consecutive >12h interval without any dynamic measurement"),
    (5, "age>=18", "age ≥ 18 at admission"),
]

_MIN_LOS_HOURS: float = 6.0
_MIN_MEASURED_BINS: int = 4
_MAX_GAP_HOURS: int = 12
_DATASET_MAX_GAP_HOURS: dict[str, int] = {"zigong": 48}

_DYNAMIC_CONCEPTS_FOR_GATING: frozenset[str] = frozenset(
    name
    for name, cc in CONCEPTS_BY_NAME.items()
    if cc.category in ("vital", "lab")
    and cc.canonical_unit is not None
    and cc.valid_range is not None
    and name != "gcs"
)

class CohortResult(TypedDict):
    cohort_path: Path
    attrition_path: Path
    n_in: int
    n_out: int
    per_step: list[dict[str, int | str | float]]

def _criteria_hash() -> str:
    """16-hex sha256 of the INCLUSION_CRITERIA list — feeds the cache extra."""
    canonical = json.dumps(INCLUSION_CRITERIA, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

def _measured_hour_buckets_lf(stays: pl.DataFrame, events_long: pl.LazyFrame) -> pl.LazyFrame:
    """LazyFrame [stay_id, hour] — unique (stay, hour-bucket) tuples with ≥1 dyn measurement.

    Hour bucket = floor((charttime - admit_time) / 1h). Restricted to
    hour in [0, floor(min(los_hours, LOS_CAP_HOURS))) (the gates are
    computed on the per-hour task grid, not the entire stay duration —
    matches YAIB's `stop_obs_at(patients, offset=7*24h)`).

    events_long is
    passed as a LazyFrame so the 48-concept filter + admits-join + window
    filter + unique can pipeline through polars' streaming engine. Eager
    `.collect()` of the full events_long at top-level (pre-refactor) blew
    past the 123 GiB host ceiling on eICU's ~50M-event events_long when
    BASE-3 multiplied the filter set from 1 (`hr`) to 48 dynamic concepts.
    """
    admits_lf = stays.lazy().select("stay_id", "admit_time", "los_hours")
    dyn_lf = events_long.filter(
        pl.col("concept").is_in(list(_DYNAMIC_CONCEPTS_FOR_GATING)) & pl.col("value").is_not_null()
    )
    joined = dyn_lf.join(admits_lf, on="stay_id", how="inner").with_columns(
        ((pl.col("charttime") - pl.col("admit_time")).dt.total_seconds() / 3600.0)
        .floor()
        .cast(pl.Int64)
        .alias("hour"),
        pl.col("los_hours").clip(0.0, float(LOS_CAP_HOURS)).cast(pl.Float64).alias("_cap"),
    )
    in_window = joined.filter(
        (pl.col("hour") >= 0) & (pl.col("hour").cast(pl.Float64) < pl.col("_cap"))
    )
    return in_window.select("stay_id", "hour").unique()

def _longest_gap_hours_per_stay_lf(stays: pl.DataFrame, measured_lf: pl.LazyFrame) -> pl.LazyFrame:
    """LazyFrame [stay_id, max_gap] — longest measurement-free run per stay.

    For each stay, the grid is hours [0, floor(min(los, LOS_CAP_HOURS))).
    Gap units are integer hours. Boundary handling: hours before the first
    measurement count as a leading gap; hours after the last measurement
    count as a trailing gap (matches YAIB's `longest_rle` semantic on the
    full grid). Sentinel measurements at -1 and at max_hour fold the
    leading + trailing gaps into the same diff-based logic.

    takes LazyFrame; the caller streams.
    """
    capped = stays.lazy().select(
        "stay_id",
        pl.col("los_hours")
        .clip(0.0, float(LOS_CAP_HOURS))
        .floor()
        .cast(pl.Int64)
        .alias("_max_hour"),
    )
    sentinels_lower = capped.select("stay_id", pl.lit(-1, dtype=pl.Int64).alias("hour"))
    sentinels_upper = capped.select("stay_id", pl.col("_max_hour").alias("hour"))
    all_hours = pl.concat(
        [measured_lf.select("stay_id", "hour"), sentinels_lower, sentinels_upper],
        how="vertical_relaxed",
    )
    sorted_lf = all_hours.sort("stay_id", "hour").with_columns(
        (pl.col("hour") - pl.col("hour").shift(1).over("stay_id")).fill_null(0).alias("_diff")
    )
    return sorted_lf.group_by("stay_id").agg(
        (pl.col("_diff").max() - 1).clip(0, None).alias("max_gap")
    )

def _stream_collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect a LazyFrame via the polars streaming engine.

    Polars 1.40 supports both ``collect(engine="streaming")`` (new) and the
    deprecated ``collect(streaming=True)``. We use the new API; if a future
    polars dropping it is detected, fall back to eager collect (with a
    warning that the cohort builder is back to non-streaming).
    """
    try:
        return lf.collect(engine="streaming")
    except (TypeError, ValueError):
        return lf.collect()

def _apply_criteria(
    stays: pl.DataFrame, events_long: pl.LazyFrame, max_gap_hours: int = _MAX_GAP_HOURS
) -> tuple[pl.DataFrame, list[dict[str, int | str | float]]]:
    """Apply the five YAIB-paper inclusion criteria in order; return survivors + attrition.

    `events_long` is a LazyFrame so steps 3 + 4 (the only ones that touch
    events_long) can stream through the polars engine without materializing
    the full ~50M-row frame in memory on eICU.
    """
    per_step: list[dict[str, int | str | float]] = []
    surviving = stays

    before = surviving.height
    surviving = surviving.filter(
        pl.col("admit_time").is_not_null()
        & pl.col("discharge_time").is_not_null()
        & (pl.col("discharge_time") >= pl.col("admit_time"))
    )
    after = surviving.height
    per_step.append(_attrition_row(1, "valid_times", before, after))

    before = surviving.height
    surviving = surviving.filter(pl.col("los_hours") >= _MIN_LOS_HOURS)
    after = surviving.height
    per_step.append(_attrition_row(2, "los>=6h", before, after))

    before = surviving.height
    if surviving.height == 0:
        after = 0
    else:
        measured_lf = _measured_hour_buckets_lf(surviving, events_long)
        n_bins_per_stay = _stream_collect(
            measured_lf.group_by("stay_id").agg(pl.len().alias("n_bins"))
        )
        if n_bins_per_stay.height == 0:
            surviving = surviving.filter(pl.lit(False))
        else:
            passing = n_bins_per_stay.filter(pl.col("n_bins") >= _MIN_MEASURED_BINS).select(
                "stay_id"
            )
            surviving = surviving.join(passing, on="stay_id", how="inner")
        after = surviving.height
    per_step.append(_attrition_row(3, "ge_4_measured_bins", before, after))

    before = surviving.height
    if surviving.height > 0:
        measured_lf = _measured_hour_buckets_lf(surviving, events_long)
        gap_per_stay = _stream_collect(_longest_gap_hours_per_stay_lf(surviving, measured_lf))
        passing = gap_per_stay.filter(pl.col("max_gap") <= max_gap_hours).select("stay_id")
        surviving = surviving.join(passing, on="stay_id", how="inner")
    after = surviving.height
    per_step.append(_attrition_row(4, "no_12h_gap", before, after))

    before = surviving.height
    surviving = surviving.filter(pl.col("age") >= 18)
    after = surviving.height
    per_step.append(_attrition_row(5, "age>=18", before, after))

    return surviving, per_step

def _attrition_row(
    step: int, criterion: str, n_before: int, n_after: int
) -> dict[str, int | str | float]:
    n_dropped = n_before - n_after
    pct = (n_dropped / n_before * 100.0) if n_before else 0.0
    return {
        "step": step,
        "criterion": criterion,
        "n_before": n_before,
        "n_after": n_after,
        "n_dropped": n_dropped,
        "pct_dropped": round(pct, 4),
    }

def _write_attrition_csv(
    path: Path,
    per_step: list[dict[str, int | str | float]],
    n_in: int,
    n_out: int,
) -> None:
    """Plain CSV with the header + 5 step rows + a 'final' summary row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "step,criterion,n_before,n_after,n_dropped,pct_dropped\n"
    body_lines = [
        f"{r['step']},{r['criterion']},{r['n_before']},{r['n_after']},"
        f"{r['n_dropped']},{r['pct_dropped']}\n"
        for r in per_step
    ]
    n_dropped = n_in - n_out
    pct = (n_dropped / n_in * 100.0) if n_in else 0.0
    final_line = f"final,total,{n_in},{n_out},{n_dropped},{round(pct, 4)}\n"
    path.write_text(header + "".join(body_lines) + final_line)

def build_base_cohort(
    *,
    dataset: str,
    interim_root: Path,
    processed_root: Path,
    repo_root: Path,
) -> CohortResult:
    """Apply the 5-step inclusion gate to the dataset's interim stays.

    Reads `interim_root/<dataset>/{stays,events_long}.parquet`, applies the
    criteria, writes the survivors to
    `processed_root/base_cohort/<dataset>/stays.parquet` plus an
    `attrition.csv`. Cache key includes the dataset name and the
    INCLUSION_CRITERIA hash.
    """
    interim_root = Path(interim_root)
    processed_root = Path(processed_root)
    stays_path = interim_root / dataset / "stays.parquet"
    events_path = interim_root / dataset / "events_long.parquet"
    if not stays_path.exists():
        raise FileNotFoundError(f"interim stays missing for {dataset}: {stays_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"interim events_long missing for {dataset}: {events_path}")

    target_stays = processed_root / "base_cohort" / dataset / "stays.parquet"
    target_attrition = processed_root / "base_cohort" / dataset / "attrition.csv"

    max_gap_hours = _DATASET_MAX_GAP_HOURS.get(dataset, _MAX_GAP_HOURS)
    extra = f"base_cohort/{dataset}/{_criteria_hash()}"
    if max_gap_hours != _MAX_GAP_HOURS:
        extra += f"/gap{max_gap_hours}"
    key = build_cache_key(
        source_paths=[stays_path, events_path],
        tree_roots=[repo_root / "critical_mm", repo_root / "configs"],
        repo_root=repo_root,
        extra=extra,
    )
    if cache_lookup(target_stays, key) is not None and target_attrition.exists():
        per_step, n_in, n_out = _replay_attrition(target_attrition)
        return CohortResult(
            cohort_path=target_stays,
            attrition_path=target_attrition,
            n_in=n_in,
            n_out=n_out,
            per_step=per_step,
        )

    stays = scan(stays_path).collect()
    events_long_lf = scan(events_path)
    n_in = stays.height

    surviving, per_step = _apply_criteria(stays, events_long_lf, max_gap_hours=max_gap_hours)
    n_out = surviving.height

    def _writer(tmp_path: Path) -> None:
        surviving.lazy().sink_parquet(tmp_path, compression="zstd", statistics=True)

    cache_write(target_stays, key, _writer)
    _write_attrition_csv(target_attrition, per_step, n_in, n_out)

    return CohortResult(
        cohort_path=target_stays,
        attrition_path=target_attrition,
        n_in=n_in,
        n_out=n_out,
        per_step=per_step,
    )

def _replay_attrition(
    attrition_path: Path,
) -> tuple[list[dict[str, int | str | float]], int, int]:
    """Reconstruct (per_step, n_in, n_out) from a previously-written CSV."""
    lines = attrition_path.read_text().splitlines()
    per_step: list[dict[str, int | str | float]] = []
    n_in = 0
    n_out = 0
    for line in lines[1:]:
        parts = line.split(",")
        if parts[0] == "final":
            n_in = int(parts[2])
            n_out = int(parts[3])
            continue
        per_step.append(
            {
                "step": int(parts[0]),
                "criterion": parts[1],
                "n_before": int(parts[2]),
                "n_after": int(parts[3]),
                "n_dropped": int(parts[4]),
                "pct_dropped": float(parts[5]),
            }
        )
    return per_step, n_in, n_out

_STAYS_SCHEMA = TABLES["stays"][0]
__all__ = ["INCLUSION_CRITERIA", "CohortResult", "build_base_cohort"]
del _STAYS_SCHEMA
