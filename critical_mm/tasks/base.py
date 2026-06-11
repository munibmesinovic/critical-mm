"""Abstract `Task` base — shared `build()` produces YAIB-shaped sta/dyn/outc.

The YAIB exporter contract () and the validation oracle () consume:
- `sta.parquet`: one row per stay, four static feature columns
  (patient_id, stay_id, age, sex, weight, height).
- `dyn.parquet`: one row per (stay_id, hour) for the prediction window,
  with one Float32 column per numeric vital/lab concept (NaN where
  missing in that hour bucket).
- `outc.parquet`: one row per (stay_id) (or per (stay_id, hour) for
  los) with label_value.

Subclasses override `build_labels()` to emit the outc frame; the rest of
the pipeline (dyn aggregation, sta extraction, atomic writes) lives here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal, TypedDict

import polars as pl

from critical_mm.concepts import CONCEPTS_BY_NAME
from critical_mm.io.parquet import scan
from critical_mm.schema import LOS_CAP_HOURS as LOS_CAP_HOURS
from critical_mm.schema import TABLES

_DYNAMIC_CONCEPT_EXCLUSIONS: frozenset[str] = frozenset({"gcs", "urine_rate"})
DYNAMIC_CONCEPTS: list[str] = sorted(
    name
    for name, cc in CONCEPTS_BY_NAME.items()
    if cc.category in ("vital", "lab")
    and cc.canonical_unit is not None
    and cc.valid_range is not None
    and name not in _DYNAMIC_CONCEPT_EXCLUSIONS
)


class TaskBuildResult(TypedDict):
    sta_path: Path
    dyn_path: Path
    outc_path: Path
    n_stays: int


class Task(ABC):
    """Base class for a CRITICAL-MM v1 task."""

    task_name: ClassVar[str]
    task_type: ClassVar[Literal["classification", "regression"]]
    outcome_min: ClassVar[float | None] = None
    outcome_max: ClassVar[float | None] = None
    prediction_horizon_hours: ClassVar[int]

    def required_dataset_capabilities(self, dataset: str) -> frozenset[str]:
        """Structural dataset features this task requires for the given dataset.

        Default returns the empty set -- the task makes no structural demands.
        Override on subclasses (especially external contrib tasks under )
        to declare requirements; ``critical_mm.training.train.train_one`` will
        raise ValueError at dispatch time when a (task, dataset) pair lists a
        capability the dataset's ``CAPABILITIES`` ClassVar does not include.

        The argument is the dataset name (e.g. "miiv", "eicu") so per-dataset
        variants can declare different requirements: a task might need
        "microbio" on miiv but accept the abx surrogate on eicu/hirid, in
        which case it returns ``frozenset()`` for the latter.
        """
        del dataset
        return frozenset()

    @abstractmethod
    def build_labels(
        self,
        base_cohort: pl.DataFrame,
        events_long: pl.DataFrame | pl.LazyFrame,
        meds: pl.DataFrame,
        dataset: str,
        microbio: pl.DataFrame | None = None,
        interventions: pl.DataFrame | None = None,
        abx_duration: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Return outc-shaped frame.

        Columns: patient_id, stay_id, label_time, label_value (and `hour`
        for los which produces per-hour rows). `label_value` is Float32 for
        regression tasks and Int8 (0/1) for classification.

        `dataset` is passed through so per-dataset variants (e.g. AKI's
        NWICU creatinine-only arm) can branch.

        `microbio` and `interventions` are loaded from interim by
        ``Task.build`` ( Task 6) and threaded through for tasks (Sepsis-3)
        that need them; other tasks accept and ignore the kwargs.

        Audit round 10c (2026-05-19): `events_long` now accepts a LazyFrame
        so the streaming-engine pipeline avoids materializing eICU's ~50M
        rows in memory. Tests passing small DataFrame fixtures still work
        — `_to_events_lf()` adapts both. Phase B of B1 unblock.
        """

    def _dyn_max_hour_per_stay(self, cohort: pl.DataFrame) -> pl.DataFrame:
        """Return [stay_id, max_hour] for the dyn grid (INCLUSIVE upper bound).

        Default (used by Mortality24, KidneyFunction): constant
        `prediction_horizon_hours` so the dyn grid is 0..horizon inclusive
        (i.e. horizon+1 rows per stay; 25 for horizon=24). Per-hour tasks
        (LengthOfStay, AKI, Sepsis) override to per-stay variable max_hour =
        floor(min(los_hours, LOS_CAP_HOURS)) so dyn spans the full stay
        and aligns 1:1 with outc.
        """
        return cohort.select(
            "stay_id",
            pl.lit(self.prediction_horizon_hours, dtype=pl.Int32).alias("max_hour"),
        )

    def build(
        self,
        *,
        dataset: str,
        processed_root: Path,
        interim_root: Path,
        repo_root: Path,
    ) -> TaskBuildResult:
        """Read base cohort + interim, write sta/dyn/outc parquets.

        Output goes to `processed_root/<task_name>/<dataset>/`. The writes
        are best-effort idempotent: re-running with identical inputs leaves
        the parquets bit-identical (polars sink_parquet is deterministic
        over deterministic content). A future will re-wrap this in
        the content-hash cache for proper invalidation.
        """
        del repo_root
        processed_root = Path(processed_root)
        interim_root = Path(interim_root)
        target_dir = processed_root / self.task_name / dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        sta_path = target_dir / "sta.parquet"
        dyn_path = target_dir / "dyn.parquet"
        outc_path = target_dir / "outc.parquet"

        base_cohort_path = processed_root / "base_cohort" / dataset / "stays.parquet"
        events_long_path = interim_root / dataset / "events_long.parquet"
        meds_path = interim_root / dataset / "meds.parquet"
        microbio_path = interim_root / dataset / "microbio.parquet"
        interventions_path = interim_root / dataset / "interventions.parquet"
        abx_duration_path = interim_root / dataset / "abx_duration.parquet"
        if not base_cohort_path.exists():
            raise FileNotFoundError(f"base cohort missing for {dataset}: {base_cohort_path}")
        if not events_long_path.exists():
            raise FileNotFoundError(
                f"interim events_long missing for {dataset}: {events_long_path}"
            )

        base_cohort = scan(base_cohort_path).collect()
        events_long = scan(events_long_path)
        meds = scan(meds_path).collect() if meds_path.exists() else pl.DataFrame()
        microbio = (
            pl.read_parquet(microbio_path)
            if microbio_path.exists()
            else pl.DataFrame(schema=TABLES["microbio"][0])
        )
        interventions = (
            pl.read_parquet(interventions_path)
            if interventions_path.exists()
            else pl.DataFrame(schema=TABLES["interventions"][0])
        )
        abx_duration = (
            pl.read_parquet(abx_duration_path)
            if abx_duration_path.exists()
            else pl.DataFrame(schema=TABLES["abx_duration"][0])
        )

        outc = self.build_labels(
            base_cohort=base_cohort,
            events_long=events_long,
            meds=meds,
            dataset=dataset,
            microbio=microbio,
            interventions=interventions,
            abx_duration=abx_duration,
        )
        surviving_ids = set(outc["stay_id"].unique().to_list())
        if not surviving_ids:
            cohort = base_cohort.head(0)
        else:
            cohort = base_cohort.filter(pl.col("stay_id").is_in(list(surviving_ids)))
        sta = _build_sta(cohort)
        max_hour_per_stay = self._dyn_max_hour_per_stay(cohort)
        dyn = _build_dyn(cohort, events_long, max_hour_per_stay)

        sta.write_parquet(sta_path, compression="zstd")
        dyn.write_parquet(dyn_path, compression="zstd")
        outc.write_parquet(outc_path, compression="zstd")

        return TaskBuildResult(
            sta_path=sta_path,
            dyn_path=dyn_path,
            outc_path=outc_path,
            n_stays=cohort.height,
        )


def _build_sta(cohort: pl.DataFrame) -> pl.DataFrame:
    """One row per stay; static feature columns per the YAIB contract."""
    sex_one_hot = (
        pl.when(pl.col("sex") == "M")
        .then(pl.lit(1.0))
        .when(pl.col("sex") == "F")
        .then(pl.lit(0.0))
        .otherwise(pl.lit(0.5))
        .alias("sex_male")
    )
    return cohort.select(
        "patient_id",
        "stay_id",
        pl.col("age").cast(pl.Float32),
        sex_one_hot.cast(pl.Float32),
        pl.col("weight").cast(pl.Float32),
        pl.col("height").cast(pl.Float32),
    )


def _build_dyn(
    cohort: pl.DataFrame,
    events_long: pl.DataFrame | pl.LazyFrame,
    max_hour_per_stay: pl.DataFrame,
) -> pl.DataFrame:
    """Hourly aggregation over hours [0, max_hour] INCLUSIVE per stay.

    `max_hour_per_stay`: DataFrame [stay_id, max_hour]. Each stay's dyn
    grid covers hours 0..max_hour inclusive (max_hour+1 rows). Events with
    `charttime in [admit_time, admit_time + (max_hour+1)*1h)` bucket via
    `floor((charttime - admit_time) / 3600)` into hours 0..max_hour. Per
    (stay_id, hour, concept) the value is the MEDIAN — matches ricu's
    `aggregate.id_tbl` default, so YAIB-models pretrained checkpoints
    ( oracle) see the same per-hour values they were trained on.

    Audit round 10c (2026-05-19, B1 phase B): events_long is a LazyFrame
    when called from Task.build; the join + bucket + group_by + pivot
    pipelines through polars' streaming engine without materializing the
    full filtered events frame. Tests passing DataFrames still work via
    `.lazy()` at the entry.
    """
    if cohort.height == 0:
        return _empty_dyn()

    events_long_lf = events_long.lazy() if isinstance(events_long, pl.DataFrame) else events_long

    stays = (
        cohort.select("stay_id", "admit_time")
        .join(max_hour_per_stay, on="stay_id", how="inner")
        .with_columns(
            pl.col("admit_time")
            .dt.offset_by((pl.col("max_hour") + 1).cast(pl.Utf8) + pl.lit("h"))
            .alias("horizon_end")
        )
    )
    grid = _expand_to_grid(stays.select("stay_id", "max_hour"))

    events_lf = events_long_lf.filter(pl.col("concept").is_in(DYNAMIC_CONCEPTS))
    stays_lf = stays.lazy()
    joined_lf = events_lf.join(stays_lf, on="stay_id", how="inner").filter(
        (pl.col("charttime") >= pl.col("admit_time"))
        & (pl.col("charttime") < pl.col("horizon_end"))
    )
    joined_lf = joined_lf.with_columns(
        ((pl.col("charttime") - pl.col("admit_time")).dt.total_seconds().floordiv(3600))
        .cast(pl.Int32)
        .alias("hour")
    )
    aggregated_lf = joined_lf.group_by(["stay_id", "hour", "concept"]).agg(
        pl.col("value").cast(pl.Float64).median().cast(pl.Float32).alias("value")
    )
    aggregated = _stream_collect(aggregated_lf)
    if aggregated.height == 0:
        return grid

    wide = aggregated.pivot(
        on="concept",
        index=["stay_id", "hour"],
        values="value",
        aggregate_function="first",
    )
    for concept in DYNAMIC_CONCEPTS:
        if concept not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float32).alias(concept))
    return (
        grid.drop(DYNAMIC_CONCEPTS, strict=False)
        .join(wide, on=["stay_id", "hour"], how="left")
        .select("stay_id", "hour", *DYNAMIC_CONCEPTS)
    )


def _stream_collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Streaming-engine collect with eager fallback (mirrors cohorts.base)."""
    try:
        return lf.collect(engine="streaming")
    except (TypeError, ValueError):
        return lf.collect()


def _expand_to_grid(stays_with_max: pl.DataFrame) -> pl.DataFrame:
    """[stay_id, max_hour] → [stay_id, hour, *DYNAMIC_CONCEPTS] for hour ∈ [0, max_hour]."""
    expanded = (
        stays_with_max.with_columns(pl.int_ranges(0, pl.col("max_hour") + 1).alias("hour"))
        .explode("hour")
        .with_columns(pl.col("hour").cast(pl.Int32))
        .select("stay_id", "hour")
    )
    for concept in DYNAMIC_CONCEPTS:
        expanded = expanded.with_columns(pl.lit(None, dtype=pl.Float32).alias(concept))
    return expanded.select("stay_id", "hour", *DYNAMIC_CONCEPTS)


def _empty_dyn() -> pl.DataFrame:
    cols: dict[str, list[object]] = {
        "stay_id": [],
        "hour": [],
        **{c: [] for c in DYNAMIC_CONCEPTS},
    }
    schema: dict[str, pl.DataType] = {
        "stay_id": pl.Utf8(),
        "hour": pl.Int32(),
        **{c: pl.Float32() for c in DYNAMIC_CONCEPTS},
    }
    return pl.DataFrame(cols, schema=schema)
