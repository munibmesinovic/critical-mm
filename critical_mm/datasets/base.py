"""Abstract base class for every dataset reader in the pipeline.

Six abstract methods (one per canonical interim table) plus a default
`harmonise_all()` that runs each, validates the result, and writes through
the content-hash cache. Concrete readers (MIMIC-IV, eICU, HiRID, NWICU,
synthetic) override the abstracts and may override `cache_source_paths` to
narrow the per-table dependency footprint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import polars as pl

from critical_mm.concepts import CONCEPTS
from critical_mm.io.cache import build_cache_key, cache_lookup, cache_write
from critical_mm.schema import validate_frame

_TABLE_NAMES: tuple[str, ...] = (
    "stays",
    "events_long",
    "meds",
    "interventions",
    "notes",
    "diagnoses",
    "microbio",
    "abx_duration",
)


class DatasetReader(ABC):
    """Contract: produce six canonical interim tables for one dataset.

    Subclasses MUST implement `dataset_name` and the six `read_*` methods.
    Subclasses MAY override `cache_source_paths` to declare per-table source
    files; the default returns every file under `raw_root` (coarse but safe —
    any change anywhere in the dataset invalidates every table).

    Subclasses MAY override `CAPABILITIES` () to declare which structural
    features they support. Tasks consult this via
    `Task.required_dataset_capabilities()`; `train_one()` raises ValueError
    when a (task, dataset) pair is structurally incompatible. Default empty
    set means "no structural guarantees declared" -- safe but tasks with
    requirements will refuse to run.
    """

    CAPABILITIES: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        raw_root: Path,
        interim_root: Path,
        repo_root: Path,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.interim_root = Path(interim_root)
        self.repo_root = Path(repo_root)

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """One of 'mimic_iv', 'eicu', 'hirid', 'nwicu', 'synthetic'."""

    @abstractmethod
    def read_stays(self) -> pl.LazyFrame: ...

    @abstractmethod
    def read_events_long(self, concepts: list[str]) -> pl.LazyFrame: ...

    @abstractmethod
    def read_meds(self) -> pl.LazyFrame: ...

    @abstractmethod
    def read_interventions(self) -> pl.LazyFrame: ...

    @abstractmethod
    def read_notes(self) -> pl.LazyFrame: ...

    @abstractmethod
    def read_diagnoses(self) -> pl.LazyFrame: ...

    @abstractmethod
    def read_microbio(self) -> pl.LazyFrame:
        """Per-ICU-stay microbiology culture samples (, B8 Phase 1).

        Returns rows of [patient_id, stay_id, charttime, specimen_type, organism].
        `organism` is null for culture-negative samples (the sampling itself
        is the meaningful signal for `susp_inf_alt`). `stay_id` must be
        non-null — readers drop microbio rows that don't fall within any
        ICU stay's [intime, outtime].
        """

    def read_abx_duration(self) -> pl.LazyFrame:
        """ricu-faithful antibiotic duration episodes (audit round 10m).

        Default implementation returns an empty frame — subclasses
        whose ricu-extensions `concept-dict.json#abx_duration.sources`
        entry is defined override this to apply the per-(source-table)
        regex / itemid match and the correct duration callback (1-min
        point event vs real `dur_var`). NWICU has no ricu source entry,
        so its abx surrogate path continues to read from `meds.parquet`
        via `drug_class == "antibiotic"`.

        Returns rows of [stay_id, starttime, endtime].
        """
        from critical_mm.schema import empty_frame

        return empty_frame("abx_duration")

    def cache_source_paths(self, table: str) -> list[Path]:
        """Per-table source files for cache invalidation.

        Default: every file under `raw_root` (coarse). Override in subclasses
        to track only the files the table actually reads (e.g. NWICU's
        `labevents.csv.gz` distinct from `chartevents.csv.gz`).
        """
        del table
        if not self.raw_root.exists():
            return []
        return sorted(p for p in self.raw_root.rglob("*") if p.is_file())

    def harmonise_all(
        self,
        *,
        concepts: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Path]:
        """Produce the six canonical parquets for this dataset.

        For each of the six tables:
          1. Build a CacheKey from `cache_source_paths(table)` + the
             critical_mm/ and configs/ tree hashes + repo git SHA + an
             `extra` string of `{dataset_name}/{table}` (plus a concepts
             hash for events_long).
          2. If `force` is False and `cache_lookup` hits, record the
             cached path and skip regeneration.
          3. Otherwise call the matching `read_*`, run `validate_frame`,
             and `cache_write` the result to
             `interim_root/<dataset_name>/<table>.parquet`.

        Returns a `dict[table_name → Path]` covering all six tables. A
        `validate_frame` failure on one table aborts that table only; tables
        written earlier in the loop remain on disk.
        """
        if concepts is None:
            concepts = [c.name for c in CONCEPTS]
        results: dict[str, Path] = {}
        for table in _TABLE_NAMES:
            results[table] = self._harmonise_one(table, concepts, force=force)
        return results

    def _harmonise_one(
        self,
        table: str,
        concepts: list[str],
        *,
        force: bool,
    ) -> Path:
        extra = f"{self.dataset_name}/{table}"
        if table == "events_long":
            extra += f";concepts={','.join(sorted(concepts))}"
        target = self.interim_root / self.dataset_name / f"{table}.parquet"
        key = build_cache_key(
            source_paths=self.cache_source_paths(table),
            tree_roots=[self.repo_root / "critical_mm", self.repo_root / "configs"],
            repo_root=self.repo_root,
            extra=extra,
        )
        if not force:
            hit = cache_lookup(target, key)
            if hit is not None:
                return hit
        df = self._dispatch_read(table, concepts)
        validate_frame(df, table)

        def _writer(tmp_path: Path) -> None:
            df.sink_parquet(tmp_path, compression="zstd", statistics=True)

        cache_write(target, key, _writer)
        return target

    def _dispatch_read(self, table: str, concepts: list[str]) -> pl.LazyFrame:
        if table == "stays":
            return self.read_stays()
        if table == "events_long":
            return self.read_events_long(concepts)
        if table == "meds":
            return self.read_meds()
        if table == "interventions":
            return self.read_interventions()
        if table == "notes":
            return self.read_notes()
        if table == "diagnoses":
            return self.read_diagnoses()
        if table == "microbio":
            return self.read_microbio()
        if table == "abx_duration":
            return self.read_abx_duration()
        raise ValueError(f"unknown table {table!r}; valid: {_TABLE_NAMES}")
