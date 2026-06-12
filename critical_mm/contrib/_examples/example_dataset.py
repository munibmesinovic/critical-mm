"""Template: a minimal new DatasetReader.

SCAFFOLD -- not auto-registered. See ``critical_mm/contrib/_examples/
example_task.py`` for the activation workflow.

The full DatasetReader contract is documented in
``docs/extending/datasets.md``. In short, a reader produces six
canonical interim parquets (the seventh + eighth optional ones, microbio
and abx_duration, default to empty).
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from critical_mm.api import DatasetReader
from critical_mm.schema import TABLES

class _ExampleSyntheticReader(DatasetReader):
    """Empty reader returning schema-correct empty frames.

    Real implementations:
      - Set ``DATASET_NAME`` ClassVar OR override ``dataset_name`` property
        to return a unique short identifier (used by paths, registry keys,
        stay_id prefixes).
      - Implement each ``read_*`` to scan raw CSVs / parquets from
        ``self.raw_root`` and return a polars LazyFrame matching the
        TABLES[<name>][0] schema in ``critical_mm/schema.py``.
      - Optionally override ``cache_source_paths(table)`` to return the
        per-table subset of raw files. The default (every file under
        raw_root) is correct but coarse: any change anywhere in your
        dataset invalidates every interim table.
    """

    DATASET_NAME: ClassVar[str] = "_example_synthetic"

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    def read_stays(self) -> pl.LazyFrame:
        return pl.LazyFrame(schema=TABLES["stays"][0])

    def read_events_long(self, concepts: list[str]) -> pl.LazyFrame:
        del concepts
        return pl.LazyFrame(schema=TABLES["events_long"][0])

    def read_meds(self) -> pl.LazyFrame:
        return pl.LazyFrame(schema=TABLES["meds"][0])

    def read_interventions(self) -> pl.LazyFrame:
        return pl.LazyFrame(schema=TABLES["interventions"][0])

    def read_notes(self) -> pl.LazyFrame:
        return pl.LazyFrame(schema=TABLES["notes"][0])

    def read_diagnoses(self) -> pl.LazyFrame:
        return pl.LazyFrame(schema=TABLES["diagnoses"][0])

    def read_microbio(self) -> pl.LazyFrame:
        return pl.LazyFrame(schema=TABLES["microbio"][0])

