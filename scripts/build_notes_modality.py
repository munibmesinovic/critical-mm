"""Materialize notes-modality timed frames + per-task aligned views + coverage.

Outputs (gitignored, under data/processed/_modalities/):
  <ds>/notes_timed.parquet
  <ds>/notes/aligned_<task>.parquet
  _notes_coverage.json

Run under mem_run.sh: bash scripts/run_notes_modality.sh
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from critical_mm.modalities.notes import EicuNoteReader, MiivNoteReader, OmixNoteReader
from critical_mm.modalities.notes_build import build_notes_aligned_view, notes_coverage_row

REPO = Path(__file__).resolve().parent.parent
TASKS = ["mortality24", "aki", "sepsis", "los", "kidney_function"]
READERS = {"miiv": MiivNoteReader, "omix": OmixNoteReader, "eicu": EicuNoteReader}


def _cohort_with_hadm(ds: str, stays: pl.LazyFrame) -> pl.LazyFrame:
    """Attach each stay's own hadm_id (miiv via icustays); null for stay-bound datasets."""
    if ds != "miiv":
        return stays.with_columns(pl.lit(None, dtype=pl.Utf8).alias("hadm_id"))
    ic = pl.scan_csv(
        REPO / "data/raw/mimic-iv-3.1/icu/icustays.csv.gz",
        infer_schema_length=5000,
        ignore_errors=True,
    ).select(
        pl.col("stay_id").cast(pl.Utf8).alias("__sid"),
        pl.col("hadm_id").cast(pl.Utf8).alias("hadm_id"),
    )
    return (
        stays.with_columns(pl.col("stay_id").cast(pl.Utf8).str.slice(5).alias("__sid"))
        .join(ic, on="__sid", how="left")
        .drop("__sid")
    )


def main() -> None:
    out_root = REPO / "data" / "processed" / "_modalities"
    coverage: list[dict[str, object]] = []
    for ds, reader_cls in READERS.items():
        ds_dir = out_root / ds
        (ds_dir / "notes").mkdir(parents=True, exist_ok=True)
        timed = reader_cls(repo_root=REPO).read_timed(ds).collect(engine="streaming")
        timed.write_parquet(ds_dir / "notes_timed.parquet")
        stays_p = REPO / "data" / "processed" / "base_cohort" / ds / "stays.parquet"
        if not stays_p.exists():
            continue
        stays = _cohort_with_hadm(ds, pl.read_parquet(stays_p).lazy())
        for task in TASKS:
            split_p = REPO / "data/processed/splits" / task / ds / "cv_rep_0_fold_0.parquet"
            if not split_p.exists():
                continue
            split_df = pl.read_parquet(split_p)
            n_cohort = split_df["stay_id"].n_unique()
            view = build_notes_aligned_view(timed.lazy(), stays, split_df.lazy(), ds).collect()
            view.write_parquet(ds_dir / "notes" / f"aligned_{task}.parquet")
            coverage.append(notes_coverage_row(task, ds, view, n_cohort))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "_notes_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(f"wrote {len(coverage)} aligned notes views -> {out_root / '_notes_coverage.json'}")


if __name__ == "__main__":
    main()
