"""Materialize diagnosis-modality timed frames + per-task aligned views + coverage.

Outputs (all gitignored under data/processed/_modalities/):
  <ds>/diagnoses_timed.parquet
  <ds>/diagnoses/aligned_<task>.parquet
  _coverage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from critical_mm.modalities.build import build_aligned_view, coverage_row
from critical_mm.modalities.diagnoses import DiagnosesModalityReader

REPO = Path(__file__).resolve().parent.parent
DATASETS = ["miiv", "eicu", "nwicu", "omix", "sicdb"]
TASKS = ["mortality24", "aki", "sepsis", "los", "kidney_function"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()
    run_datasets = list(args.datasets)

    reader = DiagnosesModalityReader(repo_root=REPO)
    out_root = REPO / "data" / "processed" / "_modalities"
    out_root.mkdir(parents=True, exist_ok=True)
    coverage: list[dict[str, object]] = []
    for ds in run_datasets:
        ds_dir = out_root / ds
        (ds_dir / "diagnoses").mkdir(parents=True, exist_ok=True)
        timed = reader.read_timed(ds).collect()
        timed.write_parquet(ds_dir / "diagnoses_timed.parquet")
        stays_p = REPO / "data" / "processed" / "base_cohort" / ds / "stays.parquet"
        if not stays_p.exists():
            continue
        stays = pl.read_parquet(stays_p).lazy()
        for task in TASKS:
            split_glob = sorted(
                (REPO / "data" / "processed" / "splits" / task / ds).glob("cv_rep_0_fold_0.parquet")
            )
            if not split_glob:
                continue
            split_df = pl.read_parquet(split_glob[0])
            n_cohort_stays = split_df["stay_id"].n_unique()
            view = build_aligned_view(timed.lazy(), stays, split_df.lazy(), ds).collect()
            view.write_parquet(ds_dir / "diagnoses" / f"aligned_{task}.parquet")
            coverage.append(coverage_row(task, ds, view, n_cohort_stays))

    cov_path = out_root / "_coverage.json"
    existing: list[dict[str, object]] = []
    if cov_path.exists():
        existing = [
            r for r in json.loads(cov_path.read_text()) if r.get("dataset") not in set(run_datasets)
        ]
    merged = existing + coverage
    cov_path.write_text(json.dumps(merged, indent=2))
    print(
        f"wrote {len(coverage)} aligned views for {run_datasets}; "
        f"coverage rows total {len(merged)} -> {cov_path}"
    )

if __name__ == "__main__":
    main()
