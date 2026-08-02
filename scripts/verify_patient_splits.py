"""Verify patient-grouped splits: zero leakage + stay conservation vs published."""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from critical_mm.patient_splits import (
    check_no_patient_leakage,
    check_stay_conservation,
    resolve_stay_to_patient,
)

REPO = Path("$CRITICAL_MM_REPO")

def verify_one(cohort: str, dataset: str, *, repo: Path = REPO) -> dict:
    new_dir = repo / "data/processed/splits_patient" / cohort / dataset
    pub_dir = repo / "data/processed/splits" / cohort / dataset
    interim = repo / "data/interim" / dataset / "stays.parquet"
    patient_folds = sorted(new_dir.glob("cv_rep_*_fold_*.parquet")) if new_dir.exists() else []
    if not patient_folds:
        pub_folds = sorted(pub_dir.glob("cv_rep_*_fold_*.parquet")) if pub_dir.exists() else []
        if pub_folds:
            return {
                "status": "gap",
                "reasons": [
                    "published splits exist but no patient splits "
                    "(source outc.parquet absent — coverage gap)"
                ],
            }
        return {"status": "skipped", "reasons": [f"no patient splits at {new_dir}"]}
    interim_df = pl.read_parquet(interim)
    reasons: list[str] = []
    for f in patient_folds:
        frame = pl.read_parquet(f)
        try:
            stp = resolve_stay_to_patient(frame["stay_id"].unique(), interim_df, dataset)
        except ValueError as e:
            reasons.append(f"{f.name}: resolver coverage fail: {e}")
            continue
        leaks = check_no_patient_leakage(frame, stp)
        if leaks:
            reasons.append(f"{f.name}: {len(leaks)} patients leak across splits")
        pub = pub_dir / f.name
        if pub.exists():
            pub_frame = pl.read_parquet(pub)
            if not check_stay_conservation(frame, pub_frame):
                reasons.append(f"{f.name}: stay-set differs from published")
    return {"status": "fail" if reasons else "pass", "reasons": reasons}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()
    n_fail = n_gap = 0
    for cohort in args.cohorts:
        for dataset in args.datasets:
            r = verify_one(cohort, dataset)
            tag = r["status"].upper()
            print(f"  {cohort:18s} {dataset:6s} {tag}" + (f" — {r['reasons']}" if r["reasons"] else ""))
            n_fail += r["status"] == "fail"
            n_gap += r["status"] == "gap"
    print(f"=== {n_fail} failing cells, {n_gap} coverage gaps ===")
    raise SystemExit(1 if (n_fail or n_gap) else 0)

if __name__ == "__main__":
    main()
