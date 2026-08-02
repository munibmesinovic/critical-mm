"""Lock PATIENT-GROUPED train/val/test splits (the initial split work).

Mirrors scripts/lock_splits.py but splits on unique patients then expands to each
patient's cohort stays. Writes data/processed/splits_patient/<cohort>/<dataset>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import polars as pl

from critical_mm.validation.cohort_fingerprint import cohort_fingerprint
from critical_mm.patient_splits import (
    build_split_frame,
    patient_two_level_folds,
    resolve_stay_to_patient,
)

REPO = Path("$CRITICAL_MM_REPO")
_CLASSIFICATION = {
    "mortality24", "sepsis", "aki", "mortality24_leaky", "mortality24_nostat", "sepsis_simple",
}

def _git_sha(repo: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"

def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def lock_one_patient(
    cohort: str, dataset: str, *, repo: Path = REPO, seed: int = 42, cv_reps: int = 5, cv_folds: int = 5
) -> dict:
    outc_path = repo / "data/processed" / cohort / dataset / "outc.parquet"
    interim_path = repo / "data/interim" / dataset / "stays.parquet"
    if not outc_path.exists() or not interim_path.exists():
        return {"status": "skipped", "reason": f"missing {outc_path if not outc_path.exists() else interim_path}"}

    outc = pl.read_parquet(outc_path)
    label_col = "label_value" if "label_value" in outc.columns else "label"
    is_cls = cohort in _CLASSIFICATION

    cohort_stays = outc.select("stay_id").unique()["stay_id"]
    interim = pl.read_parquet(interim_path)
    stp = resolve_stay_to_patient(cohort_stays, interim, dataset)

    if is_cls:
        stay_label = outc.group_by("stay_id").agg(pl.col(label_col).max().alias("_l"))
        pat = (
            stp.join(stay_label, on="stay_id", how="left")
            .group_by("patient_id")
            .agg(pl.col("_l").max().alias("_l"))
            .sort("patient_id")
        )
        patients = pat["patient_id"].to_list()
        labels = pat["_l"].cast(pl.Int64).to_list()
        n_positive = int(sum(labels))
    else:
        patients = sorted(stp["patient_id"].unique().to_list())
        labels = None
        n_positive = None

    folds = patient_two_level_folds(
        patients, labels, cv_reps=cv_reps, cv_folds=cv_folds, seed=seed, is_classification=is_cls
    )

    out_dir = repo / "data/processed/splits_patient" / cohort / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold = []
    for rep, fold, tr, val, te in folds:
        frame = build_split_frame(tr, val, te, stp)
        path = out_dir / f"cv_rep_{rep}_fold_{fold}.parquet"
        frame.write_parquet(path, compression="zstd")
        vc = dict(frame["split"].value_counts().iter_rows())
        per_fold.append({
            "cv_rep": rep, "fold": fold, "path": str(path.relative_to(repo)),
            "n_train": vc.get("train", 0), "n_val": vc.get("val", 0), "n_test": vc.get("test", 0),
            "content_sha256": _sha256(path),
        })

    manifest = {
        "cohort": cohort, "dataset": dataset, "grouping": "patient",
        "n_patients": len(patients),
        "n_stays_total": cohort_stays.len(), "n_stays": cohort_stays.len(),
        "n_positive": n_positive,
        "classification": is_cls, "split_seed": seed, "cv_reps": cv_reps, "cv_folds": cv_folds,
        "data_git_sha": _git_sha(repo),
        **cohort_fingerprint(outc_path.parent, require_all=False),
        "outc_content_sha256": _sha256(outc_path),
        "folds": per_fold,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"status": "ok", "manifest": manifest}

_DEFAULT_COHORTS = (
    "mortality24", "sepsis", "aki", "los", "kidney_function",
    "mortality24_leaky", "mortality24_nostat", "sepsis_simple",
)
_DEFAULT_DATASETS = ("eicu", "miiv", "hirid", "nwicu", "omix", "sicdb", "zigong")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohorts", nargs="+", default=list(_DEFAULT_COHORTS))
    ap.add_argument("--datasets", nargs="+", default=list(_DEFAULT_DATASETS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n_ok = n_skip = 0
    for cohort in args.cohorts:
        for dataset in args.datasets:
            res = lock_one_patient(cohort, dataset, seed=args.seed)
            if res["status"] == "ok":
                n_ok += 1
                m = res["manifest"]
                print(f"  {cohort:18s} {dataset:6s} ok  patients={m['n_patients']:6d} stays={m['n_stays']:6d}")
            else:
                n_skip += 1
                print(f"  {cohort:18s} {dataset:6s} SKIP — {res['reason']}")
    print(f"=== done ({n_ok} ok, {n_skip} skipped) ===")

if __name__ == "__main__":
    main()
