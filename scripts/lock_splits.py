"""Lock train/val/test splits to disk for reproducible multi-seed training.

ricu/YAIB's `make_single_split` (reproductions/yaib_pinned/icu_benchmarks/data/
split_process_data.py:107) uses StratifiedKFold(cv_repetitions, shuffle=True,
random_state=seed) for the outer (test/dev) split and StratifiedKFold(cv_folds)
for the inner (train/val) split. With seed=42 the splits are deterministic
*per data version*, but the original YAIB trainer coupled this seed to model
initialisation. To vary model-init seed (for 5-seed runs) without changing
the data split, we precompute the splits ONCE at seed=42 and the CM-native
trainer (`critical_mm.training.train.train_one`) loads them at training time.

Layout written by this script:

    data/processed/splits/<task>/<dataset>/
        manifest.json                       # data git_sha, fingerprints, schema
        cv_rep_<r>_fold_<f>.parquet        # one parquet per (rep, fold) pair
                                             with columns [stay_id, split]
                                             split ∈ {train, val, test}

Manifest includes per-fold content_sha256 so training scripts can verify the
splits match what the checkpoint was trained against.

Usage:
    python scripts/lock_splits.py                              # all tasks x datasets
    python scripts/lock_splits.py --tasks sepsis aki           # subset
    python scripts/lock_splits.py --datasets eicu miiv         # subset
    python scripts/lock_splits.py --seed 42 --cv-reps 5 --cv-folds 5    # explicit

The default cv_reps=5, cv_folds=5 matches YAIB's `execute_repeated_cv`
defaults; the default seed=42 matches YAIB's gin config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
from sklearn.model_selection import KFold, StratifiedKFold

REPO = Path("$CRITICAL_MM_REPO")

_DEFAULT_TASKS: tuple[str, ...] = ("sepsis", "aki", "mortality24", "los", "kidney_function")
_DEFAULT_DATASETS: tuple[str, ...] = ("eicu", "miiv", "hirid", "nwicu")

_CLASSIFICATION_TASKS: frozenset[str] = frozenset({"sepsis", "aki", "mortality24"})
_REGRESSION_TASKS: frozenset[str] = frozenset({"los", "kidney_function"})

def _git_sha(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"

def _outc_path(task: str, dataset: str) -> Path:
    return REPO / "data" / "processed" / task / dataset / "outc.parquet"

def _splits_dir(task: str, dataset: str) -> Path:
    return REPO / "data" / "processed" / "splits" / task / dataset

def _content_sha256(parquet_path: Path) -> str:
    h = hashlib.sha256()
    h.update(parquet_path.read_bytes())
    return h.hexdigest()

def _lock_one(
    task: str,
    dataset: str,
    seed: int,
    cv_reps: int,
    cv_folds: int,
) -> dict[str, object]:
    """Compute and write all (rep, fold) splits for one (task, dataset)."""
    outc_path = _outc_path(task, dataset)
    if not outc_path.exists():
        return {"status": "skipped", "reason": f"{outc_path} missing"}

    splits_dir = _splits_dir(task, dataset)
    splits_dir.mkdir(parents=True, exist_ok=True)

    outc = pl.read_parquet(outc_path)
    label_col = "label_value" if "label_value" in outc.columns else "label"

    stay_labels = (
        outc.group_by("stay_id").agg(pl.col(label_col).max().alias("_label")).sort("stay_id")
    )
    stays = stay_labels["stay_id"].to_numpy()
    labels = stay_labels["_label"].to_numpy()

    is_classification = task in _CLASSIFICATION_TASKS
    if is_classification:
        outer_cv = StratifiedKFold(cv_reps, shuffle=True, random_state=seed)
        inner_cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
        outer_splits = list(outer_cv.split(stays, labels))
    else:
        outer_cv = KFold(cv_reps, shuffle=True, random_state=seed)
        inner_cv = KFold(cv_folds, shuffle=True, random_state=seed)
        outer_splits = list(outer_cv.split(stays))

    per_fold: list[dict[str, object]] = []
    for rep_idx, (dev_idx, test_idx) in enumerate(outer_splits):
        dev_stays = stays[dev_idx]
        dev_labels = labels[dev_idx] if is_classification else None
        if is_classification:
            inner_splits = list(inner_cv.split(dev_stays, dev_labels))
        else:
            inner_splits = list(inner_cv.split(dev_stays))
        for fold_idx, (train_idx, val_idx) in enumerate(inner_splits):
            train_stays = dev_stays[train_idx]
            val_stays = dev_stays[val_idx]
            test_stays = stays[test_idx]

            stay_dtype = outc.schema["stay_id"]
            split_df = pl.DataFrame(
                {
                    "stay_id": pl.concat(
                        [
                            pl.Series("stay_id", train_stays.tolist(), dtype=stay_dtype),
                            pl.Series("stay_id", val_stays.tolist(), dtype=stay_dtype),
                            pl.Series("stay_id", test_stays.tolist(), dtype=stay_dtype),
                        ]
                    ),
                    "split": pl.concat(
                        [
                            pl.Series("split", ["train"] * len(train_stays), dtype=pl.Utf8),
                            pl.Series("split", ["val"] * len(val_stays), dtype=pl.Utf8),
                            pl.Series("split", ["test"] * len(test_stays), dtype=pl.Utf8),
                        ]
                    ),
                }
            )
            out_path = splits_dir / f"cv_rep_{rep_idx}_fold_{fold_idx}.parquet"
            split_df.write_parquet(out_path, compression="zstd")
            per_fold.append(
                {
                    "cv_rep": rep_idx,
                    "fold": fold_idx,
                    "path": str(out_path.relative_to(REPO)),
                    "n_train": len(train_stays),
                    "n_val": len(val_stays),
                    "n_test": len(test_stays),
                    "content_sha256": _content_sha256(out_path),
                }
            )

    manifest = {
        "task": task,
        "dataset": dataset,
        "n_stays_total": len(stays),
        "n_positive_stays": int(labels.sum()) if is_classification else None,
        "split_seed": seed,
        "cv_reps": cv_reps,
        "cv_folds": cv_folds,
        "classification": is_classification,
        "data_git_sha": _git_sha(REPO),
        "outc_path": str(outc_path.relative_to(REPO)),
        "outc_content_sha256": _content_sha256(outc_path),
        "folds": per_fold,
    }
    (splits_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"status": "ok", "manifest": manifest}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+", default=list(_DEFAULT_TASKS))
    ap.add_argument("--datasets", nargs="+", default=list(_DEFAULT_DATASETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-reps", type=int, default=5)
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()

    print(
        f"=== Locking splits: tasks={args.tasks} datasets={args.datasets} "
        f"seed={args.seed} cv_reps={args.cv_reps} cv_folds={args.cv_folds} ==="
    )
    print(f"data_git_sha: {_git_sha(REPO)}")
    print()

    total_start = time.perf_counter()
    n_ok = 0
    n_skipped = 0
    for task in args.tasks:
        for dataset in args.datasets:
            t0 = time.perf_counter()
            result = _lock_one(
                task=task,
                dataset=dataset,
                seed=args.seed,
                cv_reps=args.cv_reps,
                cv_folds=args.cv_folds,
            )
            dt = time.perf_counter() - t0
            if result["status"] == "ok":
                m = result["manifest"]
                n_ok += 1
                print(
                    f"  {task:18s} {dataset:6s} ok in {dt:.1f}s "
                    f"(n_stays={m['n_stays_total']:6d}, "
                    f"n_folds={len(m['folds'])})"
                )
            else:
                n_skipped += 1
                print(f"  {task:18s} {dataset:6s} SKIP — {result['reason']}")
    print()
    print(
        f"=== Done in {time.perf_counter() - total_start:.1f}s ({n_ok} ok, {n_skipped} skipped) ==="
    )
    sys.exit(0 if n_skipped == 0 else 1)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

