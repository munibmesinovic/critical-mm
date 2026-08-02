"""Patient-grouped split machinery (the initial split work).

Splits on unique patients then expands to each patient's COHORT stays, matching
scripts/lock_splits.py's nested 5x5 structure one level up.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import KFold, StratifiedKFold

def resolve_stay_to_patient(
    cohort_stays: pl.Series, interim_stays: pl.DataFrame, dataset: str
) -> pl.DataFrame:
    """Map each cohort stay_id to its patient_id via interim stays.

    Dtype-driven: integer cohort ids join to prefix-stripped interim ids; string
    cohort ids join directly on the prefixed interim stay_id. Returns
    [stay_id (native dtype), patient_id]; raises if any stay is unresolved.
    """
    native_dtype = cohort_stays.dtype
    interim = interim_stays.select(
        "patient_id", pl.col("stay_id").cast(pl.Utf8).alias("_istay")
    )
    if native_dtype.is_integer():
        interim = interim.with_columns(
            pl.col("_istay").str.replace(f"^{dataset}_", "").cast(pl.Int64).alias("_key")
        )
        cohort = pl.DataFrame({"stay_id": cohort_stays}).with_columns(
            pl.col("stay_id").cast(pl.Int64).alias("_key")
        )
    else:
        interim = interim.with_columns(pl.col("_istay").alias("_key"))
        cohort = pl.DataFrame({"stay_id": cohort_stays.cast(pl.Utf8)}).with_columns(
            pl.col("stay_id").alias("_key")
        )
    out = cohort.join(interim.select("_key", "patient_id"), on="_key", how="left").select(
        "stay_id", "patient_id"
    )
    n_unresolved = out["patient_id"].is_null().sum()
    if n_unresolved:
        raise ValueError(
            f"resolver: {n_unresolved} unresolved stay_id for dataset={dataset}"
        )
    return out.with_columns(pl.col("stay_id").cast(native_dtype))

def patient_two_level_folds(
    patients: list[str],
    patient_labels: list[int] | None,
    *,
    cv_reps: int,
    cv_folds: int,
    seed: int,
    is_classification: bool,
) -> list[tuple[int, int, list[str], list[str], list[str]]]:
    """Nested 5x5 split over UNIQUE patients (mirrors lock_splits._lock_one one
    level up). Patients are sorted before sklearn for determinism."""
    pats = np.array(sorted(patients))
    if is_classification:
        lab_map = dict(zip(patients, patient_labels, strict=True))
        labels = np.array([lab_map[p] for p in pats])
        outer = StratifiedKFold(cv_reps, shuffle=True, random_state=seed)
        inner = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
        outer_splits = list(outer.split(pats, labels))
    else:
        labels = None
        outer = KFold(cv_reps, shuffle=True, random_state=seed)
        inner = KFold(cv_folds, shuffle=True, random_state=seed)
        outer_splits = list(outer.split(pats))

    out: list[tuple[int, int, list[str], list[str], list[str]]] = []
    for rep, (dev_idx, test_idx) in enumerate(outer_splits):
        dev_p, test_p = pats[dev_idx], pats[test_idx]
        if is_classification:
            inner_splits = list(inner.split(dev_p, labels[dev_idx]))
        else:
            inner_splits = list(inner.split(dev_p))
        for fold, (tr_idx, val_idx) in enumerate(inner_splits):
            out.append(
                (rep, fold, dev_p[tr_idx].tolist(), dev_p[val_idx].tolist(), test_p.tolist())
            )
    return out

def build_split_frame(
    fold_train: list[str],
    fold_val: list[str],
    fold_test: list[str],
    stay_to_patient: pl.DataFrame,
) -> pl.DataFrame:
    """Expand each patient set to its COHORT stays (from stay_to_patient only),
    emit [stay_id, split] in native stay_id dtype, sorted for byte-stability."""
    parts = []
    for split_name, pats in (("train", fold_train), ("val", fold_val), ("test", fold_test)):
        stays = (
            stay_to_patient.filter(pl.col("patient_id").is_in(pats))
            .select("stay_id")
            .with_columns(pl.lit(split_name).alias("split"))
        )
        parts.append(stays)
    return pl.concat(parts).sort("split", "stay_id")

def check_no_patient_leakage(
    split_frame: pl.DataFrame, stay_to_patient: pl.DataFrame
) -> list[str]:
    """Return patient_ids that appear in more than one split (empty = clean)."""
    j = split_frame.join(stay_to_patient, on="stay_id", how="left")
    span = (
        j.group_by("patient_id")
        .agg(pl.col("split").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
    )
    return sorted(span["patient_id"].to_list())

def check_stay_conservation(
    new_frame: pl.DataFrame, published_frame: pl.DataFrame
) -> bool:
    """True iff new and published frames cover the identical set of stay_id."""
    return set(new_frame["stay_id"].cast(pl.Utf8).to_list()) == set(
        published_frame["stay_id"].cast(pl.Utf8).to_list()
    )
