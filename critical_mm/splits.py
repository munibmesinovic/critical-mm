"""Patient-grouped K-fold splitters for CRITICAL-MM cohorts.

The hard invariant: a patient never appears in both train and val of
the same fold. sklearn's index-based `KFold` gives wrong answers on
events_long-style frames where one patient contributes many rows;
this module operates on unique `patient_id` strings (the canonical
CRITICAL-MM identifier) so the invariant holds regardless of row
multiplicity.

`fit_string()` returns a deterministic JSON encoding of the splitter
settings; the cache layer () mixes it into the cache key so a
change of n_splits/random_state invalidates downstream artefacts.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

import polars as pl

def _contiguous_chunks(items: list[str], n_splits: int) -> list[list[str]]:
    """Split `items` into `n_splits` near-equal contiguous chunks.

    Mirrors sklearn's KFold semantics: the first `len % n_splits`
    chunks receive one extra element.
    """
    n = len(items)
    base, extra = divmod(n, n_splits)
    chunks: list[list[str]] = []
    start = 0
    for k in range(n_splits):
        size = base + (1 if k < extra else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks

class PatientGroupedKFold:
    """K-fold over unique patient_ids.

    Each patient appears in exactly one fold's val set; the remaining
    folds' patients form the train set. Operates on
    `LazyFrame.select("patient_id").unique()`, not on row indices.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: pl.LazyFrame) -> Iterator[tuple[list[str], list[str]]]:
        """Yield (train_patient_ids, val_patient_ids) for each fold."""
        pids = sorted(X.select("patient_id").unique().collect()["patient_id"].to_list())
        if len(pids) < self.n_splits:
            raise ValueError(
                f"n_splits ({self.n_splits}) exceeds the number of unique patients "
                f"({len(pids)}); folds would be empty"
            )
        if self.shuffle:
            rng = random.Random(self.random_state)
            rng.shuffle(pids)
        chunks = _contiguous_chunks(pids, self.n_splits)
        for k in range(self.n_splits):
            val = chunks[k]
            train: list[str] = []
            for j, chunk in enumerate(chunks):
                if j != k:
                    train.extend(chunk)
            yield train, val

    def fit_string(self) -> str:
        """JSON encoding of the splitter settings — used as a cache key."""
        return json.dumps(
            {
                "n_splits": self.n_splits,
                "shuffle": self.shuffle,
                "random_state": self.random_state,
            },
            sort_keys=True,
        )

class RepeatedStratifiedGroupKFold:
    """Repeated stratified group K-fold.

    Stratifies on a binary outcome label provided per patient, while
    preserving the patient-grouping invariant. `labels` is a
    LazyFrame with columns (`patient_id`, `label`) — one row per
    patient. Yields `n_repeats × n_splits` folds in total.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(
        self,
        X: pl.LazyFrame,
        labels: pl.LazyFrame,
    ) -> Iterator[tuple[list[str], list[str]]]:
        lab_df = labels.select("patient_id", pl.col("label").cast(pl.Int8)).collect()
        if lab_df["patient_id"].n_unique() != lab_df.height:
            raise ValueError("labels must contain exactly one row per patient_id; got duplicates")
        distinct = set(lab_df["label"].unique().drop_nulls().to_list())
        if not distinct.issubset({0, 1}):
            raise ValueError(
                f"labels must be binary (0/1) for stratification; got distinct values "
                f"{sorted(distinct)}"
            )
        x_pids = set(X.select("patient_id").unique().collect()["patient_id"].to_list())
        lab_pids = set(lab_df["patient_id"].to_list())
        if x_pids != lab_pids:
            raise ValueError(
                f"patient_id sets in X and labels must match exactly; "
                f"{len(x_pids - lab_pids)} in X but not labels, "
                f"{len(lab_pids - x_pids)} in labels but not X"
            )
        pos_pids = sorted(lab_df.filter(pl.col("label") == 1)["patient_id"].to_list())
        neg_pids = sorted(lab_df.filter(pl.col("label") == 0)["patient_id"].to_list())
        if len(pos_pids) < self.n_splits or len(neg_pids) < self.n_splits:
            raise ValueError(
                f"outcome too sparse for stratification: positives={len(pos_pids)}, "
                f"negatives={len(neg_pids)}, n_splits={self.n_splits}"
            )
        for repeat in range(self.n_repeats):
            rng = random.Random(self.random_state + repeat)
            pos_shuffled = list(pos_pids)
            neg_shuffled = list(neg_pids)
            rng.shuffle(pos_shuffled)
            rng.shuffle(neg_shuffled)
            pos_folds = _contiguous_chunks(pos_shuffled, self.n_splits)
            neg_folds = _contiguous_chunks(neg_shuffled, self.n_splits)
            for k in range(self.n_splits):
                val = pos_folds[k] + neg_folds[k]
                train: list[str] = []
                for j in range(self.n_splits):
                    if j != k:
                        train.extend(pos_folds[j])
                        train.extend(neg_folds[j])
                yield train, val

    def fit_string(self) -> str:
        return json.dumps(
            {
                "n_splits": self.n_splits,
                "n_repeats": self.n_repeats,
                "random_state": self.random_state,
            },
            sort_keys=True,
        )

def write_fold_assignments(
    *,
    folds: Iterator[tuple[list[str], list[str]]],
    output_dir: Path,
    scheme: str,
) -> list[Path]:
    """Write each fold's train/val patient_id lists to `output_dir/fold_<k>.json`.

    Schema per file:
        {"fold": k, "scheme": scheme, "n_splits": total,
         "train_patient_ids": [...], "val_patient_ids": [...]}

    `n_splits` is the total number of folds written (for repeated
    schemes this equals n_splits * n_repeats).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    materialised = list(folds)
    n_total = len(materialised)
    paths: list[Path] = []
    for k, (train, val) in enumerate(materialised):
        path = output_dir / f"fold_{k}.json"
        payload = {
            "fold": k,
            "scheme": scheme,
            "n_splits": n_total,
            "train_patient_ids": list(train),
            "val_patient_ids": list(val),
        }
        path.write_text(json.dumps(payload))
        paths.append(path)
    return paths

__all__ = [
    "PatientGroupedKFold",
    "RepeatedStratifiedGroupKFold",
    "write_fold_assignments",
]
