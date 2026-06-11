"""Modality feature-block builders for fusion.

Each builder returns a polars DataFrame keyed on the int-hashed ``stay_id``
(identical to the training preamble's hashing) plus value columns and a
``<modality>_present`` bit. ICD is stay-level (broadcast at join time); notes
are per-(stay,hour) for per-hour tasks and stay-level otherwise.

Train-only fitting: the ICD vocabulary and the notes PCA are derived from the
TRAIN stays only (the caller passes the hashed train stay-id set), so no
val/test statistics leak into the representation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import numpy as np
    from sklearn.decomposition import PCA


def _hash_int_expr(col: str) -> pl.Expr:
    """Vectorized form of ``train.hash_stay_id``'s fast path.

    Takes the suffix after the last ``_`` (if any), casts to Int64, and reduces
    mod 2**31. Returns null where the suffix is non-numeric so the caller can
    fall back to the Python hash for those (rare) rows. This MUST stay in lockstep
    with ``hash_stay_id_series`` / ``train.hash_stay_id`` — the single source of
    truth for the numeric branch.
    """
    s = pl.col(col).cast(pl.Utf8)
    suffix = (
        pl.when(s.str.contains("_", literal=True)).then(s.str.split("_").list.last()).otherwise(s)
    )
    return (suffix.cast(pl.Int64, strict=False) % (2**31)).alias(col)


def hash_stay_id_series(s: pl.Series) -> pl.Series:
    """Replicate train.hash_stay_id over a polars Series (suffix-int, else hash).

    Vectorized numeric branch; per-element Python fallback ONLY for rows whose
    suffix is non-numeric. Identical output to ``train.hash_stay_id`` for every
    input. This and ``_hash_stay_id_expr`` share the one numeric implementation.
    """
    import critical_mm.training.train as t

    df = pl.DataFrame({"stay_id": s}).with_columns(_hash_int_expr("stay_id").alias("__h"))
    fast = df["__h"]
    if fast.null_count() == 0:
        return fast.rename(s.name)
    src = s.to_list()
    filled = [
        int(v) if v is not None else t.hash_stay_id(orig)
        for v, orig in zip(fast.to_list(), src, strict=True)
    ]
    return pl.Series(s.name, filled, dtype=pl.Int64)


def _hash_stay_id_expr(col: str = "stay_id") -> pl.Expr:
    """Expression form of the stay_id hash for use inside ``with_columns``.

    Delegates to the same vectorized numeric branch as ``hash_stay_id_series``;
    for the (rare) non-numeric-suffix rows it falls back to the Python
    ``train.hash_stay_id`` via ``map_elements`` on those rows only.
    """
    import critical_mm.training.train as t

    fast = _hash_int_expr(col)
    return (
        pl.when(fast.is_not_null())
        .then(fast)
        .otherwise(pl.col(col).map_elements(t.hash_stay_id, return_dtype=pl.Int64))
        .alias(col)
    )


ICD_GROUPS: tuple[str, ...] = ("ccsr", "icd10_root")


def _icd_groups(aligned: pl.DataFrame, group: str) -> pl.DataFrame:
    """Add a ``group`` column (CCSR or icd10_root) to a string-stay_id frame.

    Fast-fails on an unrecognised ``group`` rather than silently coercing it to
    ``icd10_root`` (a foot-gun for the ICD-ablation sweep, where a typo'd group
    name would otherwise produce a valid-looking but wrong vocabulary).
    """
    from critical_mm.modalities.grouping import add_grouping

    if group not in ICD_GROUPS:
        raise ValueError(f"unknown icd_group {group!r}; valid: {ICD_GROUPS}")
    return add_grouping(aligned.lazy(), rep=group).collect()


def _maybe_pca(
    block: pl.DataFrame, vocab: list[str], train_stay_ids: set[int], pca_dim: int | None
) -> pl.DataFrame:
    if pca_dim is None or not vocab:
        return block
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    val_cols = [f"icd__{g}" for g in vocab]
    mat = block.select(val_cols).to_numpy().astype(np.float32)
    train_mask = block["stay_id"].is_in(list(train_stay_ids)).to_numpy()
    n_train = int(train_mask.sum())
    n_comp = max(1, min(pca_dim, n_train if n_train else mat.shape[0], mat.shape[1]))
    pca = PCA(n_components=n_comp, random_state=0)
    scaler = StandardScaler()
    train_mat = mat[train_mask] if n_train > 0 else mat
    pca.fit(train_mat)
    scaler.fit(pca.transform(train_mat))
    reduced = scaler.transform(pca.transform(mat)).astype(np.float32)
    return block.select("stay_id", "icd_present").with_columns(
        *[pl.Series(f"icd__pca_{i}", reduced[:, i]) for i in range(n_comp)]
    )


def build_icd_block(
    aligned: pl.DataFrame,
    *,
    train_stay_ids: set[int],
    group: str = "ccsr",
    min_prevalence: int = 25,
    top_k: int | None = None,
    representation: str = "multihot",
    pca_dim: int | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Per-stay multi-hot over a train-only, prevalence-filtered group vocabulary.

    Returns (block, vocab). block columns: stay_id (Int64), icd__<group>... (f32
    0/1), icd_present (f32 0/1). One row per stay that carries >=1 code. When the
    vocabulary is empty (every group below ``min_prevalence``) the block has only
    stay_id + icd_present and ``vocab == []``.

    When ``top_k`` is not None, the vocabulary is additionally capped (AFTER the
    ``min_prevalence`` floor) to the ``top_k`` MOST prevalent groups by distinct
    train-stay count. Ties are broken deterministically by group name ascending
    (so the cap is reproducible across runs); if fewer than ``top_k`` groups pass
    the floor they are all kept. The returned ``vocab`` is sorted alphabetically
    for a stable ``icd__<group>`` column order.
    """
    grouped = _icd_groups(aligned, group).with_columns(_hash_stay_id_expr("stay_id"))
    train = grouped.filter(pl.col("stay_id").is_in(list(train_stay_ids)))
    prevalence = (
        train.select("stay_id", "group")
        .unique()
        .group_by("group")
        .len()
        .filter(pl.col("len") >= min_prevalence)
    )
    if top_k is not None:
        prevalence = prevalence.sort(["len", "group"], descending=[True, False]).head(top_k)
    vocab = sorted(prevalence["group"].to_list())
    n_train = train.select("stay_id").n_unique()
    if representation == "multihot":
        per_stay = grouped.group_by("stay_id").agg(pl.col("group").unique().alias("groups"))
        cols = [
            pl.col("groups").list.contains(g).cast(pl.Float32).alias(f"icd__{g}") for g in vocab
        ]
        block = per_stay.with_columns(
            *cols, pl.lit(1.0, dtype=pl.Float32).alias("icd_present")
        ).drop("groups")
        return _maybe_pca(block, vocab, train_stay_ids, pca_dim), vocab
    counts = (
        grouped.filter(pl.col("group").is_in(vocab))
        .group_by(["stay_id", "group"])
        .len()
        .rename({"len": "w"})
    )
    if representation == "tfidf":
        import math

        df = train.select("stay_id", "group").unique().group_by("group").len()
        idf = {
            r["group"]: math.log(n_train / r["len"]) if r["len"] else 0.0
            for r in df.iter_rows(named=True)
        }
        counts = counts.with_columns(
            (pl.col("w") * pl.col("group").replace_strict(idf, default=0.0)).alias("w")
        )
    elif representation != "count":
        raise ValueError(f"unknown representation {representation!r}")
    wide = counts.pivot(values="w", index="stay_id", on="group", aggregate_function="first")
    present_stays = grouped.select("stay_id").unique()
    block = present_stays.join(wide, on="stay_id", how="left")
    cols = [
        (pl.col(g) if g in block.columns else pl.lit(0.0))
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias(f"icd__{g}")
        for g in vocab
    ]
    block = block.with_columns(*cols, pl.lit(1.0, dtype=pl.Float32).alias("icd_present"))
    block = block.select("stay_id", *[f"icd__{g}" for g in vocab], "icd_present")
    return _maybe_pca(block, vocab, train_stay_ids, pca_dim), vocab


def _key_schema(*, per_hour: bool) -> dict[str, pl.DataType]:
    """The (hashed) key dtypes a pool/block carries: stay_id Int64 [, hour Float64].

    Used for the empty-branch frames so they match the non-empty path exactly and
    join cleanly against the Int64 preamble (Fix 1).
    """
    schema: dict[str, pl.DataType] = {"stay_id": pl.Int64()}
    if per_hour:
        schema["hour"] = pl.Float64()
    return schema


def _emb_matrix_f32(emb_col: pl.Series) -> np.ndarray:
    """Materialize a List(Float)-typed embedding column as a (n, dim) float32 array.

    Goes through Arrow's fixed-size-list (`list.to_array(width).to_numpy()`) rather
    than `.to_list()` -> `np.asarray`: the latter builds a transient Python list of
    n*dim float objects (~24 B each => tens of GB for the ~1.3M-note per-hour pools)
    before the numpy copy, which is the dominant transient on the OOM path. The
    Arrow route allocates a single contiguous float32 array (~4 GB for 1.3M x 768).
    Used by the per-hour pooler only; the stay-level pooler keeps its float64
    `.to_list()` for the bit-exact baseline-equivalence gate.
    """
    import numpy as np

    raw_width = emb_col.list.len().max()
    width = 0 if raw_width is None else int(float(raw_width))  # type: ignore[arg-type]
    return emb_col.list.to_array(width).to_numpy().astype(np.float32, copy=False)


def _decay_weights(
    delta: np.ndarray, gid: np.ndarray, n_groups: int, half_life: float
) -> np.ndarray:
    """Per-note recency weight a_i = 2^((delta_i - ref_s)/H), ref_s = per-stay max delta.

    H=inf -> all-ones (caller should use the legacy path for bit-identity). The
    per-stay anchor ``ref_s`` keeps a_i <= 1 (no overflow); it cancels in the
    Sum(a_i e_i) / Sum(a_i) ratio so it introduces no leakage.
    """
    import numpy as np

    if np.isinf(half_life):
        return np.ones(delta.shape[0], dtype=np.float64)
    ref = np.full(n_groups, -np.inf, dtype=np.float64)
    np.maximum.at(ref, gid, delta)
    weights: np.ndarray = np.exp2((delta - ref[gid]) / half_life)
    return weights


def _pool_stay_level(
    aligned: pl.DataFrame, emb: pl.DataFrame, *, cutoff_h: float, half_life: float = math.inf
) -> pl.DataFrame:
    """Stay-level mean-pool: one pooled vector per stay over notes with
    delta_h_signed <= cutoff_h. stay_id is already Int64 (hashed by the caller).

    The only wide array materialized is the (n_visible_notes x dim) embedding
    matrix; the embedding is never exploded along its dimension axis (pooling is
    a segmented mean over per-stay group ids).
    """
    import numpy as np

    from critical_mm.modalities.notes_base import visible_notes_stay_level

    vis = (
        visible_notes_stay_level(aligned.lazy(), cutoff_h)
        .select("stay_id", "delta_h_signed", "note_id")
        .collect()
    )
    empty = {"stay_id": pl.Int64(), "pooled": pl.List(pl.Float64())}
    if vis.height == 0:
        return pl.DataFrame(schema=empty)
    joined = vis.join(emb, on="note_id", how="inner")
    if joined.height == 0:
        return pl.DataFrame(schema=empty)

    legacy = math.isinf(half_life)

    pairs = np.asarray(joined["emb"].to_list(), dtype=np.float64)
    width = pairs.shape[1]
    key_df = joined.select("stay_id").with_columns(pl.col("stay_id").rank("dense").alias("__gid"))
    gid = key_df["__gid"].to_numpy().astype(np.int64) - 1
    n_groups = int(gid.max()) + 1
    delta = joined["delta_h_signed"].to_numpy().astype(np.float64)

    if (not legacy) and half_life == 0.0:
        ref = np.full(n_groups, -np.inf, dtype=np.float64)
        np.maximum.at(ref, gid, delta)
        is_max = delta == ref[gid]
        sums = np.zeros((n_groups, width), dtype=np.float64)
        np.add.at(sums, gid, np.where(is_max[:, None], pairs, 0.0))
        counts = np.zeros(n_groups, dtype=np.float64)
        np.add.at(counts, gid, is_max.astype(np.float64))
        mean = sums / counts[:, None]
    elif legacy:
        sums = np.zeros((n_groups, width), dtype=np.float64)
        np.add.at(sums, gid, pairs)
        counts = np.bincount(gid, minlength=n_groups).astype(np.float64)
        mean = sums / counts[:, None]
    else:
        a = _decay_weights(delta, gid, n_groups, half_life)
        sums = np.zeros((n_groups, width), dtype=np.float64)
        np.add.at(sums, gid, a[:, None] * pairs)
        wcounts = np.zeros(n_groups, dtype=np.float64)
        np.add.at(wcounts, gid, a)
        mean = sums / wcounts[:, None]

    rep = key_df.with_row_index("__row").group_by("__gid").agg(pl.col("__row").min())
    rep_rows = rep.sort("__gid")["__row"].to_numpy()
    key_unique = joined.select("stay_id")[rep_rows]
    return key_unique.with_columns(
        pl.Series("pooled", [mean[i].tolist() for i in range(n_groups)], dtype=pl.List(pl.Float64))
    )


def _asof_to_grid(per_delta: pl.DataFrame, dyn_grid: pl.DataFrame) -> pl.DataFrame:
    """Backward as-of join of the hour grid against the per-(stay,delta) frame.

    For each (stay, hour) pick the latest note row with delta_h_signed <= hour
    (backward, by stay). Rows with no note at/below the hour get no match -> dropped
    (downstream merge zero-fills + present=0). The real preamble's dyn_grid stay_id
    is ALREADY Int64-hashed; hash here only if a caller passed a string-keyed grid,
    so the 'by' join keys match the Int64 per_delta side (Fix 1).
    """
    grid_stay = (
        pl.col("stay_id")
        if dyn_grid.schema["stay_id"] == pl.Int64()
        else _hash_stay_id_expr("stay_id")
    )
    grid = (
        dyn_grid.with_columns(grid_stay)
        .select("stay_id", pl.col("hour").cast(pl.Float64))
        .unique()
        .sort(["stay_id", "hour"])
    )
    matched = grid.join_asof(
        per_delta,
        left_on="hour",
        right_on="delta_h_signed",
        by="stay_id",
        strategy="backward",
        check_sortedness=False,
    )
    return matched.filter(pl.col("pooled").is_not_null()).select("stay_id", "hour", "pooled")


def _per_delta_meanlast(notes: pl.DataFrame, mat: np.ndarray, *, mode: str) -> pl.DataFrame:
    """Per (stay, delta): mean of the embeddings AT that delta (mode='last' = most-recent
    semantics; ties at the same delta collapse to their joint mean).

    ``notes`` must be sorted by (stay_id, delta_h_signed) and aligned row-for-row
    with ``mat``. The returned frame is sorted by (stay_id, delta_h_signed), ready
    for the as-of join.
    """
    import numpy as np

    key = notes.select("stay_id", pl.col("delta_h_signed").cast(pl.Float64)).with_row_index("__row")
    grp = key.group_by(["stay_id", "delta_h_signed"], maintain_order=True).agg(pl.col("__row"))
    grp = grp.sort(["stay_id", "delta_h_signed"])
    pooled = [
        np.asarray([mat[i] for i in rowlist]).mean(0).tolist() for rowlist in grp["__row"].to_list()
    ]
    return grp.select("stay_id", "delta_h_signed").with_columns(
        pl.Series("pooled", pooled, dtype=pl.List(pl.Float64))
    )


def _cum_weighted_mean_scan(
    delta: np.ndarray, mat: np.ndarray, gid: np.ndarray, n_groups: int, half_life: float
) -> np.ndarray:
    """Per-(stay,prefix) cumulative recency-weighted mean, anchored to each prefix.

    For note ``i`` (sorted ascending by delta within its stay) the cumulative mean
    over all earlier-or-equal notes of the same stay is

        cum_mean[i] = Sum_{j<=i, same stay} 2^((delta_j - delta_i)/H) e_j
                    / Sum_{j<=i, same stay} 2^((delta_j - delta_i)/H)

    i.e. the weights are anchored to THIS prefix's own (maximum) delta ``delta_i``,
    so the most-recent visible note always carries weight exactly 1. This is the
    mathematically-identical, numerically-stable form of the recency pool: the ratio
    is anchor-invariant, and anchoring per-prefix (rather than to the stay's GLOBAL
    max delta) guarantees the running weight ``W >= 1`` at every row -> no underflow,
    no catastrophic cancellation, no 0/0.

    Implemented as a streaming segmented scan: within a stay, advancing from note
    ``i-1`` to ``i`` re-anchors by ``r = 2^((delta_{i-1} - delta_i)/H) in (0, 1]``::

        W <- W * r + 1 S <- S * r + e_i cum_mean[i] = S / W

    ``r in [0, 1]`` (deltas ascending) so no term ever overflows; the ``+1`` keeps
    ``W >= 1`` so the divide is always finite. The accumulated decay ``r`` for a note
    too old to register at the working precision is flushed to exactly 0 (its
    contribution is genuinely negligible — the correct recency limit), which also
    avoids spurious subnormal-underflow flags. The flush threshold tracks the dtype
    of ``mat``: for the float32 per-hour path (memory-driven, see ``_pool_per_hour``)
    a decay below the float32 relative epsilon ``2**-24`` is already lost in the
    accumulation, so we floor there; for a float64 ``mat`` we floor at the float64
    ULP ``2**-53``. ``mat`` rows are aligned with ``delta``/``gid`` and the output
    ``cum_mean`` carries ``mat``'s dtype; the scan is O(n_notes) with a length-``dim``
    vector update per note.
    """
    import numpy as np

    n, dim = mat.shape
    cum_mean = np.empty((n, dim), dtype=mat.dtype)
    inv_h = 1.0 / half_life
    underflow_floor = 2.0**-24 if mat.dtype == np.float32 else 2.0**-53
    w_run = 0.0
    s_run = np.zeros(dim, dtype=mat.dtype)
    prev_g = -1
    prev_d = 0.0
    for i in range(n):
        g = int(gid[i])
        d = float(delta[i])
        if g != prev_g:
            w_run = 1.0
            s_run = mat[i].copy()
        else:
            r = 2.0 ** ((prev_d - d) * inv_h)
            if r < underflow_floor:
                r = 0.0
            w_run = w_run * r + 1.0
            s_run = s_run * r + mat[i]
        cum_mean[i] = s_run / w_run
        prev_g = g
        prev_d = d
    return cum_mean


def _pool_per_hour_perdelta(
    aligned: pl.DataFrame,
    emb: pl.DataFrame,
    *,
    half_life: float = math.inf,
    _force_legacy: bool = False,
) -> pl.DataFrame:
    """The COMPACT per-(stay,delta) pool — the pre-broadcast half of ``_pool_per_hour``.

    Returns one ``pooled`` vector per distinct (stay_id, delta_h_signed), sorted by
    (stay_id, delta_h_signed) and as-of-join-ready (no hour grid involved). Each row
    is the recency-weighted cumulative mean of all notes up to (and including) that
    delta within the stay. Splitting this out (E1) lets ``build_notes_block`` fit PCA
    on these ~n_notes COMPACT states and broadcast the REDUCED vectors afterward,
    instead of PCA-ing the hour-broadcast millions-of-rows frame.

    H=inf (or ``_force_legacy``) -> the unweighted cumulative MEAN via the UNCHANGED
    legacy integer-count path (bit-identical to the original pooler). H->0 -> the
    single most-recent note per (stay, delta).

    stay_id on ``aligned`` is already Int64 (hashed by the caller).
    """
    import numpy as np

    empty = {
        "stay_id": pl.Int64(),
        "delta_h_signed": pl.Float64(),
        "pooled": pl.List(pl.Float64()),
    }

    notes = (
        aligned.select("stay_id", "delta_h_signed", "note_id")
        .join(emb, on="note_id", how="inner")
        .sort("stay_id", "delta_h_signed")
    )
    if notes.height == 0:
        return pl.DataFrame(schema=empty)

    mat = _emb_matrix_f32(notes["emb"])
    width = mat.shape[1]
    n = mat.shape[0]
    gid = notes.select(pl.col("stay_id").rank("dense")).to_series().to_numpy().astype(np.int64) - 1
    n_groups = int(gid.max()) + 1
    delta = notes["delta_h_signed"].to_numpy().astype(np.float64)

    legacy = _force_legacy or math.isinf(half_life)

    if (not legacy) and half_life == 0.0:
        return _per_delta_meanlast(notes, mat, mode="last")

    if legacy:
        csum = np.cumsum(mat, axis=0, dtype=np.float32)
        arange = np.arange(n, dtype=np.int64)
        group_start_idx = np.full(n_groups, n, dtype=np.int64)
        np.minimum.at(group_start_idx, gid, arange)
        start_offset = np.zeros((n_groups, width), dtype=np.float32)
        nonzero = group_start_idx > 0
        start_offset[nonzero] = csum[group_start_idx[nonzero] - 1]
        cum_sum_pergroup = csum - start_offset[gid]
        del csum, start_offset
        cum_w = (arange - group_start_idx[gid] + 1).astype(np.float32)
        cum_mean = cum_sum_pergroup / cum_w[:, None]
    else:
        cum_mean = _cum_weighted_mean_scan(delta, mat, gid, n_groups, half_life)

    key = notes.select("stay_id", pl.col("delta_h_signed").cast(pl.Float64))
    last_idx = (
        key.with_row_index("__row")
        .group_by(["stay_id", "delta_h_signed"], maintain_order=True)
        .agg(pl.col("__row").max())
        .sort(["stay_id", "delta_h_signed"])
    )
    sel = last_idx["__row"].to_numpy()
    return last_idx.select("stay_id", "delta_h_signed").with_columns(
        pl.Series(
            "pooled",
            [cum_mean[i].tolist() for i in sel],
            dtype=pl.List(pl.Float64),
        )
    )


def _pool_per_hour(
    aligned: pl.DataFrame,
    emb: pl.DataFrame,
    *,
    dyn_grid: pl.DataFrame,
    half_life: float = math.inf,
    _force_legacy: bool = False,
) -> pl.DataFrame:
    """Per-(stay,hour) recency-weighted pool WITHOUT materializing the
    (stay x hour x note) cross product.

    Thin wrapper over ``_pool_per_hour_perdelta`` (the compact pool) followed by the
    backward as-of broadcast onto the hour grid. Retained for the pooling unit tests
    and the per-hour H=inf bit-identity gate; ``build_notes_block`` uses the split
    form directly so it can reduce BEFORE this broadcast (E1).

    stay_id on both ``aligned`` and ``dyn_grid`` is already Int64 (hashed by the
    caller), so the per-hour join keys match (Fix 1).
    """
    per_delta = _pool_per_hour_perdelta(
        aligned, emb, half_life=half_life, _force_legacy=_force_legacy
    )
    if per_delta.height == 0:
        return pl.DataFrame(
            schema={"stay_id": pl.Int64(), "hour": pl.Float64(), "pooled": pl.List(pl.Float64())}
        )
    return _asof_to_grid(per_delta, dyn_grid)


def _pool_compact(
    aligned: pl.DataFrame,
    emb: pl.DataFrame,
    *,
    per_hour: bool,
    cutoff_h: float | None,
    half_life: float = math.inf,
) -> pl.DataFrame:
    """Pool visible note embeddings into the COMPACT pre-broadcast frame (E1).

    Returns a frame with hashed stay_id (Int64), a list-typed ``pooled`` column
    (the recency-weighted mean of each group's visible note vectors), and — for the
    per-hour path — the ``delta_h_signed`` (Float64) of each pooled note-state. The
    per-hour frame is one row per distinct (stay, delta) and is NOT yet broadcast to
    the hour grid: ``build_notes_block`` reduces it with PCA first, then as-of-joins
    the reduced vectors onto the hour grid (reduce-before-broadcast). The stay-level
    frame is one row per stay (no broadcast at all — unchanged from before).

    Same stay_id-hash-up-front + cross-product-free guarantees as ``_pool_per_hour``;
    this variant simply omits the per-hour as-of broadcast so PCA fits on the compact
    note-states (review I2 / E1 OOM fix).
    """
    aligned = aligned.with_columns(_hash_stay_id_expr("stay_id"))
    if per_hour:
        return _pool_per_hour_perdelta(aligned, emb, half_life=half_life)
    assert cutoff_h is not None
    return _pool_stay_level(aligned, emb, cutoff_h=cutoff_h, half_life=half_life)


def build_notes_block(
    aligned: pl.DataFrame,
    emb: pl.DataFrame,
    *,
    train_stay_ids: set[int],
    per_hour: bool,
    dyn_grid: pl.DataFrame | None = None,
    cutoff_h: float | None = None,
    pca_dim: int | None = 64,
    half_life: float = math.inf,
) -> tuple[pl.DataFrame, PCA | None]:
    """Visibility-pooled note embedding, train-fit PCA + standardizer (E1 order).

    Returns (block, pca). block columns: stay_id (Int64) [, hour (Float64)],
    note__0..k (f32), notes_present (f32 0/1).

    Reduce-before-broadcast (E1): the PCA + StandardScaler are fit on the COMPACT
    pooled note-states (one row per stay for stay-level, one row per distinct
    (stay, delta) for per-hour) over TRAIN stays only; the *reduced* d-dim vectors
    are then as-of-broadcast onto the per-hour grid. This keeps the PCA fit and the
    broadcast off the millions-of-hour-rows frame. n_components is clamped to
    min(pca_dim, n_train_rows, emb_dim).

    ``pca_dim=None`` (raw): SKIP PCA entirely and emit the full-width embedding,
    train-fit StandardScaler-normalized only (mirrors ``_maybe_pca``'s None
    short-circuit for ICD). ``pca`` is then None.

    When no notes are visible at all the block is empty (still keyed on Int64
    stay_id) and pca is None.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pooled = _pool_compact(aligned, emb, per_hour=per_hour, cutoff_h=cutoff_h, half_life=half_life)
    if pooled.height == 0:
        empty_cols: dict[str, pl.DataType] = {
            **_key_schema(per_hour=per_hour),
            "notes_present": pl.Float32(),
        }
        return pl.DataFrame(schema=empty_cols), None

    compact_keys = ["stay_id", "delta_h_signed"] if per_hour else ["stay_id"]

    if per_hour:
        mat = _emb_matrix_f32(pooled["pooled"])
    else:
        mat = np.asarray(pooled["pooled"].to_list(), dtype=np.float64)
    train_mask = pooled["stay_id"].is_in(list(train_stay_ids)).to_numpy()
    n_train = int(train_mask.sum())
    train_mat = mat[train_mask] if n_train > 0 else mat

    scaler = StandardScaler()
    pca: PCA | None
    if pca_dim is None:
        scaler.fit(train_mat)
        reduced = scaler.transform(mat).astype(np.float32)
        pca = None
        n_comp = mat.shape[1]
    else:
        n_comp = max(1, min(pca_dim, n_train if n_train > 0 else mat.shape[0], mat.shape[1]))
        pca = PCA(n_components=n_comp, random_state=0)
        pca.fit(train_mat)
        scaler.fit(pca.transform(train_mat))
        reduced = scaler.transform(pca.transform(mat)).astype(np.float32)

    reduced_compact = pooled.select(compact_keys).with_columns(
        *[pl.Series(f"note__{i}", reduced[:, i]) for i in range(n_comp)]
    )

    if per_hour:
        assert dyn_grid is not None
        block = _asof_reduced_to_grid(reduced_compact, dyn_grid, n_comp=n_comp)
    else:
        block = reduced_compact.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("notes_present"))
    return block, pca


def _asof_reduced_to_grid(
    reduced_perdelta: pl.DataFrame, dyn_grid: pl.DataFrame, *, n_comp: int
) -> pl.DataFrame:
    """Backward as-of broadcast of the REDUCED per-(stay,delta) note vectors (E1).

    ``reduced_perdelta`` carries (stay_id Int64, delta_h_signed Float64, note__0..n-1).
    For each (stay, hour) we pick the latest note-state with delta_h_signed <= hour
    (backward, by stay) and attach its reduced vector + notes_present=1.0. Grid rows
    with no note at/below the hour get no match and are dropped (downstream zero-fills
    + present=0). This is the d-wide analogue of ``_asof_to_grid`` — broadcasting the
    reduced (<=128-d) vectors instead of the 768-d raw embeddings.
    """
    grid_stay = (
        pl.col("stay_id")
        if dyn_grid.schema["stay_id"] == pl.Int64()
        else _hash_stay_id_expr("stay_id")
    )
    grid = (
        dyn_grid.with_columns(grid_stay)
        .select("stay_id", pl.col("hour").cast(pl.Float64))
        .unique()
        .sort(["stay_id", "hour"])
    )
    val_cols = [f"note__{i}" for i in range(n_comp)]
    matched = grid.join_asof(
        reduced_perdelta.sort(["stay_id", "delta_h_signed"]),
        left_on="hour",
        right_on="delta_h_signed",
        by="stay_id",
        strategy="backward",
        check_sortedness=False,
    )
    return (
        matched.filter(pl.col("note__0").is_not_null())
        .select("stay_id", "hour", *val_cols)
        .with_columns(pl.lit(1.0, dtype=pl.Float32).alias("notes_present"))
    )
