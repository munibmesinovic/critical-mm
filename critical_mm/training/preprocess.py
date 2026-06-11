"""CM-native preprocessor: forward-fill + train-mean impute + scale + concat.

Replaces the vendored YAIB preprocessor (which assumed static categoricals
like `sex` need LabelEncoder + SimpleImputer). CM's exporter already
one-hot encodes sex as `sex_male` Float32, so we don't need those steps.

Inputs:
    data: {Split.train|val|test: {Segment.static|dynamic|outcome: DataFrame}}
    vars: {GROUP, LABEL, SEQUENCE, DataSegment.static|dynamic|outcome: cols}

Outputs:
    Same structure with the static frame merged into dynamic as Segment.features
    (matching what YAIB's PredictionDataset expects).

Steps applied per split (all train-fit stats reused across val/test):
    1. Missingness mask captured BEFORE any imputation, one column per dyn feature.
    2. Forward-fill within each stay's sequence (last-observed-carried-forward).
    3. Mean-impute remaining NaNs from train-split column means. Means are
       computed on raw train (non-NaN cells only); no val/test leakage.
    4. MinMaxScaler fit on train (post-impute); applied to all splits.
    5. Static columns get the same train-mean impute + MinMax pipeline.
    6. Static merged into dynamic frame as PredictionDataset.features_df.

If `mask_output_root` is provided, the dynamic missingness masks are
also persisted to <mask_output_root>/<split>_mask.parquet for each
split, with columns [stay_id, time, <col>_missing, ...]. This gives
post-hoc reproducibility / audit access to which cells the model
saw as 'imputed'.

The implementation uses pandas + sklearn (no recipys dependency) so we
don't inherit recipys' object-column-only assumptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from critical_mm.models._data.constants import DataSegment as Segment
from critical_mm.models._data.constants import DataSplit as Split


def preprocess(
    data: dict[Any, dict[Any, pd.DataFrame]],
    vars: dict[str, Any],
    use_missingness_mask: bool = True,
    mask_output_root: Path | None = None,
    generate_features: bool = False,
) -> dict[Any, dict[Any, pd.DataFrame]]:
    """CM-native preprocessor; see module docstring.

    Args:
        data: nested dict of train/val/test x static/dynamic/outcome frames.
        vars: column-name lookup (GROUP, LABEL, SEQUENCE, per-segment lists).
        use_missingness_mask: if True, prepend `<col>_missing` indicator
            columns to the dynamic features per split (default True).
        mask_output_root: if set, dump the per-split missingness mask
            parquet to this directory. Typically the (task, dataset)
            output dir, e.g. `data/processed/aki/eicu/masks/`.
        generate_features: if True, append per-stay running aggregates
            (cummax, cummin, expanding-mean, expanding-count) for every
            dynamic feature, mirroring YAIB's `StepHistorical`. Required
            for non-temporal ML models (LR, LGBM, ...) — the downstream
            `CommonDataset.get_data_and_labels()` takes `.last()` per stay,
            so the hour-T row needs to summarise the full history rather
            than just contain a single scaled+imputed observation.
    """
    group = vars["GROUP"]
    seq = vars["SEQUENCE"]
    dyn_cols = list(vars[Segment.dynamic])
    sta_cols = list(vars[Segment.static])

    def _coerce_sex(df: pd.DataFrame) -> pd.DataFrame:
        if "sex" in df.columns and df["sex"].dtype == object:
            df = df.copy()
            mapping = {"Male": 1.0, "M": 1.0, "Female": 0.0, "F": 0.0}
            df["sex"] = df["sex"].map(mapping).fillna(0.5).astype(np.float32)
        return df

    for split_name in (Split.train, Split.val, Split.test):
        data[split_name][Segment.static] = _coerce_sex(data[split_name][Segment.static])

    train_dyn = data[Split.train][Segment.dynamic][dyn_cols].astype(np.float32)
    train_dyn_means = train_dyn.mean(axis=0).fillna(0.0).astype(np.float32)
    train_sta = data[Split.train][Segment.static][sta_cols].astype(np.float32)
    train_sta_means = train_sta.mean(axis=0).fillna(0.0).astype(np.float32)

    scaler = MinMaxScaler()
    scaler.fit(train_dyn.fillna(train_dyn_means))
    sta_scaler = MinMaxScaler()
    sta_scaler.fit(train_sta.fillna(train_sta_means))

    if mask_output_root is not None:
        mask_output_root = Path(mask_output_root)
        mask_output_root.mkdir(parents=True, exist_ok=True)

    out: dict[Any, dict[Any, pd.DataFrame]] = {}
    for split_name in (Split.train, Split.val, Split.test):
        seg = data[split_name]
        dyn = seg[Segment.dynamic].copy()
        sta = seg[Segment.static].copy()
        outc = seg[Segment.outcome].copy()

        mask_frame: pd.DataFrame | None = None
        if use_missingness_mask:
            mask_cols = [f"{c}_missing" for c in dyn_cols]
            mask_arr = dyn[dyn_cols].isna().astype(np.float32).to_numpy()
            for i, col in enumerate(mask_cols):
                dyn[col] = mask_arr[:, i]
            if mask_output_root is not None:
                mask_frame = pd.DataFrame(mask_arr, columns=mask_cols, index=dyn.index)
                for key_col in (group, seq):
                    if key_col in dyn.columns:
                        mask_frame.insert(0, key_col, dyn[key_col].values)
                mask_frame.to_parquet(mask_output_root / f"{split_name}_mask.parquet")

        dyn = dyn.sort_values([group, seq])
        dyn[dyn_cols] = dyn.groupby(group, sort=False)[dyn_cols].ffill().astype(np.float32)

        dyn[dyn_cols] = dyn[dyn_cols].fillna(train_dyn_means).astype(np.float32)

        dyn[dyn_cols] = scaler.transform(dyn[dyn_cols]).astype(np.float32)

        if generate_features:
            gb = dyn.groupby(group, sort=False)[dyn_cols]
            cum_rows = dyn.groupby(group, sort=False).cumcount().astype(np.float32) + 1.0
            agg_max = gb.cummax().add_suffix("_max").astype(np.float32)
            agg_min = gb.cummin().add_suffix("_min").astype(np.float32)
            agg_mean = gb.cumsum().div(cum_rows, axis=0).add_suffix("_mean").astype(np.float32)
            count_block = np.broadcast_to(cum_rows.to_numpy()[:, None], (len(dyn), len(dyn_cols)))
            agg_count = pd.DataFrame(
                count_block,
                index=dyn.index,
                columns=[f"{c}_count" for c in dyn_cols],
                dtype=np.float32,
            )
            dyn = pd.concat([dyn, agg_max, agg_min, agg_mean, agg_count], axis=1, copy=False)

        sta[sta_cols] = sta[sta_cols].fillna(train_sta_means).astype(np.float32)
        sta[sta_cols] = sta_scaler.transform(sta[sta_cols]).astype(np.float32)

        sta_indexed = sta.set_index(group)
        sta_indexed = sta_indexed.loc[:, ~sta_indexed.columns.duplicated()]
        features = dyn.join(sta_indexed, on=group, how="left")

        out[split_name] = {
            Segment.features: features,
            Segment.outcome: outc,
        }
    return out


__all__ = ["preprocess"]
