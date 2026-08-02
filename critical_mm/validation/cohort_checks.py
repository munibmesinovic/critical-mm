"""Structural checks a built cohort must pass before anything trains on it.

WHY THIS EXISTS

Two defects of the same shape reached published results and were found only because a
rebuild happened to be diffed against the live tree (the project notes §H3, §H6):

* **OMIX** carried ``sta.sex_male == 0.5`` for every stay in all five task cohorts — a
  constant placeholder. The sex feature was dead weight in every published OMIX result,
  and `appendix_D_calibration.tex` asserted the opposite.
* **NWICU** carried ``sta.weight`` and ``sta.height`` 100% null on three of five tasks.
  A rebuild recovers real values (weight ≈85 kg on 10,703 stays).

Neither is a crash, a schema error or a test failure. A constant column and an all-null
column train perfectly happily and produce plausible numbers — they simply carry no
information, which is invisible downstream. That is precisely the class of defect a
benchmark cannot afford to find by luck.

These checks are cheap (column statistics, set comparisons) and are meant to run at
build time and in CI, not as a manual audit.

Severity:
  ERROR  — the cohort is wrong; do not train on it.
  WARN   — legitimate in some cohorts, but must be a deliberate, recorded choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

_ID_COLUMNS = frozenset({"stay_id", "patient_id", "label_time", "time", "hour"})

_CONSTANT_HORIZON_TASKS = frozenset({"mortality24", "kidney_function", "mortality24_nostat",
                                     "mortality24_leaky"})
_PER_HOUR_TASKS = frozenset({"aki", "sepsis", "los", "sepsis_simple"})

_MIN_STAYS_FOR_CONSTANT_CHECK = 10

_KNOWN_ABSENT: dict[str, frozenset[str]] = {
    "omix": frozenset({"weight", "height"}),
    "zigong": frozenset({"weight", "height"}),
}

_KNOWN_ABSENT_DYN: dict[str, frozenset[str]] = {
    "nwicu": frozenset({"be", "bnd", "fio2", "inr_pt", "map", "methb", "pco2", "ph", "urine"}),
    "omix": frozenset({"be", "bnd", "ckmb", "cl", "fgn", "k", "map", "mch", "mchc",
                       "mcv", "methb", "mg", "na"}),
    "zigong": frozenset({"bnd", "fio2"}),
}

_CLIP_CEILING: dict[str, float] = {"los": 168.0, "kidney_function": 15.0}

_MIN_ROWS_FOR_CONSTANT_DYN = 500

_CLASSIFICATION_TASKS = frozenset({
    "mortality24", "aki", "sepsis", "sepsis_simple",
    "mortality24_leaky", "mortality24_nostat",
})

@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.detail}"

def _read(cohort_dir: Path, segment: str) -> pl.DataFrame | None:
    p = cohort_dir / f"{segment}.parquet"
    return pl.read_parquet(p) if p.exists() else None

def _stay_ids(df: pl.DataFrame) -> set[str]:
    """Normalised stay ids. NWICU stores Int32 where others store a prefixed string."""
    return set(df["stay_id"].cast(pl.Utf8).to_list())

def check_cohort(cohort_dir: Path) -> list[Finding]:
    """Run every structural check over one ``data/processed/<task>/<dataset>``."""
    out: list[Finding] = []
    sta = _read(cohort_dir, "sta")
    dyn = _read(cohort_dir, "dyn")
    outc = _read(cohort_dir, "outc")

    for name, frame in (("sta", sta), ("dyn", dyn), ("outc", outc)):
        if frame is None:
            out.append(Finding("ERROR", "segment-present", f"{name}.parquet is missing"))
    if sta is None or dyn is None or outc is None:
        return out

    dataset = cohort_dir.name
    task = cohort_dir.parent.name
    known_absent = _KNOWN_ABSENT.get(dataset, frozenset())
    for col in sta.columns:
        if col in _ID_COLUMNS:
            continue
        s = sta[col]
        nulls = s.null_count()
        if nulls == s.len():
            sev = "WARN" if col in known_absent else "ERROR"
            note = (
                " (known absent upstream for this dataset — a current-code rebuild also "
                "yields all-null, so it cannot be recovered)"
                if col in known_absent else
                " — it carries no information and any claim that this cohort has it is false"
            )
            out.append(Finding(
                sev, "all-null-static",
                f"sta.{col} is null for all {s.len():,} stays{note}",
            ))
            continue
        non_null = s.drop_nulls()
        if non_null.len() >= _MIN_STAYS_FOR_CONSTANT_CHECK and non_null.n_unique() == 1:
            out.append(Finding(
                "ERROR", "constant-static",
                f"sta.{col} is the single value {non_null[0]!r} for all "
                f"{non_null.len():,} non-null stays — a placeholder, not a feature",
            ))

    dyn_absent = _KNOWN_ABSENT_DYN.get(dataset, frozenset())
    for col in dyn.columns:
        if col in _ID_COLUMNS:
            continue
        s = dyn[col]
        if s.null_count() == s.len():
            known = col in dyn_absent
            out.append(Finding(
                "WARN" if known else "ERROR", "all-null-dynamic",
                f"dyn.{col} is null for all {s.len():,} rows"
                + (" (known absent upstream for this dataset)" if known else
                   " — the model receives a dead channel and any claim this cohort "
                   "measures it is false"),
            ))
            continue
        non_null = s.drop_nulls()
        if non_null.len() >= _MIN_ROWS_FOR_CONSTANT_DYN and non_null.n_unique() == 1:
            out.append(Finding(
                "ERROR", "constant-dynamic",
                f"dyn.{col} is the single value {non_null[0]!r} across "
                f"{non_null.len():,} non-null rows — a placeholder, not a measurement",
            ))

    label_col = "label_value" if "label_value" in outc.columns else (
        "label" if "label" in outc.columns else None)
    if label_col is None:
        out.append(Finding("ERROR", "no-label-column",
                           f"outc has neither `label_value` nor `label`: {outc.columns}"))
    else:
        lab = outc[label_col].drop_nulls()
        if lab.len() == 0:
            out.append(Finding("ERROR", "empty-labels", f"outc.{label_col} is entirely null"))
        else:
            if lab.n_unique() == 1:
                out.append(Finding("ERROR", "constant-label",
                                   f"outc.{label_col} is {lab[0]!r} for every row"))
            if task in _CLASSIFICATION_TASKS:
                pos = int(lab.cast(pl.Float64).sum())
                if pos == 0:
                    out.append(Finding("ERROR", "no-positive-labels",
                                       f"{task} has 0 positive labels in {lab.len():,} rows"))
            ceiling = _CLIP_CEILING.get(task)
            if ceiling is not None:
                at = int((lab.cast(pl.Float64) >= ceiling).sum())
                if at == lab.len():
                    out.append(Finding("ERROR", "label-at-clip-ceiling",
                                       f"every {task} label is at the clip ceiling {ceiling}"))
                elif at:
                    out.append(Finding("WARN", "label-at-clip-ceiling",
                                       f"{at:,} of {lab.len():,} {task} labels sit at the "
                                       f"clip ceiling {ceiling}"))

    s_ids, o_ids, d_ids = _stay_ids(sta), _stay_ids(outc), _stay_ids(dyn)
    if s_ids != o_ids:
        out.append(Finding(
            "ERROR", "sta-outc-stays",
            f"sta and outc disagree: {len(s_ids - o_ids):,} only in sta, "
            f"{len(o_ids - s_ids):,} only in outc",
        ))
    if not d_ids <= o_ids:
        out.append(Finding(
            "ERROR", "dyn-not-subset",
            f"{len(d_ids - o_ids):,} stays appear in dyn but have no outcome row",
        ))
    if o_ids - d_ids:
        out.append(Finding(
            "WARN", "outc-without-dyn",
            f"{len(o_ids - d_ids):,} stays have an outcome but no dynamic rows",
        ))

    hour_col = "hour" if "hour" in dyn.columns else ("time" if "time" in dyn.columns else None)
    if hour_col is None:
        out.append(Finding("ERROR", "dyn-grid", "dyn has neither an `hour` nor a `time` column"))
    else:
        h = dyn[hour_col]
        if h.dtype == pl.Duration:
            h = (h.dt.total_seconds() // 3600).cast(pl.Int32)
        if h.min() is not None and h.min() < 0:
            out.append(Finding(
                "ERROR", "negative-hour",
                f"dyn.{hour_col} reaches {h.min()} — the grid must start at admission",
            ))
        per_stay_max = (
            dyn.select("stay_id", h.alias("_h")).group_by("stay_id").agg(pl.col("_h").max())
        )
        distinct_max = per_stay_max["_h"].n_unique()
        if task in _CONSTANT_HORIZON_TASKS and distinct_max != 1:
            out.append(Finding(
                "ERROR", "ragged-horizon",
                f"{task} is a constant-horizon task but stays end at {distinct_max} "
                f"different hours — this is what a partially-rebuilt cohort looks like",
            ))
        elif task in _PER_HOUR_TASKS and distinct_max == 1:
            out.append(Finding(
                "WARN", "flat-horizon",
                f"{task} should run to length of stay, but every stay ends at hour "
                f"{per_stay_max['_h'][0]}",
            ))

    if sta.height != len(s_ids):
        out.append(Finding(
            "ERROR", "duplicate-sta",
            f"sta has {sta.height:,} rows for {len(s_ids):,} stays",
        ))
    if hour_col is not None:
        n_pairs = dyn.select("stay_id", hour_col).n_unique()
        if n_pairs != dyn.height:
            out.append(Finding(
                "ERROR", "duplicate-dyn",
                f"dyn has {dyn.height:,} rows for {n_pairs:,} distinct (stay, hour) pairs",
            ))

    return out

def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "ERROR"]
