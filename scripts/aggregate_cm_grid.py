"""Aggregate the CM-native 60-cell grid into the live reference table.

Reads per-cell metadata.json from data/checkpoints/<task>/<ds>/<model>/seed_<n>/
(written by critical_mm.training.grid.run_grid). Computes mean +/- std of the
test metric per (task, dataset, model) across 5 seeds. Writes the result to
data/cm_reference/v1/cm_grid_metrics.json with a manifest fingerprint that
ties the values back to the data + locked-splits SHA at training time.

The output JSON is the source from which CM_REFERENCE in
critical_mm/validation/oracle.py is hand-lifted as a literal table (so the
oracle's import remains deterministic and tests can snapshot-lock it).

Pass `--datasets eicu hirid miiv nwicu omix --out data/cm_reference/v2/cm_grid_metrics.json`
to emit the 5-dataset CM_REFERENCE_V2 grid without touching the locked V1 file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CellTable = dict[tuple[str, str, str], list[tuple[int, float, dict[str, Any]]]]

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT_ROOT = REPO / "data" / "checkpoints"
DEFAULT_OUT = REPO / "data" / "cm_reference" / "v1" / "cm_grid_metrics.json"

CLASSIFICATION_TASKS = {"mortality24", "aki", "sepsis"}
REGRESSION_TASKS = {"los", "kidney_function"}

def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def _metric_key_for(task: str) -> str:
    if task in CLASSIFICATION_TASKS:
        return "test/AUC"
    if task in REGRESSION_TASKS:
        return "test/MAE"
    raise ValueError(f"unknown task: {task}")

def _collect_cells(
    ckpt_root: Path,
    datasets: set[str] | None = None,
    renamed: list[dict[str, str]] | None = None,
) -> CellTable:
    """Return {(task, ds, model): [(seed, metric_value, splits_fp), ...]}.

    Iterates metadata.json under <task>/<ds>/<model>/seed_<n>/. Skips cells with
    no `test_metrics` or where the metric key is absent.

    `model` is taken from the DIRECTORY, not from `config["model"]`. The 126
    `CM-EHR{,-mean}-probe-aligned` cells record `config["model"]` without the
    `-aligned` suffix, so keying on the config silently merged the aligned arm
    (Appendix M, `tab:app-ehrfm-perhour`) into the non-aligned cell and emitted
    n=6 means averaged across two different arms. The directory name is what
    every consumer of `f"{ds}_{model}"` means, and it is what
    scripts/build_auprc_table.py keys on. Disagreements are appended to
    `renamed` so the manifest records them rather than hiding them.

    Args:
        ckpt_root: Root directory containing the checkpoint tree.
        datasets: If provided, restrict to cells whose dataset is in this set.
                  If None, all canonical-task cells are collected.
        renamed: Optional sink for (path, config model, directory model) records.
    """
    out: CellTable = defaultdict(list)
    for meta_path in sorted(ckpt_root.glob("*/*/*/seed_*/metadata.json")):
        meta = json.loads(meta_path.read_text())
        cfg = meta.get("config", {})
        task = cfg.get("task")
        dataset = cfg.get("dataset")
        model = meta_path.parents[1].name
        seed = cfg.get("seed")
        if not all([task, dataset, model, isinstance(seed, int)]):
            continue
        if task not in CLASSIFICATION_TASKS and task not in REGRESSION_TASKS:
            continue
        if datasets is not None and dataset not in datasets:
            continue
        if cfg.get("model") not in (None, model) and renamed is not None:
            renamed.append(
                {
                    "path": str(meta_path.relative_to(ckpt_root)),
                    "config_model": str(cfg.get("model")),
                    "directory_model": model,
                }
            )
        metric_key = _metric_key_for(task)
        test_metrics = meta.get("test_metrics") or {}
        if metric_key not in test_metrics:
            continue
        value = float(test_metrics[metric_key])
        splits_fp = meta.get("splits_fingerprint", {})
        out[(task, dataset, model)].append((seed, value, splits_fp))
    return out

_REGRESSION_OUTCOME_SCALE = {"los": 168.0, "kidney_function": 15.0}

def _summarize(cells: CellTable) -> dict[str, dict[str, dict[str, float]]]:
    """Mean / std per (task, ds, model). Output shape:
    {task: {f"{ds}_{model}": {"mean": ..., "std": ..., "n": ..., "scaled_mean": ...}}}

    For regression tasks (los, kidney_function) `scaled_mean` is the
    YAIB-equivalent MAE -- divide raw MAE by the per-task outcome scale
    (168 hours for los, 15 mg/dL for kf). For classification tasks
    `scaled_mean` is omitted.
    """
    summary: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (task, ds, model), entries in cells.items():
        values = [v for (_seed, v, _fp) in entries]
        key = f"{ds}_{model}"
        if len(values) == 1:
            entry: dict[str, float] = {"mean": values[0], "std": 0.0, "n": 1}
        else:
            entry = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values),
                "n": len(values),
            }
        scale = _REGRESSION_OUTCOME_SCALE.get(task)
        if scale is not None and scale > 0:
            entry["scaled_mean"] = entry["mean"] / scale
            entry["scaled_std"] = entry["std"] / scale
        summary[task][key] = entry
    return summary

def _pick_canonical_fingerprint(cells: CellTable) -> dict[str, Any]:
    """Pick any cell's splits_fingerprint as the canonical reference.

    All cells SHOULD share the same fingerprint (locked splits at seed=42 per
    (task, ds)). We return the first non-empty fingerprint we find; if there's
    drift, we flag it loudly in the manifest.
    """
    seen: set[str] = set()
    first: dict[str, Any] | None = None
    for entries in cells.values():
        for _seed, _value, fp in entries:
            if not fp:
                continue
            fp_sha = fp.get("folds_combined_sha256", "")
            seen.add(fp_sha)
            if first is None:
                first = fp
    return {
        "first_fingerprint": first or {},
        "n_distinct_fold_shas": len(seen),
        "fingerprints_consistent": len(seen) <= 1,
    }

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="restrict to these datasets (default: all discovered canonical-task cells)",
    )
    args = ap.parse_args()

    renamed: list[dict[str, str]] = []
    cells = _collect_cells(
        args.ckpt_root,
        datasets=set(args.datasets) if args.datasets else None,
        renamed=renamed,
    )
    if not cells:
        print(f"aggregate_cm_grid: no metadata.json found under {args.ckpt_root}", file=sys.stderr)
        sys.exit(1)

    summary = _summarize(cells)
    fp = _pick_canonical_fingerprint(cells)
    n_cells = sum(len(entries) for entries in cells.values())
    n_uniq = len(cells)
    per_cell_counts = {cell: len(entries) for cell, entries in cells.items()}
    seeds_per_cell = sorted(set(per_cell_counts.values()))
    seeds_max = max(per_cell_counts.values()) if per_cell_counts else 0
    seeds_min = min(per_cell_counts.values()) if per_cell_counts else 0
    under_seeded_cells = sorted(
        (
            {"task": task, "dataset": ds, "model": model, "n_seeds": n}
            for (task, ds, model), n in per_cell_counts.items()
            if n < seeds_max
        ),
        key=lambda c: (c["task"], c["dataset"], c["model"]),
    )

    payload = {
        "manifest": {
            "git_sha": _git_sha(),
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "n_cells_total": n_cells,
            "n_unique_taskdsmodel": n_uniq,
            "seeds_per_cell_observed": seeds_per_cell,
            "seeds_per_cell_min": seeds_min,
            "seeds_per_cell_max": seeds_max,
            "n_under_seeded_cells": len(under_seeded_cells),
            "under_seeded_cells": under_seeded_cells,
            "n_config_model_mismatches": len(renamed),
            "config_model_mismatches": renamed,
            "splits_fingerprint": fp,
        },
        "reference": dict(summary),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"aggregate_cm_grid: wrote {args.out} ({n_uniq} cells, {n_cells} per-seed entries)")
    if renamed:
        dirs = sorted({Path(r["path"]).parent.parent.name for r in renamed})
        print(
            f" NOTE: {len(renamed)} cells whose config['model'] disagrees with their "
            f"directory; keyed on the directory. Affected model dirs: {', '.join(dirs)}"
        )
    if under_seeded_cells:
        print(
            f" UNDER-SEEDED ({len(under_seeded_cells)} of {n_uniq}, max={seeds_max}): "
            + ", ".join(
                f"{c['task']}/{c['dataset']}/{c['model']}(n={c['n_seeds']})"
                for c in under_seeded_cells[:12]
            )
            + (" ..." if len(under_seeded_cells) > 12 else ""),
            file=sys.stderr,
        )
    if not fp["fingerprints_consistent"]:
        print(
            " WARNING: splits fingerprints inconsistent across cells "
            f"({fp['n_distinct_fold_shas']} distinct fold SHAs). "
            "Verify lock_splits.py was run once before this grid.",
            file=sys.stderr,
        )

if __name__ == "__main__":
    main()
