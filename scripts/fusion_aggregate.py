"""Aggregate _fusion checkpoints into data/cm_reference/v4/cm_grid_metrics.json.

Output = V3's 165 cells verbatim (rung-1 alias) + the fusion cells keyed
"<dataset>_<model>__<rung>". Mean/std/n over the available seeds per cell.
Classification cells read test/AUC; regression cells read test/MAE (+ scaled).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from critical_mm.validation.oracle import CM_REFERENCE_V3

from critical_mm.training.config import REPO

FUSION_ROOT = REPO / "data" / "checkpoints" / "_fusion"
OUTCOME_MAX = {"los": 168.0, "kidney_function": 15.0}


def aggregate(fusion_root: Path = FUSION_ROOT) -> dict[str, dict[str, dict[str, float]]]:
    ref: dict[str, dict[str, dict[str, float]]] = {
        task: dict(cells) for task, cells in CM_REFERENCE_V3.items()
    }
    if not fusion_root.exists():
        return ref
    for task_dir in sorted(p for p in fusion_root.iterdir() if p.is_dir()):
        task = task_dir.name
        for ds_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            ds = ds_dir.name
            for model_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
                if "__" not in model_dir.name:
                    print(
                        f"fusion_aggregate: WARNING skipping non-rung dir {model_dir} "
                        "(would collide with a base cell)",
                        flush=True,
                    )
                    continue
                vals = []
                for seed_dir in sorted(model_dir.glob("seed_*")):
                    mp = seed_dir / "metadata.json"
                    if not mp.exists():
                        continue
                    tm = json.loads(mp.read_text()).get("test_metrics", {})
                    v = tm.get("test/AUC", tm.get("test/MAE"))
                    if v is not None:
                        vals.append(float(v))
                if not vals:
                    continue
                cell: dict[str, float] = {
                    "mean": statistics.fmean(vals),
                    "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
                if task in OUTCOME_MAX:
                    cell["scaled_mean"] = cell["mean"] / OUTCOME_MAX[task]
                    cell["scaled_std"] = cell["std"] / OUTCOME_MAX[task]
                ref.setdefault(task, {})[f"{ds}_{model_dir.name}"] = cell
    return ref


def observed_fusion_seed_counts(fusion_root: Path = FUSION_ROOT) -> list[int]:
    """Sorted unique per-cell seed counts found under ``_fusion``.

    Each FUSION cell's ``n`` is the number of seed dirs carrying a usable
    ``test_metrics`` entry. Feeds the lift's honest-minimum seed logic so an
    under-seeded fusion cell is reported truthfully rather than as the modal 3.
    Mirrors :func:`aggregate`'s traversal + clobber guard so the counts match
    exactly the cells that get written.
    """
    counts: set[int] = set()
    if not fusion_root.exists():
        return []
    for task_dir in sorted(p for p in fusion_root.iterdir() if p.is_dir()):
        for ds_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
                if "__" not in model_dir.name:
                    continue
                n = 0
                for seed_dir in sorted(model_dir.glob("seed_*")):
                    mp = seed_dir / "metadata.json"
                    if not mp.exists():
                        continue
                    tm = json.loads(mp.read_text()).get("test_metrics", {})
                    v = tm.get("test/AUC", tm.get("test/MAE"))
                    if v is not None:
                        n += 1
                if n:
                    counts.add(n)
    return sorted(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "data" / "cm_reference" / "v4" / "cm_grid_metrics.json",
    )
    args = ap.parse_args()
    ref = aggregate()
    observed = observed_fusion_seed_counts(FUSION_ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference": ref,
        "manifest": {
            "source_json": "data/cm_reference/v4/cm_grid_metrics.json",
            "generated_at": "fusion-aggregate",
            "seeds_per_cell_observed": observed or [3],
            "seeds_per_cell_min": min(observed) if observed else 3,
        },
    }
    args.out.write_text(json.dumps(payload, indent=2))
    n = sum(len(v) for v in ref.values())
    print(f"fusion_aggregate: wrote {args.out} ({n} cells)")


if __name__ == "__main__":
    main()
