"""Rebuild non-sepsis task outputs against the current code/interim state.

For one dataset (passed as a CLI arg), runs aki / kidney_function / los /
mortality24 ``Task.build()`` sequentially. Writes
``data/processed/<task>/<dataset>/{sta,dyn,outc}.parquet`` for each.

Wrap in scripts/mem_run.sh for memory safety:

    MEM_MAX=110G bash scripts/mem_run.sh nonsepsis-eicu \\
      python3 \\
      scripts/rebuild_nonsepsis_tasks.py eicu

Sequential by design (no parallel datasets, no parallel tasks) per the
session-7 memory discipline — phase-2 parallel rebuilds OOM-killed the
host under prior session SSH-lockout incidents.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from critical_mm.cohorts.base import build_base_cohort
from critical_mm.tasks.aki import AKI
from critical_mm.tasks.kf import KidneyFunction
from critical_mm.tasks.los import LengthOfStay
from critical_mm.tasks.mortality24 import Mortality24

TASK_CLASSES = {
    "aki": AKI,
    "kidney_function": KidneyFunction,
    "los": LengthOfStay,
    "mortality24": Mortality24,
}

DATASETS = ("eicu", "miiv", "hirid", "sicdb", "nwicu", "omix", "zigong")

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in DATASETS:
        print(
            f"usage: rebuild_nonsepsis_tasks.py {{{'|'.join(DATASETS)}}} [task ...]\n"
            f"  tasks default to all of: {' '.join(TASK_CLASSES)}",
            file=sys.stderr,
        )
        sys.exit(2)

    dataset = sys.argv[1]
    requested = sys.argv[2:] or list(TASK_CLASSES)
    unknown = [t for t in requested if t not in TASK_CLASSES]
    if unknown:
        print(f"unknown task(s): {', '.join(unknown)}", file=sys.stderr)
        sys.exit(2)
    repo = Path("$CRITICAL_MM_REPO")
    processed = repo / "data" / "processed"
    interim = repo / "data" / "interim"

    print(f"=== rebuilding non-sepsis tasks for {dataset} ===", flush=True)
    total_start = time.perf_counter()

    cohort_start = time.perf_counter()
    print("--- base_cohort ---", flush=True)
    cohort = build_base_cohort(
        dataset=dataset,
        interim_root=interim,
        processed_root=processed,
        repo_root=repo,
    )
    print(
        f"base_cohort/{dataset} ready in {time.perf_counter() - cohort_start:.1f}s "
        f"(n_in={cohort['n_in']:,} n_out={cohort['n_out']:,})",
        flush=True,
    )

    for task_name, cls in ((t, TASK_CLASSES[t]) for t in requested):
        t0 = time.perf_counter()
        print(f"\n--- {task_name}/{dataset} ---", flush=True)
        result = cls().build(
            dataset=dataset,
            processed_root=processed,
            interim_root=interim,
            repo_root=repo,
        )
        dt = time.perf_counter() - t0
        print(
            f"{task_name}/{dataset} done in {dt:.1f}s (n_stays={result['n_stays']:,})",
            flush=True,
        )
    print(
        f"\n=== {dataset}: all 4 tasks rebuilt in {time.perf_counter() - total_start:.1f}s ===",
        flush=True,
    )

if __name__ == "__main__":
    main()

    main()

