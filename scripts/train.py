"""CLI entry: CM-native training campaign.

Runs entirely in the critical-mm-train conda env -- no icu_benchmarks
package needed, no gin config files.

Usage:
    conda activate critical-mm-train
    python scripts/train.py # full default grid (300 cells)
    python scripts/train.py --tasks sepsis --datasets eicu # subset
    python scripts/train.py --models GRU --seeds 42 # subset
    python scripts/train.py --dry-run # print plan + exit
    python scripts/train.py --max-epochs 5 --precision bf16-mixed # cap epochs / set precision
    python scripts/train.py --extra-hyperparam hidden_dim=64 --extra-hyperparam dropout=0.1

Tasks, datasets, and model names are discovered from
``critical_mm.registry`` -- adding a new entry under
``critical_mm/contrib/`` makes it appear as a valid CLI choice without
editing this file. Defaults preserve the historical grid (5 tasks x
3 datasets x 4 DL models x 5 seeds = 300 cells).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from critical_mm.registry import discover_datasets, discover_models, discover_tasks
from critical_mm.training.config import REPO
from critical_mm.training.grid import run_grid

def _parse_hyperparam_value(raw: str) -> int | float | bool | str:
    """Parse a raw string into the most specific type that fits.

    Precedence: int → float → bool ('true'/'false') → str.
    """
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    return raw

def build_extra_hyperparams(args: argparse.Namespace) -> dict[str, object]:
    """Build the extra_hyperparams dict from the three CLI knobs.

    Merge order (later wins):
      1. ``--extra-hyperparam KEY=VALUE`` items (parsed via _parse_hyperparam_value)
      2. ``--precision VALUE`` convenience knob
      3. ``--max-epochs N`` convenience knob

    When all three flags are absent the returned dict is empty, preserving
    the historical default path bit-exactly.
    """
    result: dict[str, object] = {}

    for kv in args.extra_hyperparam or []:
        key, _, raw_val = kv.partition("=")
        result[key.strip()] = _parse_hyperparam_value(raw_val)

    if args.precision is not None:
        result["precision"] = args.precision

    if args.max_epochs is not None:
        result["max_epochs"] = args.max_epochs

    return result

def main() -> None:
    task_choices = sorted(discover_tasks().keys())
    dataset_choices = sorted(discover_datasets().keys())
    model_choices = sorted(discover_models().keys())

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tasks",
        nargs="+",
        choices=task_choices,
        default=["sepsis", "aki", "mortality24", "los", "kidney_function"],
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=dataset_choices,
        default=["eicu", "miiv", "hirid"],
    )
    ap.add_argument(
        "--models",
        nargs="+",
        choices=model_choices,
        default=["GRU", "LSTM", "TCN", "Transformer"],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--cv-repetitions", type=int, default=5)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--repetition-index", type=int, default=0)
    ap.add_argument("--fold-index", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", default=True)
    ap.add_argument("--no-cpu", dest="cpu", action="store_false")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain cells even when checkpoint dir already has metadata.json. "
        "Default behaviour preserves existing cells bit-exact.",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root dir containing data/ (default: the repo). Set to an isolated "
        "re-lock root to regenerate/retrain affected cells into a new reference "
        "without touching the locked tree.",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Cap training epochs for every cell in this run (e.g. 5 for fast FM probes). "
        "Injected as extra_hyperparams['max_epochs']. Default: None (uses resolver default=100).",
    )
    ap.add_argument(
        "--extra-hyperparam",
        action="append",
        metavar="KEY=VALUE",
        default=None,
        help="Generic key=value override injected into every cell's extra_hyperparams. "
        "VALUE is auto-cast: int → float → bool (true/false) → str. "
        "Repeatable: --extra-hyperparam hidden_dim=128 --extra-hyperparam dropout=0.1",
    )
    ap.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Lightning trainer precision string injected as extra_hyperparams['precision']. "
        "Examples: bf16-mixed, 32, 16-mixed. Default: None (auto: 16-mixed on CUDA, 32 on CPU).",
    )
    args = ap.parse_args()

    extra_hyperparams = build_extra_hyperparams(args)

    if args.dry_run:
        n = len(args.tasks) * len(args.datasets) * len(args.models) * len(args.seeds)
        print(
            f"=== train_grid plan: {len(args.tasks)} tasks x {len(args.datasets)} ds x "
            f"{len(args.models)} models x {len(args.seeds)} seeds = {n} cells ==="
        )
        for t in args.tasks:
            for d in args.datasets:
                for m in args.models:
                    for s in args.seeds:
                        print(f" {t}/{d}/{m}/seed_{s}")
        if extra_hyperparams:
            print(f" extra_hyperparams overrides: {extra_hyperparams}")
        else:
            print(" extra_hyperparams overrides: (none)")
        print("(dry-run; no cells executed)")
        sys.exit(0)

    result = run_grid(
        tasks=args.tasks,
        datasets=args.datasets,
        models=args.models,
        seeds=args.seeds,
        cv_repetitions=args.cv_repetitions,
        cv_folds=args.cv_folds,
        repetition_index=args.repetition_index,
        fold_index=args.fold_index,
        cpu=args.cpu,
        debug=args.debug,
        skip_existing=not args.force_retrain,
        data_root=args.data_root or REPO,
        extra_hyperparams=extra_hyperparams,
    )
    sys.exit(0 if result["n_fail"] == 0 else 1)

if __name__ == "__main__":
    main()
