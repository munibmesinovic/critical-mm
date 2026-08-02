
def export_yaib(
    *,
    task: str,
    dataset: str,
    base_cohort_root: Path,
    task_output_root: Path,
    yaib_output_root: Path,
) -> dict[str, Path]:
    """Read CRITICAL-MM task outputs and emit YAIB-shaped parquets.

    `base_cohort_root` is the directory containing per-dataset base cohort
    parquets (e.g. `processed/base_cohort/<dataset>/stays.parquet`). It is
    consulted only for the original ``sex`` String values; the task
    pipeline encodes sex numerically and the original strings are not
    recoverable from `sta.parquet` alone.

    `task_output_root` is the parent of `<task>/<dataset>/sta.parquet` —
    typically the same `processed_root` passed to `Task.build()`.

    `yaib_output_root` receives `<task>/<dataset>/{sta,dyn,outc}.parquet`.

    Returns ``{"sta_path": ..., "dyn_path": ..., "outc_path": ...}``.
    """
    task_dir = Path(task_output_root) / task / dataset
    cohort_path = Path(base_cohort_root) / dataset / "stays.parquet"
    out_dir = Path(yaib_output_root) / task / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("sta.parquet", "dyn.parquet", "outc.parquet"):
        if not (task_dir / name).exists():
            raise FileNotFoundError(f"task output missing: {task_dir / name}")
    if not cohort_path.exists():
        raise FileNotFoundError(f"base cohort missing: {cohort_path}")

    sta = pl.scan_parquet(task_dir / "sta.parquet")
    dyn = pl.scan_parquet(task_dir / "dyn.parquet")
    outc = pl.scan_parquet(task_dir / "outc.parquet")
    cohort = pl.scan_parquet(cohort_path).select("stay_id", "sex")

    out_sta = _reshape_sta(sta, cohort).collect()
    out_dyn = _reshape_dyn(dyn).collect()
    out_outc = _reshape_outc(outc).collect()

    sta_path = out_dir / "sta.parquet"
    dyn_path = out_dir / "dyn.parquet"
    outc_path = out_dir / "outc.parquet"
    out_sta.write_parquet(sta_path, compression="zstd")
    out_dyn.write_parquet(dyn_path, compression="zstd")
    out_outc.write_parquet(outc_path, compression="zstd")

    return {"sta_path": sta_path, "dyn_path": dyn_path, "outc_path": outc_path}

def verify_yaib_compatibility(
    *,
    exported: dict[str, Path],
    reference: dict[str, Path],
    row_count_tolerance: float = 0.05,
) -> bool:
    """Compare exported parquets against a known-good YAIB reference.

    Returns True if every (sta, dyn, outc) pair has matching column names,
    matching dtypes, and row counts within ``row_count_tolerance``.

    Mismatches are surfaced as printed lines so a failing run still leaves
    an audit trail (callers may pipe stdout into a report). For deeper
    introspection use the per-frame schema comparison directly.
    """
    for key in ("sta_path", "dyn_path", "outc_path"):
        e_path = exported[key]
