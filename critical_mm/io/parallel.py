"""Calibrate polars's thread pool so concurrent workers don't oversubscribe.

`POLARS_MAX_THREADS` is read only when polars first initialises its
runtime. If polars is already imported when we change the env var, the
new value may not apply; the function emits a `UserWarning` in that case.
"""

from __future__ import annotations

import os
import sys
import warnings


def calibrate_thread_pool(workers: int = 1) -> int:
    """Divide the host CPU budget across `workers` concurrent processes.

    Sets `POLARS_MAX_THREADS` to `max(1, os.cpu_count() // workers)` and
    returns the chosen per-worker thread count.

    Call this BEFORE the first `import polars` in each worker process —
    polars caches the runtime thread-pool size at import time. If polars
    is already in `sys.modules` when this runs, the change may not take
    effect; we emit a `UserWarning` so the caller can investigate.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    cpu_count = os.cpu_count() or 1
    per_worker = max(1, cpu_count // workers)
    os.environ["POLARS_MAX_THREADS"] = str(per_worker)
    if "polars" in sys.modules:
        warnings.warn(
            "polars is already imported; POLARS_MAX_THREADS may not take effect "
            "in this process. Call calibrate_thread_pool() before importing polars.",
            UserWarning,
            stacklevel=2,
        )
    return per_worker
