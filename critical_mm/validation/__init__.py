"""Validation helpers: cohort fingerprinting, checkpoint currency, and cohort sanity checks.

Imports are deliberately lazy -- import the submodule you need rather than relying on names
re-exported here, so that importing :mod:`critical_mm.validation` stays cheap and cannot fail
because of an unrelated submodule.

    from critical_mm.validation.cohort_fingerprint import checkpoint_is_current
    from critical_mm.validation.checkpoint_paths import resolve_checkpoint
    from critical_mm.validation import cohort_checks

:mod:`~critical_mm.validation.cohort_fingerprint` is the one to reach for when deciding whether a
trained cell may be reused. Checkpoint existence records only that a cell trained at some point,
not what it trained on; :func:`~critical_mm.validation.cohort_fingerprint.checkpoint_is_current`
compares cohort hashes, split hashes and a builder-code hash instead. A missing hash is treated as
stale rather than as a pass. Re-lock splits after rebuilding a cohort and before retraining, or
cells will copy the stale hash from their manifest.
"""

from __future__ import annotations

__all__ = [
    "checkpoint_paths",
    "cohort_checks",
    "cohort_fingerprint",
]
