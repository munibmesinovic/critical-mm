"""One definition of "which cohort is this", used by the lock, the trainer and resume.

WHY THIS EXISTS

Before 2026-07-28 the locked-split manifest hashed **only `outc.parquet`**, and the
trainer copied that single hash into every checkpoint as its provenance. Two real
defects walked straight through that gap:

* The kidney_function window fix changes **`dyn` only** (hour bucket 24 is dropped).
  On 4 of 7 datasets the rebuilt `outc` was byte-identical, so a provably stale
  checkpoint passed every fingerprint check in the repository.
* The OMIX static-panel fix changes **`sta` only** (`sex_male` went from a constant
  0.5 placeholder to real values). Same blindness.

Compounding it, the grid drivers resumed on bare file existence — see
``scripts/sp2_grid_patient.py`` — so a re-run intended to replace 336 stale
kidney_function cells would have trained nothing and reported success.

The rule this module enforces: **a checkpoint is current only if every input file it
was trained on still hashes to what it recorded.** `sta`, `dyn` and `outc` all count.
Adding a segment here automatically tightens the lock, the trainer and every resume
gate, because they all call this.

Cheap by construction: three file hashes per cohort, memoised per process.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

SEGMENTS: tuple[str, ...] = ("sta", "dyn", "outc")

def segment_key(segment: str) -> str:
    return f"{segment}_content_sha256"

@lru_cache(maxsize=4096)
def _sha256(path: str, mtime_ns: int, size: int) -> str:
    """Hash a file. mtime/size are cache-busting parameters, not inputs to the digest."""
    del mtime_ns, size
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def file_sha256(path: Path) -> str:
    st = path.stat()
    return _sha256(str(path), st.st_mtime_ns, st.st_size)

def cohort_fingerprint(cohort_dir: Path, *, require_all: bool = True) -> dict[str, str]:
    """Return {sta,dyn,outc}_content_sha256 for a processed cohort directory.

    `cohort_dir` is ``data/processed/<task>/<dataset>``. A missing segment raises
    unless ``require_all=False``, in which case it is simply absent from the result —
    used by callers that must tolerate partially-built trees.
    """
    out: dict[str, str] = {}
    for seg in SEGMENTS:
        p = cohort_dir / f"{seg}.parquet"
        if not p.exists():
            if require_all:
                raise FileNotFoundError(f"cohort segment missing: {p}")
            continue
        out[segment_key(seg)] = file_sha256(p)
    return out

def fingerprint_matches(recorded: dict[str, Any], current: dict[str, str]) -> tuple[bool, list[str]]:
    """Compare a recorded fingerprint against a freshly computed one.

    Returns ``(ok, mismatched_keys)``. A key the recorded fingerprint does not carry
    counts as a **mismatch**, not a pass: checkpoints written before this module
    existed recorded no `dyn`/`sta` hash, and treating that as "fine" is exactly the
    hole this closes. 126 of the 336 kidney_function cells on disk at the time carried
    no fingerprint at all.
    """
    bad: list[str] = []
    for key, value in current.items():
        if recorded.get(key) != value:
            bad.append(key)
    return (not bad), bad

def describe_mismatch(cohort_dir: Path, recorded: dict[str, Any], mismatched: list[str]) -> str:
    lines = [f"cohort {cohort_dir} has changed since this artefact was written:"]
    for key in mismatched:
        was = recorded.get(key)
        lines.append(f"  {key}: recorded {str(was)[:12] if was else '(absent)'} -> now differs")
    return "\n".join(lines)

def split_fingerprint(splits_dir: Path) -> dict[str, Any]:
    """The split-level half of a checkpoint's provenance, read from its manifest.

    Mirrors what ``critical_mm.training.train._splits_fingerprint`` records, so a
    caller can pass it to ``checkpoint_is_current`` as ``split_expectation``.
    """
    import hashlib
    import json

    mp = splits_dir / "manifest.json"
    if not mp.is_file():
        return {}
    m = json.loads(mp.read_text())
    h = hashlib.sha256()
    for fold in m.get("folds", []):
        h.update(fold["content_sha256"].encode())
    return {"folds_combined_sha256": h.hexdigest(), "n_stays_total": m.get("n_stays_total")}

def code_fingerprint(paths: list[Path]) -> str:
    """Hash the SOURCE that produces an artefact, for artefacts a cohort hash cannot see.

    The `treatments_present` fix changed ``critical_mm/fusion/blocks.py`` and no cohort
    byte, so no cohort fingerprint could tell a cell trained with the leaky presence bit
    from one trained without it. Every gate in the repo would have called those 210 cells
    current forever. Hashing the builder closes that: change the code, invalidate the
    cells that depend on it.
    """
    import hashlib

    h = hashlib.sha256()
    for p in sorted(paths):
        if not p.is_file():
            raise FileNotFoundError(
                f"code_fingerprint path does not exist: {p}. A typo'd path used to "
                "contribute the constant b'MISSING', making the hash stable and wrong."
            )
        h.update(p.name.encode())
        h.update(file_sha256(p).encode())
    return h.hexdigest()[:16]

def checkpoint_is_current(
    meta_path: Path,
    cohort_dir: Path,
    *,
    split_expectation: dict[str, Any] | None = None,
    code_expectation: str | None = None,
    require_split: bool = False,
    require_code: bool = False,
) -> tuple[bool, str]:
    """Is the artefact at ``meta_path`` trained on the cohort now at ``cohort_dir``?

    This replaces the ``metadata.json exists`` test the grid drivers used to resume on.
    That test made a re-run intended to replace 336 stale kidney_function cells train
    nothing and report success.

    Returns ``(is_current, reason)``. **Absence is not a pass**: a checkpoint with no
    metadata, no fingerprint, or a fingerprint lacking a segment hash is reported stale,
    because that is exactly the state the pre-2026-07-28 checkpoints are in and treating
    it as current is the bug. Callers that must tolerate legacy artefacts should do so
    explicitly and visibly, not by weakening this.

    ``require_split``/``require_code`` guard the OTHER half of that same bug: a missing
    *expectation* used to be a silent pass, not just a missing *stored* hash. Every
    caller but ``scripts/treatments_grid.py`` left ``split_expectation``/
    ``code_expectation`` at the default ``None``, which skipped those axes entirely and
    returned "current" for a checkpoint carrying no fingerprint at all. Passing
    ``require_split=True`` (or ``require_code=True``) without the corresponding
    expectation is a caller bug, not a state to report on the cohort, so it raises
    instead of returning ``(False, ...)``.
    """
    import json

    if require_split and split_expectation is None:
        raise ValueError(
            "require_split=True but no split_expectation was passed. A missing "
            "expectation used to be a SILENT PASS: every caller but treatments_grid "
            "left it None and got 'current' for checkpoints carrying no split hash. "
            "Pass split_fingerprint(splits_dir)."
        )
    if require_code and code_expectation is None:
        raise ValueError(
            "require_code=True but no code_expectation was passed. See require_split."
        )

    if not meta_path.exists():
        return False, "no metadata.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        return False, f"unreadable metadata.json ({exc.__class__.__name__})"

    recorded = meta.get("splits_fingerprint") or {}
    if not recorded:
        return False, "no splits_fingerprint recorded"

    try:
        current = cohort_fingerprint(cohort_dir, require_all=False)
    except OSError as exc:
        return False, f"cohort unreadable ({exc.__class__.__name__})"
    if not current:
        return False, f"no cohort segments found under {cohort_dir}"

    expected = [segment_key(s) for s in SEGMENTS]
    absent_on_disk = [k for k in expected if k not in current]
    if absent_on_disk:
        return False, "cohort is missing " + ", ".join(absent_on_disk)

    missing = [k for k in expected if k not in recorded]
    if missing:
        return False, "fingerprint predates segment hashing (missing " + ", ".join(missing) + ")"

    ok, bad = fingerprint_matches(recorded, current)
    if not ok:
        return False, "cohort changed: " + ", ".join(bad)

    for key in ("folds_combined_sha256", "n_stays_total"):
        want = split_expectation.get(key) if split_expectation else None
        if want is None:
            if require_split:
                return False, f"split expectation missing {key}"
            continue
        if recorded.get(key) != want:
            return False, f"split changed: {key}"

    if code_expectation is not None:
        got = meta.get("code_fingerprint")
        if got != code_expectation:
            return False, (
                "builder code changed since training"
                f" (recorded {got or 'nothing'}, now {code_expectation})"
            )

    return True, "current"
