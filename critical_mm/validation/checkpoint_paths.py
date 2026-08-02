"""One definition of "which file holds this cell's weights".

WHY THIS EXISTS

Lightning's ``ModelCheckpoint(filename="model")`` does not overwrite. A second fit
into the same directory writes ``model-v1.ckpt`` and leaves ``model.ckpt`` in place.
Every reader in this repository opened the hard-coded ``model.ckpt``, so after the
2026-07-28 retrain campaign **322 cell directories** held current metadata beside
superseded weights — including 105 ``LSTM__treatments`` cells whose ``model.ckpt``
predates the ``treatments_present`` leak fix, and 180 main-grid cells on rebuilt
cohorts. Nothing compared the loaded file against its sibling ``metadata.json``, so a
re-score on those weights produces plausible, wrong numbers and no error.

Writers now pass ``enable_version_counter=False`` so new runs overwrite. This module
exists for the trees already on disk, and as the single place a reader may name a
checkpoint file.
"""

from __future__ import annotations

import re
from pathlib import Path

_VERSION_RE = re.compile(r"^(?P<stem>.+?)(?:-v(?P<n>\d+))?\.ckpt$")

class CheckpointShadowError(RuntimeError):
    """Version order and mtime order disagree — refuse rather than guess."""

def resolve_checkpoint(ckpt_dir: Path, stem: str = "model") -> Path:
    """Return the newest checkpoint for ``stem`` in ``ckpt_dir``.

    Lightning increments the ``-vN`` suffix, so the highest N is newest. We verify
    that against mtime and raise if they disagree, because a hand-moved file is
    exactly the situation where silently picking one ships stale weights.
    """
    candidates: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob(f"{stem}*.ckpt"):
        m = _VERSION_RE.match(p.name)
        if m is None or m.group("stem") != stem:
            continue
        candidates.append((int(m.group("n") or 0), p))

    if not candidates:
        raise FileNotFoundError(f"no {stem!r} checkpoint under {ckpt_dir}")

    candidates.sort(key=lambda t: t[0])
    version, newest = candidates[-1]
    if version > 0:
        newest_mtime = newest.stat().st_mtime
        for other_v, other in candidates[:-1]:
            if other.stat().st_mtime > newest_mtime:
                raise CheckpointShadowError(
                    f"{newest.name} is version {version} but {other.name} "
                    f"(version {other_v}) has a later mtime in {ckpt_dir}; "
                    "refusing to guess which holds current weights"
                )
    return newest
