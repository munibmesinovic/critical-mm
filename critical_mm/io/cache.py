"""Three-stage content-hashed cache for the harmoniser pipeline.

Stages live under `data/interim/<dataset>/<stage>/`:
- joined/ heavy joins (Dask-driven) before harmonisation
- harmonised/ the six canonical interim tables
- cohort/ task-specific bundles

A cache hit returns the cached parquet path; a miss returns None and the
caller regenerates via `cache_write`. The key combines source-file mtime +
size, a content hash of the active code/config trees, and the current git
SHA. Any drift invalidates the entry.

This is the highest-subtlety IO module: sidecar JSON is the source of truth
for "is this cached entry still valid". The parquet alone is never trusted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import xxhash

logger = logging.getLogger(__name__)

STAGE_DIRS: dict[str, str] = {
    "joined": "joined",
    "harmonised": "harmonised",
    "cohort": "cohort",
}

_TREE_HASH_SUFFIXES: frozenset[str] = frozenset({".py", ".csv", ".md"})
_TREE_HASH_SKIP_DIRS: frozenset[str] = frozenset(
    {"__pycache__", ".git", ".pytest_cache", ".venv", ".mypy_cache", ".ruff_cache"}
)

@dataclass(frozen=True)
class CacheKey:
    """All inputs that affect whether a cached parquet is still valid."""

    source_paths: tuple[Path, ...]
    source_mtimes: tuple[float, ...]
    source_sizes: tuple[int, ...]
    tree_xxhash: str
    git_sha: str | None
    extra: str

    def digest(self) -> str:
        """16-hex sha256 of the JSON-canonical form of all fields."""
        payload = {
            "source_paths": [str(p) for p in self.source_paths],
            "source_mtimes": list(self.source_mtimes),
            "source_sizes": list(self.source_sizes),
            "tree_xxhash": self.tree_xxhash,
            "git_sha": self.git_sha,
            "extra": self.extra,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_sidecar_dict(self) -> dict[str, object]:
        """Return the full sidecar payload (includes the digest)."""
        body: dict[str, object] = {"digest": self.digest()}
        body.update(asdict(self))
        body["source_paths"] = [str(p) for p in self.source_paths]
        body["source_mtimes"] = list(self.source_mtimes)
        body["source_sizes"] = list(self.source_sizes)
        return body

def compute_tree_xxhash(root: Path) -> str:
    """Return a 16-hex xxhash64 of all `.py`/`.csv`/`.md` files under `root`.

    Files are walked in deterministic sorted order; directories named in
    `_TREE_HASH_SKIP_DIRS` are skipped. Empty / unreadable files contribute
    zero bytes to the hash but their path still affects determinism.
    """
    if not root.exists():
        return xxhash.xxh64(b"").hexdigest()
    h = xxhash.xxh64()
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _TREE_HASH_SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in _TREE_HASH_SUFFIXES:
            continue
        files.append(path)
    for path in files:
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\x00")
        h.update(path.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()

def get_git_sha(repo_root: Path) -> str | None:
    """Return the short (8-hex) SHA of HEAD; soft-fail to None if unavailable.

    Returns None when:
    - `.git/` does not exist under `repo_root`
    - the `git` binary is not on PATH
    - the subprocess fails for any other reason (logs a warning)

    Working-tree-dirty state is intentionally not captured here — that's the
    job of `compute_tree_xxhash` over the active source tree.
    """
    if not (repo_root / ".git").exists():
        return None
    if shutil.which("git") is None:
        logger.warning("git binary not on PATH; cache key will use git_sha=None")
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("git rev-parse failed for %s: %s; falling back to None", repo_root, exc)
        return None
    return result.stdout.strip() or None

def build_cache_key(
    *,
    source_paths: list[Path],
    tree_roots: list[Path],
    repo_root: Path,
    extra: str = "",
) -> CacheKey:
    """Construct a CacheKey from current source mtimes/sizes + tree hash + git SHA."""
    paths = tuple(Path(p) for p in source_paths)
    mtimes: tuple[float, ...] = tuple(p.stat().st_mtime for p in paths)
    sizes: tuple[int, ...] = tuple(p.stat().st_size for p in paths)

    h = xxhash.xxh64()
    for root in tree_roots:
        h.update(str(root).encode("utf-8"))
        h.update(b"\x00")
        h.update(compute_tree_xxhash(root).encode("utf-8"))
        h.update(b"\x00")

    return CacheKey(
        source_paths=paths,
        source_mtimes=mtimes,
        source_sizes=sizes,
        tree_xxhash=h.hexdigest(),
        git_sha=get_git_sha(repo_root),
        extra=extra,
    )

def _sidecar_path(cache_path: Path) -> Path:
    return cache_path.parent / (cache_path.name + ".cache_key.json")

def cache_lookup(cache_path: Path, key: CacheKey) -> Path | None:
    """Return `cache_path` if the sidecar's digest matches; else None.

    A missing parquet, missing sidecar, malformed sidecar, or digest
    mismatch all yield None. Malformed sidecars are logged via
    `warnings.warn` so the operator notices — the cache self-heals on
    the next `cache_write`.
    """
    if not cache_path.exists():
        return None
    sidecar = _sidecar_path(cache_path)
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(
            f"cache sidecar at {sidecar} is malformed ({exc}); treating as miss",
            UserWarning,
            stacklevel=2,
        )
        return None
    if not isinstance(payload, dict) or payload.get("digest") != key.digest():
        return None
    return cache_path

def cache_write(
    cache_path: Path,
    key: CacheKey,
    writer: Callable[[Path], None],
) -> Path:
    """Run `writer(cache_path)` atomically, then drop the sidecar JSON.

    Sequence:
      1. Create the parent directory if absent.
      2. Call `writer(<cache_path>.tmp)`; the writer is responsible for
         producing a complete file at the tmp path.
      3. `Path.replace(.tmp → cache_path)` — atomic on the same filesystem.
      4. Write the sidecar JSON to `<cache_path>.cache_key.json.tmp`, then
         `Path.replace` into place. Sidecar lands AFTER the parquet so a
         crash mid-write never leaves a sidecar pointing at missing data.
      5. On any exception during 2-4, unlink the .tmp artifacts so the
         final paths stay clean.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_tmp = cache_path.parent / (cache_path.name + ".tmp")
    sidecar_path = _sidecar_path(cache_path)
    sidecar_tmp = sidecar_path.parent / (sidecar_path.name + ".tmp")
    try:
        writer(parquet_tmp)
        parquet_tmp.replace(cache_path)
        sidecar_tmp.write_text(json.dumps(key.to_sidecar_dict(), sort_keys=True, indent=2))
        sidecar_tmp.replace(sidecar_path)
    except Exception:
        for f in (parquet_tmp, sidecar_tmp):
            if f.exists():
                f.unlink(missing_ok=True)
        raise
    return cache_path
