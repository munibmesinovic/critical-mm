"""Embed the notes text surface -> per-note vectors.

Dataset -> encoder: miiv/eicu -> modernbert_clinical_en; omix -> bge_large_zh.
Output (gitignored): data/processed/_modalities/<ds>/notes_emb/<encoder_id>@<rev>/
  part_00000.parquet, part_00001.parquet, ... (one part per shard)
  Each part: columns note_id (str), embedding (list[f32]), n_tokens (int)
  Consumers: pl.read_parquet(dir) # polars reads all parts
Resumable: skips note_ids already in any existing part-file.
GPU; run via scripts/run_embed_notes.sh.

Usage: python scripts/embed_notes.py [--datasets miiv omix eicu] [--limit N] [--shard-size N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from critical_mm.modalities.notes_base import NoteEncoder

REPO = Path(__file__).resolve().parent.parent
DATASET_ENCODER = {
    "miiv": "modernbert_clinical_en",
    "eicu": "modernbert_clinical_en",
    "omix": "bge_large_zh",
}

def embed_frame(timed: pl.DataFrame, encoder: NoteEncoder, existing_ids: set[str]) -> pl.DataFrame:
    """Encode the not-yet-embedded rows of a notes_timed frame. Pure + testable.

    ``encoder.encode(texts)`` returns ``(emb ndarray[n,d], n_tokens ndarray[n])``.
    """
    todo = timed.filter(~pl.col("note_id").is_in(list(existing_ids))) if existing_ids else timed
    if todo.height == 0:
        return pl.DataFrame(
            schema={"note_id": pl.Utf8, "embedding": pl.List(pl.Float32), "n_tokens": pl.Int64}
        )
    result: Any = encoder.encode(todo["text"].to_list())
    emb, n_tok = result
    return pl.DataFrame(
        {
            "note_id": todo["note_id"],
            "embedding": pl.Series(emb.tolist(), dtype=pl.List(pl.Float32)),
            "n_tokens": pl.Series([int(x) for x in n_tok], dtype=pl.Int64),
        }
    )

def _existing_ids_from_dir(out_dir: Path) -> set[str]:
    parts = sorted(out_dir.glob("part_*.parquet"))
    if not parts:
        return set()
    return set(pl.read_parquet(parts, columns=["note_id"])["note_id"].to_list())

def embed_dataset(
    timed: pl.DataFrame, encoder: NoteEncoder, out_dir: Path, *, shard_size: int = 20000
) -> int:
    """Embed all not-yet-embedded notes in ``timed`` into ``out_dir`` as part-files,
    one part per ``shard_size`` notes (incremental checkpoint; resumable mid-dataset).
    Returns the number of newly embedded notes. ``timed`` needs columns note_id, text.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_ids_from_dir(out_dir)
    todo = timed.filter(~pl.col("note_id").is_in(list(existing))) if existing else timed
    n_existing_parts = len(list(out_dir.glob("part_*.parquet")))
    total = 0
    for k, start in enumerate(range(0, todo.height, shard_size)):
        shard = todo.slice(start, shard_size)
        new = embed_frame(shard, encoder, existing_ids=set())
        if new.height == 0:
            continue
        part_idx = n_existing_parts + k
        new.write_parquet(out_dir / f"part_{part_idx:05d}.parquet")
        total += new.height
    return total

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["miiv", "omix", "eicu"])
    ap.add_argument(
        "--limit", type=int, default=0, help="0 = all; else cap notes per dataset (smoke)"
    )
    ap.add_argument("--batch-size", type=int, default=0, help="0 = encoder default")
    ap.add_argument("--shard-size", type=int, default=20000, help="notes per part-file checkpoint")
    args = ap.parse_args()

    from critical_mm.modalities.notes_base import NoteEncoder as _NoteEncoder
    from critical_mm.registry import get_note_encoder

    mod_root = REPO / "data" / "processed" / "_modalities"
    cache: dict[str, _NoteEncoder] = {}
    for ds in args.datasets:
        timed_p = mod_root / ds / "notes_timed.parquet"
        if not timed_p.exists():
            print(f"[skip] {ds}: no notes_timed.parquet")
            continue
        enc_id = DATASET_ENCODER[ds]
        if enc_id not in cache:
            cls = get_note_encoder(enc_id)
            cache[enc_id] = cls(batch_size=args.batch_size) if args.batch_size else cls()
        encoder = cache[enc_id]
        rev = getattr(encoder, "revision", "na")
        out_dir = mod_root / ds / "notes_emb" / f"{enc_id}@{rev}"

        timed = pl.read_parquet(timed_p, columns=["note_id", "text"])
        if args.limit:
            timed = timed.head(args.limit)

        n = embed_dataset(timed, encoder, out_dir, shard_size=args.shard_size)
        n_parts = len(list(out_dir.glob("part_*.parquet")))
        if n == 0:
            print(f"[done] {ds}: all embedded (dir total parts={n_parts}) -> {enc_id}@{rev}/")
        else:
            print(f"[ok] {ds}: +{n} new (dir total parts={n_parts}) -> {enc_id}@{rev}/")

if __name__ == "__main__":
    main()
