"""Rebuild the canonical interim parquets for one dataset.

Calls ``Reader.harmonise_all()`` which goes through the content-hash
cache: every table whose key has changed (code edit, raw-file mtime
change, git SHA change) is regenerated; unchanged tables short-circuit.

Use this whenever a reader-side code change must propagate into the
``data/interim/<dataset>/*.parquet`` cache. ``rebuild_nonsepsis_tasks.py``
and ``Sepsis().build()`` both read those parquets directly (no implicit
cache miss), so the interim rebuild has to be triggered explicitly.

Wrap in scripts/mem_run.sh for memory safety:

    MEM_MAX=110G bash scripts/mem_run.sh harmonise-eicu \\
      python3 \\
      scripts/rebuild_interim.py eicu

Per-dataset MEM_MAX (empirically safe under cold cache):
    eicu  → 110G    (events_long pivot dominates)
    miiv  → 80G
    hirid → 100G
    nwicu → 30G
    sicdb → ~60G (data_float_h scan)  [update after first build]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from critical_mm.datasets.base import DatasetReader
from critical_mm.datasets.eicu import EICUReader
from critical_mm.datasets.hirid import HiRIDReader
from critical_mm.datasets.mimic_iv import MIMICIVReader
from critical_mm.datasets.nwicu import NWICUReader
from critical_mm.datasets.omix import OMIXReader
from critical_mm.datasets.sicdb import SICdbReader
from critical_mm.datasets.zigong import ZigongReader

READERS: dict[str, tuple[type[DatasetReader], str]] = {
    "eicu": (EICUReader, "data/raw/eicu-crd-2.0"),
    "miiv": (MIMICIVReader, "data/raw/mimic-iv-3.1"),
    "hirid": (HiRIDReader, "data/raw/hirid-1.1.1"),
    "nwicu": (NWICUReader, "data/raw/nwicu-0.1.0"),
    "omix": (OMIXReader, "data/raw/OMIX005817"),
    "sicdb": (
        SICdbReader,
        "data/raw/sicdb/salzburg-intensive-care-database-sicdb-a-freely-accessible-intensive-care-database-1.0.8",
    ),
    "zigong": (ZigongReader, "data/raw/zigong/DataTables"),
}

def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in READERS:
        print(
            f"usage: rebuild_interim.py {{{'|'.join(READERS)}}}",
            file=sys.stderr,
        )
        sys.exit(2)

    dataset = sys.argv[1]
    cls, raw_rel = READERS[dataset]
    repo = Path("$CRITICAL_MM_REPO")
    reader = cls(
        raw_root=repo / raw_rel,
        interim_root=repo / "data" / "interim",
        repo_root=repo,
    )

    print(f"=== harmonise_all({dataset}) ===", flush=True)
    t0 = time.perf_counter()
    results = reader.harmonise_all()
    dt = time.perf_counter() - t0
    print(f"\n=== done in {dt:.1f}s ===", flush=True)
    for table, path in results.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {table:14s} -> {path} ({size_mb:,.1f} MB)", flush=True)

if __name__ == "__main__":
    main()

