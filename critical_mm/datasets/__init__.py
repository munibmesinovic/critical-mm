"""DatasetReader ABC and concrete readers.

Every concrete dataset reader (MIMICIVReader, NWICUReader, EICUReader,
HiRIDReader, and the synthetic sandbox) inherits from `DatasetReader` and
implements the six abstract read_* methods. The base class supplies
`harmonise_all()`, which validates and writes through the content-hash cache.

Importing this package triggers @register_dataset on every concrete reader,
so ``critical_mm.registry.discover_datasets()`` returns all five built-ins
without an explicit import per file.
"""

from __future__ import annotations

from critical_mm.datasets.base import DatasetReader
from critical_mm.datasets.eicu import EICUReader
from critical_mm.datasets.hirid import HiRIDReader
from critical_mm.datasets.mimic_iv import MIMICIVReader
from critical_mm.datasets.nwicu import NWICUReader
from critical_mm.datasets.omix import OMIXReader
from critical_mm.datasets.sicdb import SICdbReader
from critical_mm.datasets.synthetic import SyntheticReader

__all__ = [
    "DatasetReader",
    "EICUReader",
    "HiRIDReader",
    "MIMICIVReader",
    "NWICUReader",
    "OMIXReader",
    "SICdbReader",
    "SyntheticReader",
]
