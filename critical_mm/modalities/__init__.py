"""Builtin modality readers. Importing this package registers them."""

from __future__ import annotations

from critical_mm.modalities import diagnoses as _diagnoses
from critical_mm.modalities import notes as _notes
from critical_mm.modalities import (
    notes_encode as _notes_encode,
)

__all__ = ["_diagnoses", "_notes", "_notes_encode"]
