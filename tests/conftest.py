"""Shared pytest configuration for the CRITICAL-MM test suite.

Adds ``scripts/`` to sys.path so tests can import helpers defined there
(e.g. ``build_extra_hyperparams`` from ``scripts/train.py``) without
needing to install the scripts as a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
