"""
Pytest configuration.

This codebase treats `src/` as a top-level module directory (it is added to
`sys.path` by `main.py` when running scripts). For `pytest`, we add it here so
tests can import `model`, `smatrix`, `scattering`, etc. without each test
needing to manage sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

