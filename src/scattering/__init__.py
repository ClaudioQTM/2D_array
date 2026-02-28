"""
Public entrypoint for two-photon scattering integral routines.

Importable as a top-level module because `src/` is added to `sys.path`:

    from scattering import scattering_integral_vegas, _make_integrand_and_bounds
"""

from __future__ import annotations

from .api import (
    disconnected_scattering_integral,
    scattering_integral,
    scattering_integral_nquad,
    scattering_integral_qmc,
    scattering_integral_vegas,
)
from .integrand import _make_integrand_and_bounds

__all__ = [
    "disconnected_scattering_integral",
    "scattering_integral",
    "scattering_integral_nquad",
    "scattering_integral_qmc",
    "scattering_integral_vegas",
    "_make_integrand_and_bounds",
]
