"""
Defaults and lightweight constants for the S-matrix domain.

Keeping these in one place makes it easy for scripts/tests to rely on the same
default lattice and regularisation parameters as before.
"""

from __future__ import annotations

import numpy as np

from model import EMField, SquareLattice, self_energy

# Regularisation parameter used in k-space summations.
alpha = 1e-4

# Common polarisation vectors used in several scripts/notebooks.
polar_vec1 = np.array([1, 1j, 0]) / np.sqrt(2)
polar_vec2 = np.array([1, -1j, 0]) / np.sqrt(2)

field = EMField()

square_lattice = SquareLattice(
    a=0.6 * 2 * np.pi,
    omega_e=1,
    dipole_vector=np.array([1, 1j, 0]) / np.sqrt(2),
    field=field,
)

# Preserve the previous eager computation at import time.
collective_lamb_shift = self_energy(
    0,
    0,
    square_lattice.a,
    square_lattice.d,
    square_lattice.omega_e,
    alpha,
).real

__all__ = [
    "alpha",
    "polar_vec1",
    "polar_vec2",
    "field",
    "square_lattice",
    "collective_lamb_shift",
]

