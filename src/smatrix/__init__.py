"""
Public entrypoint for the S-matrix / single-particle building blocks.

This package lives inside `src/`, which is added to `sys.path` by `main.py`
and tests. It is therefore importable as a top-level module:

    from smatrix import square_lattice, t, legs, ...
"""

from __future__ import annotations

# Re-export physics/model helpers that used to be reachable via `S_matrix`'s
# `from model import *` side effect.
from model import self_energy

from model.defaults import (
    alpha,
    collective_lamb_shift,
    field
)
from .self_energy_interp import (
    create_self_energy_interpolator_numba,
)
from .kinematics import coord_convert
from .propagators import sw_propagator
from .amplitudes import t, t_reg, S_disconnected, legs, connected_amplitude,L
from .tau import tau_matrix_element
from .tau_interp import create_tau_interpolator_numba, parallel_tau_matrix_grid

__all__ = [
    # model re-exports
    "self_energy",
    # defaults / constants
    "alpha",
    "field",
    "collective_lamb_shift",
    # self-energy interpolation helpers
    "create_self_energy_interpolator_numba",
    # kinematics / propagators
    "coord_convert",
    "sw_propagator",
    # amplitudes
    "t",
    "t_reg",
    "S_disconnected",
    "legs",
    "connected_amplitude",
    # two-particle tau
    "tau_matrix_element",
    "create_tau_interpolator_numba",
    "parallel_tau_matrix_grid",
    "L"
]
