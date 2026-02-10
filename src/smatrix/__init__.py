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

from .defaults import (
    alpha,
    collective_lamb_shift,
    field,
    polar_vec1,
    polar_vec2,
    square_lattice,
)
from .self_energy_interp import (
    parallel_self_energy_grid,
    create_self_energy_interpolator,
    create_self_energy_interpolator_numba,
)
from .kinematics import coord_convert
from .propagators import sw_propagator
from .amplitudes import t, S_disconnected, legs
from .tau import tau_matrix_element, tau_matrix_element_polar

__all__ = [
    # model re-exports
    "self_energy",
    # defaults / constants
    "alpha",
    "polar_vec1",
    "polar_vec2",
    "field",
    "square_lattice",
    "collective_lamb_shift",
    # self-energy interpolation helpers
    "parallel_self_energy_grid",
    "create_self_energy_interpolator",
    "create_self_energy_interpolator_numba",
    # kinematics / propagators
    "coord_convert",
    "sw_propagator",
    # amplitudes
    "t",
    "S_disconnected",
    "legs",
    # two-particle tau
    "tau_matrix_element",
    "tau_matrix_element_polar",
]

