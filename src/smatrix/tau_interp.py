"""
Tau matrix element grid generation helpers.

This provides a parallel helper to precompute tau(E, Q) on a 2D momentum grid,
similar to how `parallel_self_energy_grid` precomputes sigma(k).
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from .tau import tau_matrix_element


def parallel_tau_matrix_grid(n_points, E, n_jobs, lattice, sigma_func_period):
    """
    Compute tau(E, Q) on a regular 2D Q-grid in parallel.

    Parameters
    ----------
    n_points : int
        Number of grid points along each momentum axis.
    E : float
        Total photon energy used in `tau_matrix_element`.
    n_jobs : int
        Number of parallel workers for joblib.
    lattice : SquareLattice
        Lattice object providing `a` and `omega_e`.
    sigma_func_period : callable
        Periodic self-energy interpolator, typically created by
        `create_self_energy_interpolator_numba`.

    Returns
    -------
    qx_grid : ndarray
        1D array of Qx values spanning the Brillouin zone.
    qy_grid : ndarray
        1D array of Qy values spanning the Brillouin zone.
    tau_grid : ndarray
        2D complex array of shape (n_points, n_points) with
        tau(E, Q) evaluated on the grid.
    """
    # Brillouin-zone extent in each direction: |Qx|, |Qy| <= pi / a
    q_max = float(np.pi / lattice.a)
    qx_grid = np.linspace(-q_max, q_max, n_points)
    qy_grid = np.linspace(-q_max, q_max, n_points)

    q_points = [(qx, qy) for qx in qx_grid for qy in qy_grid]

    results = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(tau_matrix_element)(
            E, np.array([qx, qy], dtype=float), lattice, sigma_func_period
        )
        for (qx, qy) in q_points
    )

    tau_grid = np.array(results, dtype=complex).reshape(n_points, n_points)
    return qx_grid, qy_grid, tau_grid


__all__ = ["parallel_tau_matrix_grid"]
