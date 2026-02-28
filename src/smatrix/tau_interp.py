"""
Tau matrix element grid generation helpers.

This provides a parallel helper to precompute tau(E, Q) on a 2D momentum grid,
similar to how `parallel_self_energy_grid` precomputes sigma(k).
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from numba import njit

from .tau import tau_matrix_element


def create_tau_interpolator_numba(qx_grid, qy_grid, tau_grid, lattice):
    """
    Create a Numba-compatible periodic tau interpolator on the *full* 1st BZ.

    Notes
    -----
    Unlike the self-energy grid used in `create_self_energy_interpolator_numba`,
    the `tau_grid` is assumed to cover the entire Brillouin zone (e.g.
    qx,qy in [-pi/a, +pi/a]). Therefore we only apply periodic wrapping into
    [-q/2, +q/2] and do *not* use reflection symmetry (no abs()).
    """
    qx_arr = np.ascontiguousarray(qx_grid, dtype=np.float64)
    qy_arr = np.ascontiguousarray(qy_grid, dtype=np.float64)

    tau_arr = np.asarray(tau_grid)
    if tau_arr.ndim != 2:
        raise ValueError(
            f"tau_grid must be 2D, got ndim={tau_arr.ndim} with shape={tau_arr.shape}"
        )

    q = float(lattice.q)

    qx_min = float(qx_arr[0])
    qy_min = float(qy_arr[0])
    nx = int(len(qx_arr))
    ny = int(len(qy_arr))
    dx = float(qx_arr[1] - qx_arr[0])
    dy = float(qy_arr[1] - qy_arr[0])

    real_grid = np.ascontiguousarray(tau_arr.real, dtype=np.float64)
    imag_grid = np.ascontiguousarray(tau_arr.imag, dtype=np.float64)

    if real_grid.shape != (nx, ny):
        raise ValueError(
            f"2D tau_grid shape mismatch: expected {(nx, ny)}, got {real_grid.shape}"
        )

    @njit(cache=True)
    def bilinear_interp(x, y, x_min, y_min, dx, dy, nx, ny, z_grid):
        x = max(x_min, min(x, x_min + (nx - 1) * dx - 1e-10))
        y = max(y_min, min(y, y_min + (ny - 1) * dy - 1e-10))

        ix = int((x - x_min) / dx)
        iy = int((y - y_min) / dy)
        ix = max(0, min(ix, nx - 2))
        iy = max(0, min(iy, ny - 2))

        tx = (x - (x_min + ix * dx)) / dx
        ty = (y - (y_min + iy * dy)) / dy

        z00 = z_grid[ix, iy]
        z10 = z_grid[ix + 1, iy]
        z01 = z_grid[ix, iy + 1]
        z11 = z_grid[ix + 1, iy + 1]

        return (
            z00 * (1 - tx) * (1 - ty)
            + z10 * tx * (1 - ty)
            + z01 * (1 - tx) * ty
            + z11 * tx * ty
        )

    @njit(cache=True)
    def tau_func_period_numba(qx, qy):
        # Map to the first Brillouin zone [-q/2, q/2] (periodic wrapping only)
        qx_bz = (qx + q / 2) % q - q / 2
        qy_bz = (qy + q / 2) % q - q / 2

        real_part = bilinear_interp(
            qx_bz, qy_bz, qx_min, qy_min, dx, dy, nx, ny, real_grid
        )
        imag_part = bilinear_interp(
            qx_bz, qy_bz, qx_min, qy_min, dx, dy, nx, ny, imag_grid
        )
        return real_part + 1j * imag_part

    return tau_func_period_numba


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


__all__ = ["create_tau_interpolator_numba", "parallel_tau_matrix_grid"]
