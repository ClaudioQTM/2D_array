"""
Self-energy grid generation and interpolation helpers.

This contains the "heavy" helpers used to precompute sigma(k) on a grid and
evaluate it efficiently in hot paths (including a Numba-friendly variant).
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from scipy.interpolate import RectBivariateSpline

from model import self_energy

from .defaults import alpha


def parallel_self_energy_grid(n_points, omega, n_jobs, lattice):
    k_max = float(lattice.q / 2)
    kx_grid = np.linspace(0, k_max, n_points)
    ky_grid = np.linspace(0, k_max, n_points)

    k_points = [(kx, ky) for kx in kx_grid for ky in ky_grid]

    results = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(self_energy)(kx, ky, lattice.a, lattice.d, omega, alpha) for (kx, ky) in k_points
    )

    self_energy_grid = np.array(results, dtype=complex).reshape(n_points, n_points)
    return kx_grid, ky_grid, self_energy_grid


def create_self_energy_interpolator(kx_grid, ky_grid, sigma_grid, lattice=None, kx=3, ky=3):
    """
    Create a periodic self-energy interpolator using scipy's RectBivariateSpline.

    If `lattice` is provided, the returned function has signature `(kx, ky)`.
    Otherwise it has signature `(kx, ky, lattice)` (matching older scripts).
    """
    real_spline = RectBivariateSpline(kx_grid, ky_grid, sigma_grid.real, kx=kx, ky=ky)
    imag_spline = RectBivariateSpline(kx_grid, ky_grid, sigma_grid.imag, kx=kx, ky=ky)

    def sigma_interp(kx_val, ky_val, grid=False):
        real_part = real_spline(kx_val, ky_val, grid=grid)
        imag_part = imag_spline(kx_val, ky_val, grid=grid)
        return real_part + 1j * imag_part

    def _sigma_func_period(kx_val, ky_val, lattice_val):
        q = float(lattice_val.q)
        kx_bz = (kx_val + q / 2) % q - q / 2
        kx_bz = abs(kx_bz)
        ky_bz = (ky_val + q / 2) % q - q / 2
        ky_bz = abs(ky_bz)
        return sigma_interp(kx_bz, ky_bz)

    if lattice is not None:
        q = float(lattice.q)

        def sigma_func_period(kx_val, ky_val):
            kx_bz = (kx_val + q / 2) % q - q / 2
            kx_bz = abs(kx_bz)
            ky_bz = (ky_val + q / 2) % q - q / 2
            ky_bz = abs(ky_bz)
            return sigma_interp(kx_bz, ky_bz)

        return sigma_func_period

    return _sigma_func_period


def create_self_energy_interpolator_numba(kx_grid, ky_grid, sigma_grid, lattice):
    """
    Create a Numba-compatible periodic self-energy interpolator using bilinear interpolation.

    Returns a Numba-compiled function `sigma_func_period_numba(kx, ky)`.
    """
    kx_arr = np.ascontiguousarray(kx_grid, dtype=np.float64)
    ky_arr = np.ascontiguousarray(ky_grid, dtype=np.float64)
    real_grid = np.ascontiguousarray(sigma_grid.real, dtype=np.float64)
    imag_grid = np.ascontiguousarray(sigma_grid.imag, dtype=np.float64)

    q = float(lattice.q)

    kx_min = float(kx_arr[0])
    ky_min = float(ky_arr[0])
    nx = int(len(kx_arr))
    ny = int(len(ky_arr))
    dx = float(kx_arr[1] - kx_arr[0])
    dy = float(ky_arr[1] - ky_arr[0])

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
    def sigma_func_period_numba(kx, ky):
        kx_bz = (kx + q / 2) % q - q / 2
        kx_bz = abs(kx_bz)
        ky_bz = (ky + q / 2) % q - q / 2
        ky_bz = abs(ky_bz)

        real_part = bilinear_interp(kx_bz, ky_bz, kx_min, ky_min, dx, dy, nx, ny, real_grid)
        imag_part = bilinear_interp(kx_bz, ky_bz, kx_min, ky_min, dx, dy, nx, ny, imag_grid)
        return real_part + 1j * imag_part

    return sigma_func_period_numba


__all__ = [
    "parallel_self_energy_grid",
    "create_self_energy_interpolator",
    "create_self_energy_interpolator_numba",
]

