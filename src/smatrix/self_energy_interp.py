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


def parallel_self_energy_grid(n_points, omega, n_jobs, lattice,dim,omega_cutoff=None,omega_points=None):
    k_max = float(lattice.q / 2)
    kx_grid = np.linspace(0, k_max, n_points)
    ky_grid = np.linspace(0, k_max, n_points)

    k_points = [(kx, ky) for kx in kx_grid for ky in ky_grid]
    if dim == 3:
        omega_grid = np.linspace(omega-omega_cutoff, omega+omega_cutoff, omega_points)
        results = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(self_energy)(kx, ky, lattice.a, lattice.d, omega, alpha) for (kx, ky) in k_points for omega in omega_grid
        )
        self_energy_grid = np.array(results, dtype=complex).reshape(n_points, n_points, omega_points)
        return kx_grid, ky_grid, omega_grid, self_energy_grid
    elif dim == 2:
        results = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(self_energy)(kx, ky, lattice.a, lattice.d, omega, alpha) for (kx, ky) in k_points
        )
        self_energy_grid = np.array(results, dtype=complex).reshape(n_points, n_points)
        return kx_grid, ky_grid, self_energy_grid
    else:
        raise ValueError(f"Invalid dimension: {dim}")
    



def create_self_energy_interpolator_numba(kx_grid, ky_grid, sigma_grid, lattice, omega_grid=None):
    """
    Create a Numba-compatible periodic self-energy interpolator.

    Supports:
    - 2D `sigma_grid` with shape (nx, ny), returns `sigma(kx, ky)`.
    - 3D `sigma_grid` with shape (nx, ny, nw), requires `omega_grid`,
      returns `sigma(kx, ky, omega)`.
    """
    kx_arr = np.ascontiguousarray(kx_grid, dtype=np.float64)
    ky_arr = np.ascontiguousarray(ky_grid, dtype=np.float64)

    sigma_arr = np.asarray(sigma_grid)
    if sigma_arr.ndim not in (2, 3):
        raise ValueError(
            f"sigma_grid must be 2D or 3D, got ndim={sigma_arr.ndim} with shape={sigma_arr.shape}"
        )

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
    def trilinear_interp(x, y, w, x_min, y_min, w_min, dx, dy, dw, nx, ny, nw, z_grid):
        x = max(x_min, min(x, x_min + (nx - 1) * dx - 1e-10))
        y = max(y_min, min(y, y_min + (ny - 1) * dy - 1e-10))
        w = max(w_min, min(w, w_min + (nw - 1) * dw - 1e-10))

        ix = int((x - x_min) / dx)
        iy = int((y - y_min) / dy)
        iw = int((w - w_min) / dw)
        ix = max(0, min(ix, nx - 2))
        iy = max(0, min(iy, ny - 2))
        iw = max(0, min(iw, nw - 2))

        tx = (x - (x_min + ix * dx)) / dx
        ty = (y - (y_min + iy * dy)) / dy
        tw = (w - (w_min + iw * dw)) / dw

        z000 = z_grid[ix, iy, iw]
        z100 = z_grid[ix + 1, iy, iw]
        z010 = z_grid[ix, iy + 1, iw]
        z110 = z_grid[ix + 1, iy + 1, iw]
        z001 = z_grid[ix, iy, iw + 1]
        z101 = z_grid[ix + 1, iy, iw + 1]
        z011 = z_grid[ix, iy + 1, iw + 1]
        z111 = z_grid[ix + 1, iy + 1, iw + 1]

        c00 = z000 * (1 - tx) + z100 * tx
        c10 = z010 * (1 - tx) + z110 * tx
        c01 = z001 * (1 - tx) + z101 * tx
        c11 = z011 * (1 - tx) + z111 * tx

        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty

        return c0 * (1 - tw) + c1 * tw

    if sigma_arr.ndim == 2:
        real_grid = np.ascontiguousarray(sigma_arr.real, dtype=np.float64)
        imag_grid = np.ascontiguousarray(sigma_arr.imag, dtype=np.float64)

        if real_grid.shape != (nx, ny):
            raise ValueError(
                f"2D sigma_grid shape mismatch: expected {(nx, ny)}, got {real_grid.shape}"
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

    if omega_grid is None:
        raise ValueError("omega_grid is required when sigma_grid is 3D")

    omega_arr = np.ascontiguousarray(omega_grid, dtype=np.float64)
    nw = int(len(omega_arr))
    w_min = float(omega_arr[0])
    dw = float(omega_arr[1] - omega_arr[0])

    real_grid = np.ascontiguousarray(sigma_arr.real, dtype=np.float64)
    imag_grid = np.ascontiguousarray(sigma_arr.imag, dtype=np.float64)

    if real_grid.shape != (nx, ny, nw):
        raise ValueError(
            f"3D sigma_grid shape mismatch: expected {(nx, ny, nw)}, got {real_grid.shape}"
        )

    @njit(cache=True)
    def sigma_func_period_numba(kx, ky, omega):
        kx_bz = (kx + q / 2) % q - q / 2
        kx_bz = abs(kx_bz)
        ky_bz = (ky + q / 2) % q - q / 2
        ky_bz = abs(ky_bz)

        real_part = trilinear_interp(
            kx_bz, ky_bz, omega, kx_min, ky_min, w_min, dx, dy, dw, nx, ny, nw, real_grid
        )
        imag_part = trilinear_interp(
            kx_bz, ky_bz, omega, kx_min, ky_min, w_min, dx, dy, dw, nx, ny, nw, imag_grid
        )
        return real_part + 1j * imag_part

    return sigma_func_period_numba


__all__ = [
    "parallel_self_energy_grid",
    "create_self_energy_interpolator",
    "create_self_energy_interpolator_numba",
]

