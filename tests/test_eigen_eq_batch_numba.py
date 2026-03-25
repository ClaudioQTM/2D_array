from __future__ import annotations

import os

import numpy as np

from eigenstate_solving.eigen_eq_integrand import (
    _make_eigen_eq_integrand,
    _make_eigen_eq_integrand_numba,
)
from smatrix import create_self_energy_interpolator_numba, square_lattice


def test_make_eigen_eq_integrand_numba_matches_legacy():
    debug = os.environ.get("EIGEN_EQ_DEBUG_VALUES") == "1"

    # Load from file (comment out if computing fresh)
    sigma_data = np.load("data/sigma_grid0f2a.npz")
    kx = sigma_data["kx"]
    ky = sigma_data["ky"]
    sigma_grid = sigma_data["sigma_grid"]
    sigma_func_period_numba = create_self_energy_interpolator_numba(
        kx, ky, sigma_grid, lattice=square_lattice
    )

    omega_e = float(square_lattice.omega_e)
    cases = [
        (
            2.0 * omega_e,
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        ),
        (
            2.2 * omega_e,
            np.array([10.0, 50.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        ),
        (
            2.6 * omega_e,
            np.array([60.0, 20.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        ),
        (
            3.0 * omega_e,
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        ),
    ]
    tEQ = np.exp(1j * np.pi / 4)

    # Cover interior points and points near box boundaries in Vegas coordinates.
    x_samples = np.array(
        [
            [-0.8, -0.8, 0.2],
            [-0.4, 0.3, 0.7],
            [0.0, 0.0, 0.5],
            [0.6, -0.2, 0.1],
            [0.9, 0.9, 0.95],
            [0.0, 0.0, 0.01],
        ],
        dtype=np.float64,
    )

    # Evaluate all three execution paths:
    # 1) legacy scalar implementation,
    # 2) numba integrand called point-by-point,
    # 3) numba integrand called in batch mode.
    for case_idx, (E, Q, G, H) in enumerate(cases):
        integrand_legacy = _make_eigen_eq_integrand(
            E, Q, G, H, square_lattice, sigma_func_period_numba, tEQ
        )
        integrand_numba = _make_eigen_eq_integrand_numba(
            E, Q, G, H, square_lattice, sigma_func_period_numba, tEQ
        )
        legacy_vals = np.array(
            [integrand_legacy(x) for x in x_samples], dtype=np.complex128
        )
        numba_vals_scalar = np.array(
            [integrand_numba(x) for x in x_samples], dtype=np.complex128
        )
        numba_vals_batch = np.asarray(integrand_numba(x_samples), dtype=np.complex128)
        if debug:
            print(f"\nCase {case_idx}: E={E}")
            print("x_samples:")
            print(x_samples)
            print("legacy_vals:")
            print(legacy_vals)
            print("numba_vals_scalar:")
            print(numba_vals_scalar)
            print("numba_vals_batch:")
            print(numba_vals_batch)

        # Require tight agreement in both scalar and batch paths for each case.
        assert np.allclose(numba_vals_scalar, legacy_vals, rtol=1e-10, atol=1e-10)
        assert np.allclose(numba_vals_batch, legacy_vals, rtol=1e-10, atol=1e-10)
