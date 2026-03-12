from __future__ import annotations

import numpy as np

from eigenstate_solving.eigen_eq_integrand import (
    _make_eigen_eq_integrand,
    _make_eigen_eq_integrand_numba,
)
from smatrix import square_lattice


def _zero_sigma_interpolator():
    def _sigma_func_period(kx, ky):
        _ = kx, ky
        return 0.0 + 0.0j

    return _sigma_func_period


def test_make_eigen_eq_integrand_numba_matches_legacy():
    # Use a deterministic, simple self-energy so this test isolates integrand logic only.
    sigma_func = _zero_sigma_interpolator()
    # Sweep multiple reciprocal-lattice grid points for Q, G, H and varying energies.
    q = float(square_lattice.q)
    omega_e = float(square_lattice.omega_e)
    cases = [
        (
            2.0 * omega_e,
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([q, -q], dtype=np.float64),
        ),
        (
            2.2 * omega_e,
            np.array([q, 0.0], dtype=np.float64),
            np.array([-q, q], dtype=np.float64),
            np.array([0.0, q], dtype=np.float64),
        ),
        (
            2.6 * omega_e,
            np.array([-q, q], dtype=np.float64),
            np.array([2.0 * q, -q], dtype=np.float64),
            np.array([-2.0 * q, 0.0], dtype=np.float64),
        ),
        (
            3.0 * omega_e,
            np.array([2.0 * q, -2.0 * q], dtype=np.float64),
            np.array([q, q], dtype=np.float64),
            np.array([-q, -q], dtype=np.float64),
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
        ],
        dtype=np.float64,
    )

    # Evaluate all three execution paths:
    # 1) legacy scalar implementation,
    # 2) numba integrand called point-by-point,
    # 3) numba integrand called in batch mode.
    for E, Q, G, H in cases:
        integrand_legacy = _make_eigen_eq_integrand(
            E, Q, G, H, square_lattice, sigma_func, tEQ
        )
        integrand_numba = _make_eigen_eq_integrand_numba(
            E, Q, G, H, square_lattice, sigma_func, tEQ
        )
        legacy_vals = np.array(
            [integrand_legacy(x) for x in x_samples], dtype=np.complex128
        )
        numba_vals_scalar = np.array(
            [integrand_numba(x) for x in x_samples], dtype=np.complex128
        )
        numba_vals_batch = np.asarray(integrand_numba(x_samples), dtype=np.complex128)

        # Require tight agreement in both scalar and batch paths for each case.
        assert np.allclose(numba_vals_scalar, legacy_vals, rtol=1e-10, atol=1e-10)
        assert np.allclose(numba_vals_batch, legacy_vals, rtol=1e-10, atol=1e-10)


