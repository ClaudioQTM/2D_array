"""Compare Cartesian and polar tau implementations at fixed total energy."""

import numpy as np

from smatrix import (
    create_self_energy_interpolator_numba,
    square_lattice,
    tau_matrix_element,
    tau_matrix_element_polar,
    collective_lamb_shift,
)


def _build_sigma_func_period(lattice):
    """Create a lightweight periodic self-energy interpolator for testing."""
    k_half = float(lattice.q / 2.0)
    kx_grid = np.array([0.0, k_half], dtype=np.float64)
    ky_grid = np.array([0.0, k_half], dtype=np.float64)
    sigma_grid = np.zeros((2, 2), dtype=np.complex128)
    return create_self_energy_interpolator_numba(
        kx_grid, ky_grid, sigma_grid, lattice=lattice
    )


def test_tau_agreement_fixed_energy_100_points():
    """Sample 100 points in the 1st BZ at fixed E and compare tau values."""
    lattice = square_lattice
    sigma_func_period = _build_sigma_func_period(lattice)

    # Fix total energy E for all sampled Q points.
    E = 2.0 * lattice.omega_e + collective_lamb_shift
    bound = np.pi / lattice.a

    # Randomized 100 points in the 1st Brillouin zone (reproducible seed).
    rng = np.random.default_rng(42)
    Q_points = rng.uniform(-bound, bound, size=(400, 2)).astype(np.float64)

    rel_diffs = []
    failed_points = []
    atol = 1e-5
    rtol = 5e-2

    for Q in Q_points:
        tau_cart = tau_matrix_element(E, Q, lattice, sigma_func_period)
        tau_pol = tau_matrix_element_polar(E, Q, lattice, sigma_func_period, n_jobs=4)

        if np.isclose(abs(tau_cart), 0.0):
            rel_diff = abs(tau_cart - tau_pol)
        else:
            rel_diff = abs(tau_cart - tau_pol) / abs(tau_cart)

        rel_diffs.append(rel_diff)

        if not np.isclose(tau_cart, tau_pol, rtol=rtol, atol=atol):
            failed_points.append((Q, tau_cart, tau_pol, rel_diff))

    max_rel_diff = float(np.max(rel_diffs)) if rel_diffs else 0.0

    assert not failed_points, (
        f"tau_matrix_element and tau_matrix_element_polar disagree at "
        f"{len(failed_points)}/{len(Q_points)} points. "
        f"max_rel_diff={max_rel_diff:.3e}, "
        f"example={failed_points[0]}"
    )
