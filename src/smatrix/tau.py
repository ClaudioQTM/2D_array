"""Two-photon tau matrix element calculations."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy import integrate

from model import c


def tau_matrix_element(E, Q, lattice, sigma_func_period):
    """Compute tau matrix element via 2D integration over the Brillouin zone."""
    Qx, Qy = Q
    bound = np.pi / lattice.a

    def integrand(qx, qy):
        sigma1 = sigma_func_period(qx, qy)
        sigma2 = sigma_func_period(Qx - qx, Qy - qy)
        return 1 / (E - 2 * lattice.omega_e - sigma1 - sigma2)

    integration_opts = {"limit": 50}

    re_integral, _ = integrate.nquad(
        lambda qx, qy: integrand(qx, qy).real,
        [[-bound, bound], [-bound, bound]],
        opts=integration_opts,
    )
    im_integral, _ = integrate.nquad(
        lambda qx, qy: integrand(qx, qy).imag,
        [[-bound, bound], [-bound, bound]],
        opts=integration_opts,
    )

    Pi = (lattice.a / (2 * np.pi)) ** 2 * (re_integral + 1j * im_integral)
    return -1 / Pi


def tau_matrix_element_polar(E, Q, lattice, sigma_func_period, n_jobs=4):
    """
    Compute tau matrix element via 2D integration over the square BZ in polar coords.

    Square region: |kx| <= pi/a, |ky| <= pi/a, centered at (0,0).
    In polar coordinates (k_abs, theta), the radial cutoff depends on theta:
        k_abs <= min((pi/a)/|cos(theta)|, (pi/a)/|sin(theta)|).
    """
    if not (lattice.omega_e < np.pi / lattice.a):
        raise ValueError(
            f"Function requires lattice.omega_e < pi/(lattice.a). "
            f"Got omega_e={lattice.omega_e:.6e}, pi/a={np.pi/lattice.a:.6e}"
        )

    Qx, Qy = Q
    bound = np.pi / lattice.a
    k_LC = lattice.omega_e / float(c)

    def integrand(k_abs, theta):
        sigma1 = sigma_func_period(k_abs * np.cos(theta), k_abs * np.sin(theta))
        sigma2 = sigma_func_period(Qx - k_abs * np.cos(theta), Qy - k_abs * np.sin(theta))
        return 1 / (E - 2 * lattice.omega_e - sigma1 - sigma2)

    def k_abs_range(theta):
        r_x = bound / abs(np.cos(theta)) if abs(np.cos(theta)) > 1e-12 else np.inf
        r_y = bound / abs(np.sin(theta)) if abs(np.sin(theta)) > 1e-12 else np.inf
        return [k_LC, min(r_x, r_y)]

    def integrand_real(k_abs, theta):
        return (integrand(k_abs, theta) * k_abs).real

    def integrand_imag(k_abs, theta):
        return (integrand(k_abs, theta) * k_abs).imag

    def run_nquad(func, ranges, opts):
        result, _ = integrate.nquad(func, ranges, opts=opts)
        return result

    integration_opts = {"limit": 150, "epsabs": 1.49e-4, "epsrel": 1.49e-4}
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_nquad)(func, ranges, integration_opts)
        for func, ranges in [
            (integrand_real, [[0.0, k_LC], [0.0, 2 * np.pi]]),
            (integrand_real, [k_abs_range, [0.0, 2 * np.pi]]),
            (integrand_imag, [[0.0, k_LC], [0.0, 2 * np.pi]]),
            (integrand_imag, [k_abs_range, [0.0, 2 * np.pi]]),
        ]
    )

    re_integral_LC, re_integral_rest, im_integral_LC, im_integral_rest = results
    Pi = (lattice.a / (2 * np.pi)) ** 2 * (
        (re_integral_LC + re_integral_rest) + 1j * (im_integral_LC + im_integral_rest)
    )
    return -1 / Pi


__all__ = ["tau_matrix_element", "tau_matrix_element_polar"]

