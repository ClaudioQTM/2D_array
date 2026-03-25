"""Two-photon tau matrix element calculations."""

from __future__ import annotations

import numpy as np
from scipy import integrate



def tau_matrix_element(E, Q, lattice, sigma_func_period):
    """Compute tau matrix element via 2D integration over the Brillouin zone."""
    Qx, Qy = Q
    bound = np.pi / lattice.a

    def integrand(qx, qy):
        sigma1 = sigma_func_period(qx, qy)
        sigma2 = sigma_func_period(Qx - qx, Qy - qy)
        return 1 / (E - 2 * lattice.omega_e - sigma1 - sigma2)

    integration_opts = {"limit": 500}

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
#   Pedersen's convention
#    Pi = (lattice.a / (2 * np.pi)) ** 2 * (re_integral + 1j * im_integral)
#   My convention
    Pi = 0.5 / (2 * np.pi)**2 * (re_integral + 1j * im_integral)
    return -1 / Pi


__all__ = ["tau_matrix_element"]
