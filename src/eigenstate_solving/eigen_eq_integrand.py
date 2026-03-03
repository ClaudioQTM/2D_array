import numpy as np
from model.model import SquareLattice
from smatrix.amplitudes import t, legs


def BZ_proj(v: np.ndarray, lattice: SquareLattice):
    """Project v into the first Brillouin zone relative to Q.

    Parameters
    ----------
    v : array of shape (2,)
    Q : array of shape (2,)
    """
    v = np.asarray(v, dtype=float)
    q = float(lattice.q)
    return ((v + q / 2) % q) - q / 2



def _eigen_eq_integrand(
    E: float,
    Q: np.ndarray,
    r: np.ndarray,
    D: float,
    G: np.ndarray,
    H: np.ndarray,
    lattice: SquareLattice,
    sigma_func_period,
    tEQ: float,
):
    E1 = E + D / 2  # energy of the first photon
    E2 = E - D / 2  # energy of the second photon
    rG = r + G
    sH = BZ_proj(Q - r, lattice) + H
    rz = np.sqrt(E1**2 - np.linalg.norm(rG) ** 2)
    sz = np.sqrt(E2**2 - np.linalg.norm(sH) ** 2)

    jacobian_1 = E1 / rz
    jacobian_2 = E2 / sz

    constant_factor = -1j / 2 * (2 * np.pi) ** 3 / lattice.a**4

    transmission_factors = 1 / (t(rG, E1, lattice) * t(sH, E2, lattice) - tEQ)

    leg_factors = legs(rG, E1, sH, E2, lattice, sigma_func_period, direction="in") * legs(
        rG, E1, sH, E2, lattice, sigma_func_period, direction="out"
    )

    return (
        jacobian_1 * jacobian_2 * constant_factor * transmission_factors * leg_factors
    )

