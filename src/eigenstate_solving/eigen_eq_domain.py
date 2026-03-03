import numpy as np
from model.model import SquareLattice
from .eigen_eq_integrand import BZ_proj



def _make_Delta_domain(
    E: float,
    Q: np.ndarray,
    r: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    lattice: SquareLattice,
):
    """Make the domain of the integral over two photon energy difference Delta"""
    upper = E / 2 - np.linalg.norm(BZ_proj(Q - r, lattice) + H)
    lower = np.linalg.norm(r + G) - E / 2
    return lower, upper





__all__ = ["_make_Delta_domain"]
