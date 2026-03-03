import numpy as np
from .eigen_eq_integrand import _eigen_eq_integrand, BZ_proj
from .eigen_eq_domain import _make_Delta_domain
from model.model import SquareLattice
from smatrix.tau import tau_matrix_element
from scattering.filters import GH_filter_vectorized


def _vegas_transformed_integrand(E,Q,rx_normalized, ry_normalized, Delta_normalized,G:np.ndarray,H:np.ndarray,lattice:SquareLattice,sigma_func_period,tEQ):
    """transform the integral domain to be [-1,1]x[-1,1]x[0,1]"""
    rx = rx_normalized * lattice.q / 2
    ry = ry_normalized * lattice.q / 2
    r = np.concatenate([rx, ry])
    Delta = np.linalg.norm(E - np.linalg.norm(r + G) - np.linalg.norm(BZ_proj(r - H, lattice))) * Delta_normalized

    return _eigen_eq_integrand(E, Q, r, Delta,G, H, lattice, sigma_func_period,tEQ)


def eigen_eq_itr():
    """integrate the eigenvalue equation using Vegas"""
    prefactor = tau_matrix_element(E, Q, lattice, sigma_func_period)
    G,H = GH_filter_vectorized(Q, E, lattice)