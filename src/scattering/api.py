"""Public API for connected/disconnected scattering integrals."""

from __future__ import annotations

import numpy as np

from smatrix import S_disconnected

from .filters import J_filter
from .integrand import _make_integrand_and_bounds
from .backends import _integrate_nquad, _integrate_qmc, _integrate_vegas


def disconnected_scattering_integral(
    q_para, Eq, l_para, El, in_state, lattice, sigma_func_period
):
    return S_disconnected(
        q_para, Eq, l_para, El, lattice, sigma_func_period
    ) * in_state(q_para, Eq, l_para, El)


def scattering_integral_nquad(
    k_para, Ek, p_para, Ep, lattice, in_state, sigma_func_period
):
    """Compute scattering integral using quadrature (nquad)."""
    E = Ek + Ep
    bound = np.pi / lattice.a
    J_x, J_y = J_filter(k_para, p_para, lattice)
    integrand, D_bounds = _make_integrand_and_bounds(
        E, lattice, in_state, sigma_func_period
    )
    return _integrate_nquad(
        J_x, J_y, k_para, p_para, E, bound, lattice, integrand, D_bounds
    )


def scattering_integral_qmc(
    k_para, Ek, p_para, Ep, lattice, in_state, sigma_func_period, m=13, seed=None
):
    """Compute scattering integral using Quasi-Monte Carlo (Sobol sequence)."""
    E = Ek + Ep
    bound = np.pi / lattice.a
    J_x, J_y = J_filter(k_para, p_para, lattice)
    integrand, D_bounds = _make_integrand_and_bounds(
        E, lattice, in_state, sigma_func_period
    )
    return _integrate_qmc(
        J_x, J_y, k_para, p_para, E, bound, lattice, integrand, D_bounds, m, seed
    )


def scattering_integral_vegas(
    k_para,
    Ek,
    p_para,
    Ep,
    lattice,
    in_state,
    sigma_func_period,
    nitn1=3,
    nitn2=10,
    neval=5e4,
):
    """Compute scattering integral using Vegas adaptive Monte Carlo."""
    E = Ek + Ep
    bound = np.pi / lattice.a
    J_x, J_y = J_filter(k_para, p_para, lattice)
    integrand, D_bounds = _make_integrand_and_bounds(
        E, lattice, in_state, sigma_func_period
    )
    return _integrate_vegas(
        J_x,
        J_y,
        k_para,
        p_para,
        E,
        bound,
        lattice,
        integrand,
        D_bounds,
        nitn1,
        nitn2,
        neval,
    )


# Convenience alias (older scripts used `scattering_integral` for the nquad backend).
scattering_integral = scattering_integral_nquad


__all__ = [
    "disconnected_scattering_integral",
    "scattering_integral",
    "scattering_integral_nquad",
    "scattering_integral_qmc",
    "scattering_integral_vegas",
]
