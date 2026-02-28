"""Scattering amplitudes and external-leg factors."""

from __future__ import annotations

import numpy as np

from model import self_energy

from .defaults import alpha
from .kinematics import coord_convert
from .propagators import sw_propagator
from .tau import tau_matrix_element


def t(k_para, E, lattice, sigma_func_period=None):
    """
    Single-photon transmission amplitude.

    If `sigma_func_period` is provided, it is used as the self-energy (fast path). Otherwise we call `model.self_energy`.
    """
    k = coord_convert(k_para, E)
    if k.ndim == 1:
        kz = k[2]
        kx, ky = float(k[0]), float(k[1])
        sigma_val = (
            sigma_func_period(kx, ky)
            if sigma_func_period is not None
            else self_energy(kx, ky, lattice.a, lattice.d, float(E), alpha)
        )
    else:
        kz = k[:, 2]
        kx = k[:, 0]
        ky = k[:, 1]
        if sigma_func_period is not None:
            sigma_val = np.array(
                [sigma_func_period(float(xx), float(yy)) for xx, yy in zip(kx, ky)],
                dtype=np.complex128,
            )
        else:
            sigma_val = np.array(
                [
                    self_energy(
                        float(xx), float(yy), lattice.a, lattice.d, float(ee), alpha
                    )
                    for xx, yy, ee in zip(
                        kx,
                        ky,
                        (
                            np.asarray(E).reshape(-1)
                            if np.ndim(E)
                            else np.full(kx.shape[0], float(E))
                        ),
                    )
                ],
                dtype=np.complex128,
            )

    prefactor = -1j / lattice.a**2 * np.asarray(E) / kz
    numerator = np.abs(lattice.ge(k)) ** 2
    denominator = np.asarray(E) - lattice.omega_e - sigma_val
    return 1 + prefactor * (numerator / denominator)


def S_disconnected(q_para, Eq, l_para, El, lattice, sigma_func_period=None):
    return t(q_para, Eq, lattice, sigma_func_period) * t(
        l_para, El, lattice, sigma_func_period
    )


def legs(q_para, Eq, l_para, El, lattice, sigma_func_period, direction="in"):
    q = coord_convert(q_para, Eq)
    l = coord_convert(l_para, El)  # noqa: E741
    if direction == "in":
        coupling = lattice.ge(q) * lattice.ge(l)
    elif direction == "out":
        coupling = np.conj(lattice.ge(q)) * np.conj(lattice.ge(l))
    else:
        raise ValueError(f"Invalid direction: {direction}")
    return (
        coupling
        * sw_propagator(q_para, Eq, lattice, sigma_func_period)
        * sw_propagator(l_para, El, lattice, sigma_func_period)
    )


def connected_amplitude(
    k_para,
    Ek,
    p_para,
    Ep,
    lattice,
    in_state,
    sigma_func_period=None,
    nitn1=3,
    nitn2=10,
    neval=5e4,
):
    from scattering.api import scattering_integral_vegas

    outgoing_legs = legs(
        k_para, Ek, p_para, Ep, lattice, sigma_func_period, direction="out"
    )
    Q = np.asarray(k_para + p_para)
    excitation_vertex = tau_matrix_element(Ek + Ep, Q, lattice, sigma_func_period)
    integral_term = scattering_integral_vegas(
        k_para,
        Ek,
        p_para,
        Ep,
        lattice,
        in_state,
        sigma_func_period,
        nitn1=nitn1,
        nitn2=nitn2,
        neval=neval,
    )
    return outgoing_legs * excitation_vertex * integral_term


__all__ = ["t", "S_disconnected", "legs", "connected_amplitude"]
