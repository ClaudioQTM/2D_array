"""Numerical backends for scattering integrals (nquad, QMC, Vegas)."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.integrate as integrate

from .filters import GH_filter
from .vegas_transform import _build_vegas_transform_kernel


def _integrate_nquad(J_x, J_y, k_para, p_para, E, bound, lattice, integrand, D_bounds):
    total = 0.0 + 0.0j

    for j in range(len(J_x)):
        J = np.array([J_x[j], J_y[j]])
        COM_K = 0.5 * (k_para + p_para + J)
        G_x, G_y, H_x, H_y = GH_filter(COM_K, E, lattice)

        Dpx_bounds = [abs(COM_K[0]) - bound, bound - abs(COM_K[0])]
        Dpy_bounds = [abs(COM_K[1]) - bound, bound - abs(COM_K[1])]

        for i in range(len(G_x)):
            G = np.array([G_x[i], G_y[i]])
            H = np.array([H_x[i], H_y[i]])

            result_real, _ = integrate.nquad(
                lambda D, Dpx, Dpy: integrand(D, Dpx, Dpy, COM_K, G, H).real,
                [
                    lambda Dpx, Dpy: D_bounds(Dpx, Dpy, COM_K, G, H),
                    Dpx_bounds,
                    Dpy_bounds,
                ],
                opts={"epsabs": 1e-3, "epsrel": 1e-3, "limit": 30},
            )

            result_imag, _ = integrate.nquad(
                lambda D, Dpx, Dpy: integrand(D, Dpx, Dpy, COM_K, G, H).imag,
                [
                    lambda Dpx, Dpy: D_bounds(Dpx, Dpy, COM_K, G, H),
                    Dpx_bounds,
                    Dpy_bounds,
                ],
                opts={"epsabs": 1e-3, "epsrel": 1e-3, "limit": 30},
            )

            total += result_real + 1j * result_imag

    return total


def _integrate_qmc(J_x, J_y, k_para, p_para, E, bound, lattice, integrand, D_bounds, m=13, seed=None):
    from scipy.stats import qmc

    n_samples = 2**m
    total = 0.0 + 0.0j

    for j in range(len(J_x)):
        J = np.array([J_x[j], J_y[j]])
        COM_K = 0.5 * (k_para + p_para + J)
        G_x, G_y, H_x, H_y = GH_filter(COM_K, E, lattice)

        Dpx_lo, Dpx_hi = abs(COM_K[0]) - bound, bound - abs(COM_K[0])
        Dpy_lo, Dpy_hi = abs(COM_K[1]) - bound, bound - abs(COM_K[1])

        for i in range(len(G_x)):
            G = np.array([G_x[i], G_y[i]])
            H = np.array([H_x[i], H_y[i]])

            sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
            samples = sampler.random_base2(m)

            u1, u2, u3 = samples[:, 0], samples[:, 1], samples[:, 2]
            Dpx_arr = Dpx_lo + u1 * (Dpx_hi - Dpx_lo)
            Dpy_arr = Dpy_lo + u2 * (Dpy_hi - Dpy_lo)

            Dp_arr = np.stack([Dpx_arr, Dpy_arr], axis=1)
            q_para_arr = COM_K + Dp_arr / 2 + G
            l_para_arr = COM_K - Dp_arr / 2 + H
            D_lo_arr = np.linalg.norm(q_para_arr, axis=1) - E / 2
            D_hi_arr = E / 2 - np.linalg.norm(l_para_arr, axis=1)

            valid_mask = D_hi_arr > D_lo_arr
            if not np.any(valid_mask):
                continue

            D_arr = D_lo_arr[valid_mask] + u3[valid_mask] * (D_hi_arr[valid_mask] - D_lo_arr[valid_mask])
            Dpx_valid = Dpx_arr[valid_mask]
            Dpy_valid = Dpy_arr[valid_mask]

            jacobian_arr = (Dpx_hi - Dpx_lo) * (Dpy_hi - Dpy_lo) * (D_hi_arr[valid_mask] - D_lo_arr[valid_mask])

            integral_sum = 0.0 + 0.0j
            for idx in range(len(D_arr)):
                f_val = integrand(D_arr[idx], Dpx_valid[idx], Dpy_valid[idx], COM_K, G, H)
                integral_sum += f_val * jacobian_arr[idx]

            total += integral_sum / n_samples

    return total


def _integrate_vegas(
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
):
    import vegas

    total = 0.0 + 0.0j
    vegas_transform_kernel = _build_vegas_transform_kernel(E)

    for j in range(len(J_x)):
        J = np.array([J_x[j], J_y[j]])
        COM_K = 0.5 * (k_para + p_para + J)
        G_x, G_y, H_x, H_y = GH_filter(COM_K, E, lattice)

        if len(G_x) == 0:
            continue

        Dpx_lo, Dpx_hi = abs(COM_K[0]) - bound, bound - abs(COM_K[0])
        Dpy_lo, Dpy_hi = abs(COM_K[1]) - bound, bound - abs(COM_K[1])

        for i in range(len(G_x)):
            G = np.array([G_x[i], G_y[i]])
            H = np.array([H_x[i], H_y[i]])
            COM_K64 = np.asarray(COM_K, dtype=np.float64)
            G64 = np.asarray(G, dtype=np.float64)
            H64 = np.asarray(H, dtype=np.float64)

            @vegas.lbatchintegrand
            def vegas_integrand_real(xbatch):
                if vegas_transform_kernel is not None:
                    Dpx, Dpy, D, _, _, jacobian, valid = vegas_transform_kernel(
                        np.asarray(xbatch, dtype=np.float64),
                        Dpx_lo,
                        Dpx_hi,
                        Dpy_lo,
                        Dpy_hi,
                        COM_K64,
                        G64,
                        H64,
                    )
                else:
                    u1 = xbatch[:, 0]
                    u2 = xbatch[:, 1]
                    u3 = xbatch[:, 2]

                    Dpx = Dpx_lo + u1 * (Dpx_hi - Dpx_lo)
                    Dpy = Dpy_lo + u2 * (Dpy_hi - Dpy_lo)

                    D_lo, D_hi = D_bounds(Dpx, Dpy, COM_K, G, H)
                    valid = D_hi > D_lo
                    D = D_lo + u3 * (D_hi - D_lo)
                    jacobian = (Dpx_hi - Dpx_lo) * (Dpy_hi - Dpy_lo) * (D_hi - D_lo)

                f_val = integrand(D, Dpx, Dpy, COM_K, G, H)
                result = f_val.real * jacobian
                return np.where(valid, result, 0.0)

            @vegas.lbatchintegrand
            def vegas_integrand_imag(xbatch):
                if vegas_transform_kernel is not None:
                    Dpx, Dpy, D, _, _, jacobian, valid = vegas_transform_kernel(
                        np.asarray(xbatch, dtype=np.float64),
                        Dpx_lo,
                        Dpx_hi,
                        Dpy_lo,
                        Dpy_hi,
                        COM_K64,
                        G64,
                        H64,
                    )
                else:
                    u1 = xbatch[:, 0]
                    u2 = xbatch[:, 1]
                    u3 = xbatch[:, 2]

                    Dpx = Dpx_lo + u1 * (Dpx_hi - Dpx_lo)
                    Dpy = Dpy_lo + u2 * (Dpy_hi - Dpy_lo)

                    D_lo, D_hi = D_bounds(Dpx, Dpy, COM_K, G, H)
                    valid = D_hi > D_lo
                    D = D_lo + u3 * (D_hi - D_lo)
                    jacobian = (Dpx_hi - Dpx_lo) * (Dpy_hi - Dpy_lo) * (D_hi - D_lo)

                f_val = integrand(D, Dpx, Dpy, COM_K, G, H)
                result = f_val.imag * jacobian
                return np.where(valid, result, 0.0)

            vegas_integ_re = vegas.Integrator([[0, 1], [0, 1], [0, 1]])
            vegas_integ_re(vegas_integrand_real, nitn=nitn1, neval=neval)
            result_real = vegas_integ_re(vegas_integrand_real, nitn=nitn2, neval=neval)
            real_iters = nitn1 + nitn2
            while result_real.Q <= 0.1 and real_iters <= 20:
                result_real = vegas_integ_re(vegas_integrand_real, nitn=nitn2, neval=neval)
                real_iters += nitn2
            if result_real.Q <= 0.1:
                warnings.warn(
                    f"VEGAS real-part Q stayed <= 0.1 after {real_iters} iterations (Q={result_real.Q}).",
                    RuntimeWarning,
                )

            vegas_integ_im = vegas.Integrator([[0, 1], [0, 1], [0, 1]])
            vegas_integ_im(vegas_integrand_imag, nitn=nitn1, neval=neval)
            result_imag = vegas_integ_im(vegas_integrand_imag, nitn=nitn2, neval=neval)
            imag_iters = nitn1 + nitn2
            while result_imag.Q <= 0.1 and imag_iters <= 20:
                result_imag = vegas_integ_im(vegas_integrand_imag, nitn=nitn2, neval=neval)
                imag_iters += nitn2
            if result_imag.Q <= 0.1:
                warnings.warn(
                    f"VEGAS imag-part Q stayed <= 0.1 after {imag_iters} iterations (Q={result_imag.Q}).",
                    RuntimeWarning,
                )

            total += result_real.mean + 1j * result_imag.mean

    return total


__all__ = ["_integrate_nquad", "_integrate_qmc", "_integrate_vegas"]

