"""Integrand construction for connected scattering integrals."""

from __future__ import annotations

import numpy as np

from input_states import gaussian_in_state
from model import c, epsilon_0
from smatrix import legs

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback when numba is unavailable
    njit = None


def _build_numba_integrand_kernel(E, lattice, in_state, sigma_func_period):
    """
    Try to build a Numba kernel for the hot integrand path.

    If the inputs are not compatible with nopython mode, returns None and the
    caller falls back to a Python/NumPy implementation.
    """
    if njit is None:
        return None
    if not isinstance(in_state, gaussian_in_state):
        return None

    try:
        q0 = np.asarray(in_state.q0, dtype=np.float64)
        l0 = np.asarray(in_state.l0, dtype=np.float64)
        sigma_arr = np.asarray(in_state.sigma, dtype=np.float64)
        if q0.shape != (3,) or l0.shape != (3,):
            return None
        if sigma_arr.ndim == 0:
            sigma_vec = np.full(3, float(sigma_arr), dtype=np.float64)
        else:
            sigma_flat = sigma_arr.reshape(-1)
            if sigma_flat.size != 3:
                return None
            sigma_vec = sigma_flat.astype(np.float64)
        if np.any(sigma_vec <= 0):
            return None

        d_vec = np.asarray(lattice.d, dtype=np.complex128).reshape(3)
        omega_e = float(lattice.omega_e)
        c_val = float(c)
        eps0 = float(epsilon_0)
        e_half = 0.5 * float(E)
        e_total = float(E)
        sqrt2_inv = 1.0 / np.sqrt(2.0)
        norm_pref = (2.0 * np.pi) ** (-1.5) * (np.prod(sigma_vec) ** (-1.0))
    except Exception:
        return None

    @njit(cache=True)
    def _ge_single(kx, ky, kz, d):
        kxy2 = kx * kx + ky * ky
        kxy_norm = np.sqrt(kxy2)
        if kxy_norm < 1e-14:
            return 0.0

        total = 0.0
        for v in (-1.0, 1.0):
            kz_signed = v * kz
            k_norm = np.sqrt(kxy2 + kz_signed * kz_signed)
            if k_norm < 1e-14:
                continue

            e1x = ky / kxy_norm
            e1y = -kx / kxy_norm
            e1z = 0.0

            denom = k_norm * kxy_norm
            e2x = -(-kz_signed * kx) / denom
            e2y = -(-kz_signed * ky) / denom
            e2z = -(kxy2) / denom

            p0x = sqrt2_inv * (e1x + 1j * e2x)
            p0y = sqrt2_inv * (e1y + 1j * e2y)
            p0z = sqrt2_inv * (e1z + 1j * e2z)
            coup0 = np.conjugate(p0x) * d[0] + np.conjugate(p0y) * d[1] + np.conjugate(p0z) * d[2]

            p1x = sqrt2_inv * (e1x - 1j * e2x)
            p1y = sqrt2_inv * (e1y - 1j * e2y)
            p1z = sqrt2_inv * (e1z - 1j * e2z)
            coup1 = np.conjugate(p1x) * d[0] + np.conjugate(p1y) * d[1] + np.conjugate(p1z) * d[2]

            disp = c_val * k_norm
            pref = np.sqrt(disp / (2.0 * eps0))
            g0 = pref * coup0
            g1 = pref * coup1
            total += (g0.real * g0.real + g0.imag * g0.imag)
            total += (g1.real * g1.real + g1.imag * g1.imag)

        return np.sqrt(total)

    @njit(cache=True)
    def _gaussian_state_value(qx, qy, Eq, lx, ly, El):
        qz2 = (Eq / c_val) * (Eq / c_val) - (qx * qx + qy * qy)
        lz2 = (El / c_val) * (El / c_val) - (lx * lx + ly * ly)
        if qz2 <= 0.0 or lz2 <= 0.0:
            return 0.0

        qz = np.sqrt(qz2)
        lz = np.sqrt(lz2)

        dq0 = (qx - q0[0]) / sigma_vec[0]
        dq1 = (qy - q0[1]) / sigma_vec[1]
        dq2 = (qz - q0[2]) / sigma_vec[2]
        dl0 = (lx - l0[0]) / sigma_vec[0]
        dl1 = (ly - l0[1]) / sigma_vec[1]
        dl2 = (lz - l0[2]) / sigma_vec[2]

        exp_arg = -0.25 * (dq0 * dq0 + dq1 * dq1 + dq2 * dq2 + dl0 * dl0 + dl1 * dl1 + dl2 * dl2)
        return norm_pref * np.exp(exp_arg)

    @njit(cache=True)
    def _integrand_kernel(D, Dpx, Dpy, COM_K, G, H):
        n = D.shape[0]
        out = np.zeros(n, dtype=np.complex128)

        for idx in range(n):
            dpx = Dpx[idx]
            dpy = Dpy[idx]
            d = D[idx]

            qx = COM_K[0] + 0.5 * dpx + G[0]
            qy = COM_K[1] + 0.5 * dpy + G[1]
            lx = COM_K[0] - 0.5 * dpx + H[0]
            ly = COM_K[1] - 0.5 * dpy + H[1]

            Eq = e_half + d
            El = e_half - d

            indicator_arg = (
                e_total
                - np.sqrt((qx + G[0]) * (qx + G[0]) + (qy + G[1]) * (qy + G[1]))
                - np.sqrt((lx + H[0]) * (lx + H[0]) + (ly + H[1]) * (ly + H[1]))
            )
            if indicator_arg < 0.0:
                continue

            q_norm2 = qx * qx + qy * qy
            l_norm2 = lx * lx + ly * ly

            qz2_jac = Eq * Eq - q_norm2
            lz2_jac = El * El - l_norm2
            if qz2_jac <= 0.0 or lz2_jac <= 0.0:
                continue

            qz2 = (Eq / c_val) * (Eq / c_val) - q_norm2
            lz2 = (El / c_val) * (El / c_val) - l_norm2
            if qz2 <= 0.0 or lz2 <= 0.0:
                continue

            qz = np.sqrt(qz2)
            lz = np.sqrt(lz2)

            ge_q = _ge_single(qx, qy, qz, d_vec)
            ge_l = _ge_single(lx, ly, lz, d_vec)
            if ge_q == 0.0 or ge_l == 0.0:
                continue

            sigma_q = sigma_func_period(qx, qy)
            sigma_l = sigma_func_period(lx, ly)
            sw_q = 1.0 / (Eq - omega_e - sigma_q)
            sw_l = 1.0 / (El - omega_e - sigma_l)
            legs_val = ge_q * ge_l * sw_q * sw_l

            in_state_val = _gaussian_state_value(qx, qy, Eq, lx, ly, El)
            if in_state_val == 0.0:
                continue

            jacobian = (Eq / np.sqrt(qz2_jac)) * (El / np.sqrt(lz2_jac))
            out[idx] = jacobian * legs_val * in_state_val

        return out

    try:
        _ = _integrand_kernel(
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        )
    except Exception:
        return None

    return _integrand_kernel


def _make_integrand_and_bounds(E, lattice, in_state, sigma_func_period):
    """Factory function to create integrand and D_bounds functions."""
    integrand_kernel_numba = _build_numba_integrand_kernel(E, lattice, in_state, sigma_func_period)

    def integrand(D, Dpx, Dpy, COM_K, G, H):
        """Integrand with D as innermost variable so its bounds can depend on Dpx, Dpy.

        Supports scalar inputs or batched inputs (arrays) for D, Dpx, Dpy.
        """
        if integrand_kernel_numba is not None:
            is_scalar = np.ndim(D) == 0 and np.ndim(Dpx) == 0 and np.ndim(Dpy) == 0
            D_arr = np.atleast_1d(np.asarray(D, dtype=np.float64))
            Dpx_arr = np.atleast_1d(np.asarray(Dpx, dtype=np.float64))
            Dpy_arr = np.atleast_1d(np.asarray(Dpy, dtype=np.float64))
            COM_K_arr = np.asarray(COM_K, dtype=np.float64)
            G_arr = np.asarray(G, dtype=np.float64)
            H_arr = np.asarray(H, dtype=np.float64)
            out = integrand_kernel_numba(D_arr, Dpx_arr, Dpy_arr, COM_K_arr, G_arr, H_arr)
            if is_scalar:
                return out[0]
            return out

        Dpx_arr = np.asarray(Dpx)
        Dpy_arr = np.asarray(Dpy)
        D_arr = np.asarray(D)

        Dp = np.stack([Dpx_arr, Dpy_arr], axis=-1)
        q_para = COM_K + Dp / 2 + G
        l_para = COM_K - Dp / 2 + H

        Eq = E / 2 + D_arr
        El = E / 2 - D_arr

        q_norm = np.linalg.norm(q_para, axis=-1)
        l_norm = np.linalg.norm(l_para, axis=-1)
        indicator = np.heaviside(
            E - np.linalg.norm(q_para + G, axis=-1) - np.linalg.norm(l_para + H, axis=-1),
            0.5,
        )

        mask = indicator != 0

        if np.ndim(mask) == 0:
            if not mask:
                return 0.0 + 0.0j

            qz = np.sqrt(Eq**2 - q_norm**2)
            lz = np.sqrt(El**2 - l_norm**2)
            jacobian = (Eq / qz) * (El / lz)
            return (
                jacobian
                * legs(q_para, Eq, l_para, El, lattice, sigma_func_period, direction="in")
                * in_state(q_para, Eq, l_para, El)
            )

        value = np.zeros_like(indicator, dtype=np.complex128)
        if np.any(mask):
            Eq_m = Eq[mask]
            El_m = El[mask]
            q_norm_m = q_norm[mask]
            l_norm_m = l_norm[mask]
            q_para_m = q_para[mask]
            l_para_m = l_para[mask]

            qz = np.sqrt(Eq_m**2 - q_norm_m**2)
            lz = np.sqrt(El_m**2 - l_norm_m**2)
            jacobian = (Eq_m / qz) * (El_m / lz)
            value[mask] = (
                jacobian
                * legs(q_para_m, Eq_m, l_para_m, El_m, lattice, sigma_func_period, direction="in")
                * in_state(q_para_m, Eq_m, l_para_m, El_m)
            )

        return value

    def D_bounds(Dpx, Dpy, COM_K, G, H):
        """D bounds depend on Dpx, Dpy through q_para, l_para.

        Works with both scalar and array inputs.
        """
        Dpx_arr = np.atleast_1d(Dpx)
        Dpy_arr = np.atleast_1d(Dpy)
        is_scalar = np.ndim(Dpx) == 0 and np.ndim(Dpy) == 0

        Dp = np.stack([Dpx_arr, Dpy_arr], axis=-1)
        q_para = COM_K + Dp / 2 + G
        l_para = COM_K - Dp / 2 + H

        D_min = np.linalg.norm(q_para, axis=-1) - E / 2
        D_max = E / 2 - np.linalg.norm(l_para, axis=-1)

        if is_scalar:
            return [D_min.item(), D_max.item()]
        return [D_min, D_max]

    return integrand, D_bounds


__all__ = ["_build_numba_integrand_kernel", "_make_integrand_and_bounds"]

