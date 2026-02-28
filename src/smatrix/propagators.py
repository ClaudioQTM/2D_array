"""Propagators used by S-matrix building blocks."""

from __future__ import annotations

import numpy as np


def sw_propagator(k_para, E, lattice, sigma_func_period):
    k_arr = np.asarray(k_para, dtype=np.float64)
    E_arr = np.asarray(E, dtype=np.float64)

    if k_arr.ndim == 1:
        sigma_val = sigma_func_period(k_arr[0], k_arr[1])
        denom = E_arr - lattice.omega_e - sigma_val
        return 1 / denom

    if k_arr.ndim == 2:
        if k_arr.shape[1] == 2:
            k_para_norm = k_arr
        elif k_arr.shape[0] == 2:
            k_para_norm = k_arr.T
        else:
            raise ValueError("k_para must have shape (n, 2) or (2, n).")

        if E_arr.ndim == 0:
            E_vec = np.full(k_para_norm.shape[0], float(E_arr))
        else:
            E_vec = E_arr.reshape(-1)
            if E_vec.shape[0] != k_para_norm.shape[0]:
                raise ValueError("E must have the same length as k_para.")

        sigma_vals = np.array(
            [sigma_func_period(kx, ky) for kx, ky in k_para_norm],
            dtype=np.complex128,
        )
        denom = E_vec - lattice.omega_e - sigma_vals
        return 1 / denom

    raise ValueError("k_para must be a 1D or 2D array.")


__all__ = ["sw_propagator"]
