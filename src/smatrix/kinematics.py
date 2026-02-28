"""Kinematics helpers (coordinate conversions, dispersion-related utilities)."""

from __future__ import annotations

import numpy as np

from model import c


def coord_convert(k_para, E):
    """
    Convert 2D parallel momentum [kx, ky] to 3D Cartesian [kx, ky, kz] using
    dispersion |k| = E/c, so kz = sqrt((E/c)^2 - kx^2 - ky^2).

    Accepts single vectors or batches.
    """
    k_arr = np.asarray(k_para, dtype=np.float64)
    c_val = float(c)

    if k_arr.ndim == 1:
        kz = np.sqrt(
            (np.asarray(E, dtype=np.float64) / c_val) ** 2 - np.linalg.norm(k_arr) ** 2
        )
        if np.ndim(kz) == 0:
            return np.concatenate([k_arr, [kz]])
        k_para_arr = np.broadcast_to(k_arr, (kz.shape[0], 2))
        return np.column_stack([k_para_arr, kz])

    if k_arr.ndim == 2:
        if k_arr.shape[1] == 2:
            k_para_norm = k_arr
        elif k_arr.shape[0] == 2:
            k_para_norm = k_arr.T
        else:
            raise ValueError("k_para must have shape (n, 2) or (2, n).")

        E_arr = np.asarray(E, dtype=np.float64)
        if E_arr.ndim == 0:
            E_vec = np.full(k_para_norm.shape[0], float(E_arr))
        else:
            E_vec = E_arr.reshape(-1)
            if E_vec.shape[0] != k_para_norm.shape[0]:
                raise ValueError("E must have the same length as k_para.")

        kz = np.sqrt((E_vec / c_val) ** 2 - np.sum(k_para_norm**2, axis=1))
        return np.column_stack([k_para_norm, kz])

    raise ValueError("k_para must be a 1D or 2D array.")


__all__ = ["coord_convert"]
