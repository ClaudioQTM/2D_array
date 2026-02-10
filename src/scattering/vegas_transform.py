"""Vegas variable transforms for scattering integrals."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def _build_vegas_transform_kernel(E):
    """
    Build a Numba kernel for the Vegas variable transform: [0,1]^3 -> (Dpx, Dpy, D).

    Returns a compiled function if Numba is available, otherwise returns None.
    """
    if njit is None:
        return None

    E_half = 0.5 * float(E)

    @njit(cache=True)
    def _transform(xbatch, Dpx_lo, Dpx_hi, Dpy_lo, Dpy_hi, COM_K, G, H):
        n = xbatch.shape[0]
        Dpx = np.empty(n, dtype=np.float64)
        Dpy = np.empty(n, dtype=np.float64)
        D = np.empty(n, dtype=np.float64)
        D_lo = np.empty(n, dtype=np.float64)
        D_hi = np.empty(n, dtype=np.float64)
        jacobian = np.empty(n, dtype=np.float64)
        valid = np.empty(n, dtype=np.bool_)

        vol_Dpx = Dpx_hi - Dpx_lo
        vol_Dpy = Dpy_hi - Dpy_lo
        prefactor = vol_Dpx * vol_Dpy

        for idx in range(n):
            u1 = xbatch[idx, 0]
            u2 = xbatch[idx, 1]
            u3 = xbatch[idx, 2]

            dpx = Dpx_lo + u1 * vol_Dpx
            dpy = Dpy_lo + u2 * vol_Dpy
            Dpx[idx] = dpx
            Dpy[idx] = dpy

            qx = COM_K[0] + 0.5 * dpx + G[0]
            qy = COM_K[1] + 0.5 * dpy + G[1]
            lx = COM_K[0] - 0.5 * dpx + H[0]
            ly = COM_K[1] - 0.5 * dpy + H[1]

            dlo = np.sqrt(qx * qx + qy * qy) - E_half
            dhi = E_half - np.sqrt(lx * lx + ly * ly)
            D_lo[idx] = dlo
            D_hi[idx] = dhi

            D[idx] = dlo + u3 * (dhi - dlo)

            jacobian[idx] = prefactor * (dhi - dlo)
            valid[idx] = dhi > dlo

        return Dpx, Dpy, D, D_lo, D_hi, jacobian, valid

    return _transform


__all__ = ["_build_vegas_transform_kernel"]

