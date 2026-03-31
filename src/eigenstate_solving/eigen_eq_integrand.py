import numpy as np

from model.model import SquareLattice
from smatrix import sw_propagator
from smatrix.amplitudes import legs, t, t_reg

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


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


def _make_eigen_eq_integrand_OLD(
    E: float,
    Q: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    lattice: SquareLattice,
    sigma_func_period,
    tEQ: float,
):

    def _integrand(x):
        rx_normalized, ry_normalized, D_normalized = x
        rx = rx_normalized * lattice.q / 2
        ry = ry_normalized * lattice.q / 2
        r = np.array([rx, ry])

        rG = r + G
        sH = BZ_proj(Q - r, lattice) + H

        width = E - np.linalg.norm(rG) - np.linalg.norm(sH)
        if width < 0:
            return 0.0 + 0.0j
        D = np.linalg.norm(rG) - E / 2 + width * D_normalized
        constant_factor = 1j / 2 / lattice.a**4
        E1 = E / 2 + D  # energy of the first photon
        E2 = E / 2 - D  # energy of the second photon
        transmission_factors = 1 / (
            t(rG, E1, lattice, sigma_func_period)
            * t(sH, E2, lattice, sigma_func_period)
            - tEQ
        )
        if E1**2 - np.linalg.norm(rG) ** 2 < 0 or E2**2 - np.linalg.norm(sH) ** 2 < 0:
            print(E1, E2, G, H, r, BZ_proj(Q - r, lattice), D, rG, sH)
        rz = np.sqrt(E1**2 - np.linalg.norm(rG) ** 2)
        sz = np.sqrt(E2**2 - np.linalg.norm(sH) ** 2)

        leg_factors = legs(
            rG, E1, sH, E2, lattice, sigma_func_period, direction="in"
        ) * legs(rG, E1, sH, E2, lattice, sigma_func_period, direction="out")

        E1_k1z_jacobian = E1 / rz
        E2_k2z_jacobian = E2 / sz
        # since we are integrating over [-1,1]x[-1,1]x[0,1], width is the jacobian for delta and (lattice.q/2)**2 is the jacobian for r_x and r_y.
        return (
            width
            * (lattice.q / 2) ** 2
            * E1_k1z_jacobian
            * E2_k2z_jacobian
            * constant_factor
            * transmission_factors
            * leg_factors
        )

    return _integrand


def _make_eigen_eq_integrand(
    E: float,
    Q: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    lattice: SquareLattice,
    sigma_func_period,
    tEQ: complex,
):

    def _integrand(x):
        rx_normalized, ry_normalized, D_normalized = x
        rx = rx_normalized * lattice.q / 2
        ry = ry_normalized * lattice.q / 2
        r = np.array([rx, ry])

        rG = r + G
        sH = BZ_proj(Q - r, lattice) + H

        width = E - np.linalg.norm(rG) - np.linalg.norm(sH)
        if width < 0:
            return 0.0 + 0.0j
        D = np.linalg.norm(rG) - E / 2 + width * D_normalized
        constant_factor = 2j
        E1 = E / 2 + D  # energy of the first photon
        E2 = E / 2 - D  # energy of the second photon
        transmission_factors = 1 / (
            t_reg(rG, E1, lattice, sigma_func_period)
            * t_reg(sH, E2, lattice, sigma_func_period)
            - tEQ
        )
        if E1**2 - np.linalg.norm(rG) ** 2 < 0 or E2**2 - np.linalg.norm(sH) ** 2 < 0:
            print(E1, E2, G, H, r, BZ_proj(Q - r, lattice), D, rG, sH)
        gamma1 = np.imag(
            sigma_func_period(rG[0], rG[1])
        )  # Here, (-2) is moved into the constant prefactor
        gamma2 = np.imag(sigma_func_period(sH[0], sH[1]))
        fraction1 = gamma1 * sw_propagator(rG, E1, lattice, sigma_func_period) ** 2
        fraction2 = gamma2 * sw_propagator(sH, E2, lattice, sigma_func_period) ** 2

        # since we are integrating over [-1,1]x[-1,1]x[0,1], width is the jacobian for delta and (lattice.q/2)**2 is the jacobian for r_x and r_y.
        return (
            width
            * (lattice.q / 2) ** 2
            * constant_factor
            * transmission_factors
            * fraction1
            * fraction2
        )

    return _integrand


def _build_eigen_eq_geometry_kernel(
    E: float, Q: np.ndarray, G: np.ndarray, H: np.ndarray, q: float
):
    """Build a Numba kernel for geometry terms in the current integrand."""
    if njit is None:
        return None

    E_val = float(E)
    q_val = float(q)
    half_q = q_val / 2.0
    Qx, Qy = float(Q[0]), float(Q[1])
    Gx, Gy = float(G[0]), float(G[1])
    Hx, Hy = float(H[0]), float(H[1])

    @njit(cache=True)
    def _kernel(xbatch):
        n = xbatch.shape[0]
        width = np.zeros(n, dtype=np.float64)
        E1 = np.zeros(n, dtype=np.float64)
        E2 = np.zeros(n, dtype=np.float64)
        rGx = np.zeros(n, dtype=np.float64)
        rGy = np.zeros(n, dtype=np.float64)
        sHx = np.zeros(n, dtype=np.float64)
        sHy = np.zeros(n, dtype=np.float64)
        valid = np.zeros(n, dtype=np.bool_)

        for idx in range(n):
            rx = xbatch[idx, 0] * half_q
            ry = xbatch[idx, 1] * half_q
            d_norm = xbatch[idx, 2]

            rgx = rx + Gx
            rgy = ry + Gy
            r_norm = np.sqrt(rgx * rgx + rgy * rgy)

            proj_x = ((Qx - rx + half_q) % q_val) - half_q
            proj_y = ((Qy - ry + half_q) % q_val) - half_q
            shx = proj_x + Hx
            shy = proj_y + Hy
            s_norm = np.sqrt(shx * shx + shy * shy)

            width_val = E_val - r_norm - s_norm
            if width_val < 0.0:
                continue

            D = r_norm - E_val / 2.0 + width_val * d_norm
            E1_val = E_val / 2.0 + D
            E2_val = E_val / 2.0 - D

            width[idx] = width_val
            E1[idx] = E1_val
            E2[idx] = E2_val
            rGx[idx] = rgx
            rGy[idx] = rgy
            sHx[idx] = shx
            sHy[idx] = shy
            valid[idx] = True

        return width, E1, E2, rGx, rGy, sHx, sHy, valid

    try:
        _ = _kernel(np.zeros((1, 3), dtype=np.float64))
    except Exception:
        return None

    return _kernel


def _make_eigen_eq_integrand_numba(
    E: float,
    Q: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    lattice: SquareLattice,
    sigma_func_period,
    tEQ: complex,
):
    """Current integrand with optional Numba-accelerated geometry and batch support."""
    base_integrand = _make_eigen_eq_integrand(
        E, Q, G, H, lattice, sigma_func_period, tEQ
    )
    kernel = _build_eigen_eq_geometry_kernel(E, Q, G, H, lattice.q)
    jacobian_xy = (lattice.q / 2) ** 2
    constant_factor = 2j

    def _value_from_geometry(width, E1, E2, rGx, rGy, sHx, sHy):
        rG = np.array([rGx, rGy], dtype=np.float64)
        sH = np.array([sHx, sHy], dtype=np.float64)
        transmission_factors = 1 / (
            t_reg(rG, E1, lattice, sigma_func_period)
            * t_reg(sH, E2, lattice, sigma_func_period)
            - tEQ
        )
        gamma1 = np.imag(sigma_func_period(rG[0], rG[1]))
        gamma2 = np.imag(sigma_func_period(sH[0], sH[1]))
        fraction1 = gamma1 * sw_propagator(rG, E1, lattice, sigma_func_period) ** 2
        fraction2 = gamma2 * sw_propagator(sH, E2, lattice, sigma_func_period) ** 2
        return (
            width
            * jacobian_xy
            * constant_factor
            * transmission_factors
            * fraction1
            * fraction2
        )

    def _integrand(x):
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            if kernel is None:
                return base_integrand(x_arr)

            packed = x_arr.reshape(1, 3)
            width, E1, E2, rGx, rGy, sHx, sHy, valid = kernel(packed)
            if not valid[0]:
                return 0.0 + 0.0j
            return _value_from_geometry(
                width[0], E1[0], E2[0], rGx[0], rGy[0], sHx[0], sHy[0]
            )

        if x_arr.ndim != 2 or x_arr.shape[1] != 3:
            raise ValueError("Input must be shape (3,) or (n, 3)")

        if kernel is None:
            out = np.zeros(x_arr.shape[0], dtype=np.complex128)
            for idx in range(x_arr.shape[0]):
                out[idx] = base_integrand(x_arr[idx])
            return out

        width, E1, E2, rGx, rGy, sHx, sHy, valid = kernel(x_arr)
        out = np.zeros(x_arr.shape[0], dtype=np.complex128)
        valid_idx = np.where(valid)[0]
        for idx in valid_idx:
            out[idx] = _value_from_geometry(
                width[idx], E1[idx], E2[idx], rGx[idx], rGy[idx], sHx[idx], sHy[idx]
            )
        return out

    return _integrand


def _build_eigen_eq_geometry_kernel_OLD(
    E: float, Q: np.ndarray, G: np.ndarray, H: np.ndarray, q: float
):
    """Build a Numba kernel for the geometry-heavy part of the integrand."""
    if njit is None:
        return None

    E_val = float(E)
    q_val = float(q)
    half_q = q_val / 2.0
    Qx, Qy = float(Q[0]), float(Q[1])
    Gx, Gy = float(G[0]), float(G[1])
    Hx, Hy = float(H[0]), float(H[1])

    @njit(cache=True)
    def _kernel(xbatch):
        n = xbatch.shape[0]
        width = np.zeros(n, dtype=np.float64)
        E1 = np.zeros(n, dtype=np.float64)
        E2 = np.zeros(n, dtype=np.float64)
        rGx = np.zeros(n, dtype=np.float64)
        rGy = np.zeros(n, dtype=np.float64)
        sHx = np.zeros(n, dtype=np.float64)
        sHy = np.zeros(n, dtype=np.float64)
        rz = np.zeros(n, dtype=np.float64)
        sz = np.zeros(n, dtype=np.float64)
        valid = np.zeros(n, dtype=np.bool_)

        for idx in range(n):
            rx = xbatch[idx, 0] * half_q
            ry = xbatch[idx, 1] * half_q
            d_norm = xbatch[idx, 2]

            rgx = rx + Gx
            rgy = ry + Gy
            r_norm = np.sqrt(rgx * rgx + rgy * rgy)

            proj_x = ((Qx - rx + half_q) % q_val) - half_q
            proj_y = ((Qy - ry + half_q) % q_val) - half_q
            shx = proj_x + Hx
            shy = proj_y + Hy
            s_norm = np.sqrt(shx * shx + shy * shy)

            width_val = E_val - r_norm - s_norm
            if width_val <= 0.0:
                continue

            D = r_norm - E_val / 2.0 + width_val * d_norm
            E1_val = E_val / 2.0 + D
            E2_val = E_val / 2.0 - D

            rz_sq = E1_val * E1_val - r_norm * r_norm
            sz_sq = E2_val * E2_val - s_norm * s_norm
            if rz_sq <= 0.0 or sz_sq <= 0.0:
                continue

            width[idx] = width_val
            E1[idx] = E1_val
            E2[idx] = E2_val
            rGx[idx] = rgx
            rGy[idx] = rgy
            sHx[idx] = shx
            sHy[idx] = shy
            rz[idx] = np.sqrt(rz_sq)
            sz[idx] = np.sqrt(sz_sq)
            valid[idx] = True

        return width, E1, E2, rGx, rGy, sHx, sHy, rz, sz, valid

    try:
        _ = _kernel(np.zeros((1, 3), dtype=np.float64))
    except Exception:
        return None

    return _kernel


def _make_eigen_eq_integrand_numba_OLD(
    E: float,
    Q: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    lattice: SquareLattice,
    sigma_func_period,
    tEQ: float,
):
    """Like _make_eigen_eq_integrand, with optional Numba-accelerated geometry and batch support."""
    base_integrand = _make_eigen_eq_integrand_OLD(
        E, Q, G, H, lattice, sigma_func_period, tEQ
    )
    kernel = _build_eigen_eq_geometry_kernel_OLD(E, Q, G, H, lattice.q)
    constant_factor = 1j / 2 / lattice.a**4

    def _integrand(x):
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            if kernel is None:
                return base_integrand(x_arr)

            packed = x_arr.reshape(1, 3)
            width, E1, E2, rGx, rGy, sHx, sHy, rz, sz, valid = kernel(packed)
            if not valid[0]:
                return 0.0 + 0.0j
            rG = np.array([rGx[0], rGy[0]])
            sH = np.array([sHx[0], sHy[0]])
            transmission_factors = 1 / (
                t(rG, E1[0], lattice, sigma_func_period)
                * t(sH, E2[0], lattice, sigma_func_period)
                - tEQ
            )
            leg_factors = legs(
                rG, E1[0], sH, E2[0], lattice, sigma_func_period, direction="in"
            ) * legs(rG, E1[0], sH, E2[0], lattice, sigma_func_period, direction="out")
            return (
                width[0]
                * (lattice.q / 2) ** 2
                * (E1[0] / rz[0])
                * (E2[0] / sz[0])
                * constant_factor
                * transmission_factors
                * leg_factors
            )

        if x_arr.ndim != 2 or x_arr.shape[1] != 3:
            raise ValueError("Input must be shape (3,) or (n, 3)")

        if kernel is None:
            out = np.zeros(x_arr.shape[0], dtype=np.complex128)
            for idx in range(x_arr.shape[0]):
                out[idx] = base_integrand(x_arr[idx])
            return out

        width, E1, E2, rGx, rGy, sHx, sHy, rz, sz, valid = kernel(x_arr)
        out = np.zeros(x_arr.shape[0], dtype=np.complex128)
        if np.any(valid):
            rG_valid = np.column_stack((rGx[valid], rGy[valid]))
            sH_valid = np.column_stack((sHx[valid], sHy[valid]))
            E1_valid = E1[valid]
            E2_valid = E2[valid]

            transmission_factors = 1 / (
                t(rG_valid, E1_valid, lattice, sigma_func_period)
                * t(sH_valid, E2_valid, lattice, sigma_func_period)
                - tEQ
            )
            leg_factors = legs(
                rG_valid,
                E1_valid,
                sH_valid,
                E2_valid,
                lattice,
                sigma_func_period,
                direction="in",
            ) * legs(
                rG_valid,
                E1_valid,
                sH_valid,
                E2_valid,
                lattice,
                sigma_func_period,
                direction="out",
            )
            out[valid] = (
                width[valid]
                * (lattice.q / 2) ** 2
                * (E1_valid / rz[valid])
                * (E2_valid / sz[valid])
                * constant_factor
                * transmission_factors
                * leg_factors
            )
        return out

    return _integrand


__all__ = ["BZ_proj", "_make_eigen_eq_integrand_numba", "_make_eigen_eq_integrand_OLD"]
