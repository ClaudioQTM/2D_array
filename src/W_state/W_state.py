from smatrix import t_reg
from eigenstate_solving import BZ_proj
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad_vec



def LHS_Kz(Kz, q, E, r_para, Q_para, lattice):
    """
    Solve for total momentum Kz from on-shell condition.
    F(Kz; q) = E1 + E2 - E.
    Uses np.hypot for numerical stability:
        hypot(a,b) = sqrt(a^2 + b^2).
    """
    abs_r_para = np.linalg.norm(r_para)
    abs_s_para = np.linalg.norm(BZ_proj(Q_para - r_para, lattice))

    k1z = 0.5 * Kz + q
    k2z = 0.5 * Kz - q

    E1 = np.hypot(abs_r_para, k1z)
    E2 = np.hypot(abs_s_para, k2z)

    return E1 + E2 - E


def q_bounds(E, r_para, Q_para, lattice):
    """
    Allowed q interval for propagating channels.
    Requires E >= m1 + m2.
    """
    abs_r_para = np.linalg.norm(r_para)
    abs_s_para = np.linalg.norm(BZ_proj(Q_para - r_para, lattice))
    if E < abs_r_para + abs_s_para:
        raise ValueError("No propagating two-photon channel: E < m1 + m2.")
    if (E - abs_r_para) ** 2 - abs_s_para**2 < 0 or (
        E - abs_s_para
    ) ** 2 - abs_r_para**2 < 0:
        raise ValueError("Negative value under square root in q_bounds.")
    q_min = -0.5 * np.sqrt((E - abs_r_para) ** 2 - abs_s_para**2)
    q_max = 0.5 * np.sqrt((E - abs_s_para) ** 2 - abs_r_para**2)

    return q_min, q_max


def solve_Kz(q, E, r_para, Q_para, lattice, xtol=1e-12, rtol=1e-12):
    """
    Solve F(Kz; q)=0 on the physical branch k1z,k2z >= 0.
    Returns Kz.
    """
    q_min, q_max = q_bounds(E, r_para, Q_para, lattice)

    if q < q_min - 1e-13 or q > q_max + 1e-13:
        raise ValueError(f"q={q} is outside allowed interval [{q_min}, {q_max}].")

    K_low = 2.0 * abs(q)
    K_high = E

    f_low = LHS_Kz(K_low, q, E, r_para, Q_para, lattice)
    f_high = LHS_Kz(K_high, q, E, r_para, Q_para, lattice)

    # Root can sit exactly at endpoint, especially near grazing thresholds.
    if abs(f_low) < xtol:
        return K_low
    if abs(f_high) < xtol:
        return K_high

    if f_low > 0 or f_high < 0:
        raise RuntimeError(f"Bad bracket: F(K_low)={f_low}, F(K_high)={f_high}.")

    return brentq(
        lambda K: LHS_Kz(K, q, E, r_para, Q_para, lattice),
        K_low,
        K_high,
        xtol=xtol,
        rtol=rtol,
        maxiter=100,
    )




def _denom_BM(r_para, q, p_para, E1, E, Q_para, lattice, sigma_func_period):
    """
    Return the unregularized Bethe-Morette W-state denominator.

    The relative momentum q fixes the on-shell total longitudinal momentum Kz,
    which then determines the intermediate photon energy Et. The denominator
    combines the two single-excitation detunings with the difference between
    the current two-photon t-matrix eigenvalue and the reference incoming one.
    """

    Kz = solve_Kz(q, E, r_para, Q_para, lattice)
    rz = Kz / 2 + q

    Et = np.sqrt(np.linalg.norm(r_para) ** 2 + rz**2)
    s_para = BZ_proj(Q_para - r_para, lattice)

    tt = t_reg(p_para, E1, lattice, sigma_func_period) * t_reg(
        BZ_proj(Q_para - p_para, lattice), E - E1, lattice, sigma_func_period
    )

    D1 = Et - lattice.omega_e - sigma_func_period(r_para[0], r_para[1])
    D2 = E - Et - lattice.omega_e - sigma_func_period(s_para[0], s_para[1])
    denom = (
        D1
        * D2
        * (
            t_reg(r_para, Et, lattice, sigma_func_period)
            * t_reg(s_para, E - Et, lattice, sigma_func_period)
            - tt
        )
    )
    return denom


def W_profile_BM(r_para, q, p_para, E1, E, Q_para, lattice, sigma_func_period, eta):
    """
    Evaluate W-state profile at a fixed transverse momentum.

    q parameterizes the longitudinal relative momentum on the energy shell. The
    Jacobian J1 converts from Kz to q, while eta regularizes poles in the
    W-state denominator.
    """

    Kz = solve_Kz(q, E, r_para, Q_para, lattice)
    rz = Kz / 2 + q
    sz = Kz / 2 - q

    Et = np.sqrt(np.linalg.norm(r_para) ** 2 + rz**2)

    J1 = 2 / ((rz / Et) + (sz / (E - Et)))

    s_para = BZ_proj(Q_para - r_para, lattice)
    # eigenvalue of this W state
    tt = t_reg(p_para, E1, lattice, sigma_func_period) * t_reg(
        BZ_proj(Q_para - p_para, lattice), E - E1, lattice, sigma_func_period
    )

    D1 = Et - lattice.omega_e - sigma_func_period(r_para[0], r_para[1])
    D2 = E - Et - lattice.omega_e - sigma_func_period(s_para[0], s_para[1])
    denom = (
        D1
        * D2
        * (
            t_reg(r_para, Et, lattice, sigma_func_period)
            * t_reg(s_para, E - Et, lattice, sigma_func_period)
            - tt
        )
    )
    # the regulator eta is added to the denominator
    return J1 / (denom + 1j * eta)


def W_k_sp_grid(
    r_para,
    p_para,
    E1,
    E,
    Q_para,
    Zc,
    lattice,
    sigma_func_period,
    n_points,
    eta,
    eps=1e-10,
):
    """
    Sample the outgoing W-state profile on an evenly spaced q grid.

    Returns the q grid and the corresponding complex momentum-space amplitudes,
    including the outgoing center-of-mass phase exp(i Zc Kz). The endpoint
    offset eps avoids threshold singularities at the edge of the allowed q range.
    """
    q_min, q_max = q_bounds(E, r_para, Q_para, lattice)
    q_grid = np.linspace(q_min + eps, q_max - eps, n_points, endpoint=False)
    value_grid = np.zeros(len(q_grid), dtype=np.complex128)
    i = 0
    for q in q_grid:
        Kz = solve_Kz(q, E, r_para, Q_para, lattice)
        value_grid[i] = W_profile_BM(
            r_para, q, p_para, E1, E, Q_para, lattice, sigma_func_period, eta
        ) * np.exp(1j * Zc * Kz)  # positive sign for outgoing wave
        i += 1
    return q_grid, value_grid



def quad_FT(r_para, p_para, Zc, z, E1, E, Q_para, lattice, sigma_func_period, eta):
    """quad version of Fourier transform, for benchmarking against FFT."""
    def W_quad_integrand(q):
        return (
            W_profile_BM(
                r_para, q, p_para, E1, E, Q_para, lattice, sigma_func_period, eta
            )
            * np.exp(1j * Zc * solve_Kz(q, E, r_para, Q_para, lattice))
            * np.exp(1j * q * z)
        )

    q_min, q_max = q_bounds(E, r_para, Q_para, lattice)
    integral, integral_err = quad_vec(
        W_quad_integrand,
        q_min + 1e-10,
        q_max - 1e-10,
        epsabs=1e-9,
        epsrel=1e-9,
        limit=2000,
        quadrature="gk21",
    )

    return integral, integral_err


__all__ = [
    "W_profile_BM",
    "solve_Kz",
    "W_k_sp_grid",
    "quad_FT"
]
