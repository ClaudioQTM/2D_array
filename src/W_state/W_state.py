from pylab import ndarray
from smatrix import t_reg
from eigenstate_solving import BZ_proj
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad_vec
from smatrix import L

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


"""
def solve_Kz2(q, E, r_para, Q_para, lattice):
    abs_r_para = np.linalg.norm(r_para)
    abs_s_para = np.linalg.norm(BZ_proj(Q_para - r_para, lattice))

    R = abs_r_para**2
    S = abs_s_para**2

    A = 4*q**2 - E**2
    B= 4*q*(R-S)
    C = (E**2+R-S)**2 - 4*(R+q**2)*E**2

    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        raise ValueError("No real root for Kz.")

    if A == 0.0:
        if B == 0.0:
            return E
        else:
            Kz_candidates = [-C/B]
    else:
        Kz_candidates = np.array([(-B+np.sqrt(discriminant))/(2*A), (-B-np.sqrt(discriminant))/(2*A)])

    tol = 1e-9 * max(1.0, abs(E))
    lower = 2 * np.abs(q)

    mask = (
        ((Kz_candidates <= E) | np.isclose(Kz_candidates, E, rtol=0.0, atol=tol))
        &
        ((lower <= Kz_candidates) | np.isclose(Kz_candidates, lower, rtol=0.0, atol=tol))
    )

    Kz_candidates = Kz_candidates[mask]
    if len(Kz_candidates) != 1:
        raise ValueError("Kz is not unique.")

    return Kz_candidates[0]
"""


def solve_Kz_vec(q, E, r_para, Q_para, lattice, tol=1e-12):
    q_arr = np.asarray(q, dtype=float)
    scalar_input = q_arr.ndim == 0
    q_flat = q_arr.reshape(-1)

    abs_r_para = np.linalg.norm(r_para)
    abs_s_para = np.linalg.norm(BZ_proj(Q_para - r_para, lattice))
    threshold = abs_r_para + abs_s_para
    if E < threshold:
        raise ValueError("No propagating two-photon channel: E < m1 + m2.")

    R = abs_r_para**2
    S = abs_s_para**2
    scale = max(1.0, abs(E))
    endpoint_tol = max(tol, 1e-12)
    q_tol = endpoint_tol * scale
    physical_tol = max(1e-9 * scale, q_tol)

    rad_left = (E - abs_r_para) ** 2 - S
    rad_right = (E - abs_s_para) ** 2 - R
    rad_tol = endpoint_tol * max(1.0, E**2)
    if rad_left < -rad_tol or rad_right < -rad_tol:
        raise ValueError("Negative value under square root in q_bounds.")
    q_min = -0.5 * np.sqrt(max(rad_left, 0.0))
    q_max = 0.5 * np.sqrt(max(rad_right, 0.0))
    if np.any((q_flat < q_min - q_tol) | (q_flat > q_max + q_tol)):
        raise ValueError(f"q is outside allowed interval [{q_min}, {q_max}].")
    q_flat = np.where(np.isclose(q_flat, q_min, rtol=0.0, atol=q_tol), q_min, q_flat)
    q_flat = np.where(np.isclose(q_flat, q_max, rtol=0.0, atol=q_tol), q_max, q_flat)

    def energy_residual(Kz, q_vals):
        k1z = 0.5 * Kz + q_vals
        k2z = 0.5 * Kz - q_vals
        return np.hypot(abs_r_para, k1z) + np.hypot(abs_s_para, k2z) - E

    def bisect_fallback(q_vals):
        lower_vals = 2 * np.abs(q_vals)
        upper_vals = np.full_like(q_vals, E, dtype=float)
        f_low = energy_residual(lower_vals, q_vals)
        f_high = energy_residual(upper_vals, q_vals)

        out = np.empty_like(q_vals, dtype=float)
        at_low = np.abs(f_low) <= endpoint_tol
        at_high = np.abs(f_high) <= endpoint_tol
        active = ~(at_low | at_high)

        if np.any(f_low[active] > endpoint_tol) or np.any(f_high[active] < -endpoint_tol):
            raise RuntimeError("Bad bracket in solve_Kz_vec fallback.")

        out[at_low] = lower_vals[at_low]
        out[at_high] = upper_vals[at_high]

        low = lower_vals[active]
        high = upper_vals[active]
        q_active = q_vals[active]
        for _ in range(80):
            mid = 0.5 * (low + high)
            f_mid = energy_residual(mid, q_active)
            move_low = f_mid < 0
            low = np.where(move_low, mid, low)
            high = np.where(move_low, high, mid)
        out[active] = 0.5 * (low + high)
        return out

    A = 4 * q_flat**2 - E**2
    B = 4 * q_flat * (R - S)
    C = (E**2 + R - S) ** 2 - 4 * (R + q_flat**2) * E**2

    discriminant = B**2 - 4 * A * C
    bad_discriminant = discriminant < 0
    discriminant = np.maximum(discriminant, 0.0)

    Kz_plus = np.full_like(q_flat, np.nan, dtype=float)
    Kz_minus = np.full_like(q_flat, np.nan, dtype=float)
    A_tol = endpoint_tol * max(1.0, E**2)
    nonlinear_mask = np.abs(A) > A_tol
    linear_mask = ~nonlinear_mask

    sqrt_discriminant = np.sqrt(discriminant[nonlinear_mask])
    Kz_plus[nonlinear_mask] = (-B[nonlinear_mask] + sqrt_discriminant) / (2 * A[nonlinear_mask])
    Kz_minus[nonlinear_mask] = (-B[nonlinear_mask] - sqrt_discriminant) / (2 * A[nonlinear_mask])

    linear_nonzero = linear_mask & (B != 0.0)
    Kz_plus[linear_nonzero] = -C[linear_nonzero] / B[linear_nonzero]
    Kz_plus[linear_mask & (B == 0.0)] = E

    Kz_candidates = np.vstack([Kz_plus, Kz_minus])
    
    lower = 2 * np.abs(q_flat)

    mask = (
        np.isfinite(Kz_candidates)
        & ((Kz_candidates <= E) | np.isclose(Kz_candidates, E, rtol=0.0, atol=physical_tol))
        & ((lower <= Kz_candidates) | np.isclose(Kz_candidates, lower, rtol=0.0, atol=physical_tol))
    )

    candidate_residual = np.abs(energy_residual(Kz_candidates, q_flat[None, :]))
    candidate_residual = np.where(mask, candidate_residual, np.inf)
    n_candidates = np.sum(mask, axis=0)
    chosen = np.argmin(candidate_residual, axis=0)
    best_residual = candidate_residual[chosen, np.arange(q_flat.size)]
    Kz = Kz_candidates[chosen, np.arange(q_flat.size)]

    near_threshold = E - threshold < 1e-8 * scale
    fallback_mask = (
        bad_discriminant
        | linear_mask
        | (n_candidates != 1)
        | ~np.isfinite(best_residual)
        | (best_residual > 1e-10)
        | (Kz < lower - physical_tol)
        | (Kz > E + physical_tol)
    )
    if near_threshold:
        fallback_mask = np.ones_like(fallback_mask, dtype=bool)
    if np.any(fallback_mask):
        Kz[fallback_mask] = bisect_fallback(q_flat[fallback_mask])
    Kz = np.clip(Kz, lower, E)

    if scalar_input:
        return Kz[0]
    return Kz.reshape(q_arr.shape)




def _denom_BM_OLD(r_para, q, p_para, E1, E, Q_para, eta,lattice, sigma_func_period):
    """
    Return the unregularized W-state denominator.

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
    reg_tt = (np.expm1(eta)+1)*tt # (np.expm1(eta)+1) is a more precise alternative to np.exp(eta) for eta close to 0.
    denom = (
        D1
        * D2
        * (reg_tt -
            t_reg(r_para, Et, lattice, sigma_func_period)
            * t_reg(s_para, E - Et, lattice, sigma_func_period)
        )
    )
    return Kz, denom


def _denom_BM_vec_OLD(r_para, q, p_para, E1, E, Q_para, eta,lattice, sigma_func_period):
    

    Kz = solve_Kz_vec(q, E, r_para, Q_para, lattice)
    rz = Kz / 2 + q

    Et = np.sqrt(np.linalg.norm(r_para) ** 2 + rz**2)
    s_para = BZ_proj(Q_para - r_para, lattice)

    tt = t_reg(p_para, E1, lattice, sigma_func_period) * t_reg(
        BZ_proj(Q_para - p_para, lattice), E - E1, lattice, sigma_func_period
    )

    D1 = Et - lattice.omega_e - sigma_func_period(r_para[0], r_para[1])
    D2 = E - Et - lattice.omega_e - sigma_func_period(s_para[0], s_para[1])
    reg_tt = (np.expm1(eta)+1)*tt # (np.expm1(eta)+1) is a more precise alternative to np.exp(eta) for eta close to 0.
    denom = (
        D1
        * D2
        * (reg_tt -
            t_reg(r_para, Et, lattice, sigma_func_period)
            * t_reg(s_para, E - Et, lattice, sigma_func_period)
        )
    )
    return Kz, denom



def _denom_BM_vec(r_para, q, p_para, E1, E, Q_para, eta,lattice, sigma_func_period):
    
    Kz = solve_Kz_vec(q, E, r_para, Q_para, lattice)
    rz = Kz / 2 + q

    Et = np.sqrt(np.linalg.norm(r_para) ** 2 + rz**2)
    s_para = BZ_proj(Q_para - r_para, lattice)

    tt = t_reg(p_para, E1, lattice, sigma_func_period) * t_reg(
        BZ_proj(Q_para - p_para, lattice), E - E1, lattice, sigma_func_period
    )

    reg_tt = (np.expm1(eta)+1)*tt # (np.expm1(eta)+1) is a more precise alternative to np.exp(eta) for eta close to 0.
    denom = reg_tt - t_reg(r_para, Et, lattice, sigma_func_period) * t_reg(s_para, E - Et, lattice, sigma_func_period)
        
    
    return Kz, denom



def W_profile_BM(r_para:ndarray, q:float, p_para:ndarray, E1:float, E:float, Q_para:ndarray, lattice, sigma_func_period, eta:float, MEQ:complex, C_term:complex):
    """
    Evaluate W-state profile at a fixed transverse momentum.

    q parameterizes the longitudinal relative momentum on the energy shell. The
    Jacobian J1 converts from Kz to q, while eta regularizes poles in the
    W-state denominator.
    """

    Kz, denom = _denom_BM_vec(r_para, q, p_para, E1, E, Q_para, eta, lattice, sigma_func_period)
    rz = Kz / 2 + q
    sz = Kz / 2 - q

    Et = np.sqrt(np.linalg.norm(r_para) ** 2 + rz**2)

    # The Jacobian factor when we change from Et1 to relative momentum q coordinate.
    J1 = 2 / ((Et / rz) + ((E - Et)/ sz))

    on_shell_checker1 = rz >= 0.0
    on_shell_checker2 = sz >= 0.0
    if np.any(~on_shell_checker1) or np.any(~on_shell_checker2):
        raise ValueError("rz and sz, one or both of them is imaginary or negative in W_profile_BM.")
    leg_factor = L(r_para, Et, lattice, sigma_func_period, "out",True) * L(BZ_proj(Q_para-r_para,lattice), E-Et, lattice, sigma_func_period, "out",True) 
    connected_term =  -1j/2 * (2*np.pi)**3/lattice.a**4 * MEQ * leg_factor  / denom * C_term
    
    return J1 * connected_term


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
    q_grid,
    eta,
    MEQ,
    C_term
):
    """
    Sample the outgoing W-state profile on a q grid.

    q_grid may be any shape; the returned complex momentum-space amplitudes
    have the same shape and include the outgoing center-of-mass phase
    exp(i Zc Kz).
    """
    q_grid = np.asarray(q_grid, dtype=float)
    Kz_grid = solve_Kz_vec(q_grid, E, r_para, Q_para, lattice)
    return W_profile_BM(
        r_para, q_grid, p_para, E1, E, Q_para, lattice, sigma_func_period, eta
    ,MEQ,C_term) * np.exp(1j * Zc * Kz_grid)  # positive sign for outgoing wave



def quad_FT(r_para, p_para, Zc, z, E1, E, Q_para, lattice, sigma_func_period, eta,MEQ,C_term):
    """quad version of Fourier transform, for benchmarking against FFT."""
    def W_quad_integrand(q):
        return (
            W_profile_BM(
                r_para, q, p_para, E1, E, Q_para, lattice, sigma_func_period, eta,MEQ,C_term
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
    integral = integral / (2 * np.pi)
    return integral, integral_err



def _pole_loc(r_para, p_para, E1, E, Q_para,eta, lattice, sigma_func_period,imrtol=1e-7,imatol=1e-10):
        """Find pole locations along the q axis using the quadratic formula.

        Returns
        -------
        list[list[float]]
            A list of pole locations. Each entry is ``[q, Kz]``, where ``q`` is the
            real pole coordinate along the q axis and ``Kz`` is the corresponding
            total z-direction wavevector. Small numerical imaginary parts are
            discarded before returning.
        """
        r_para = np.asarray(r_para, dtype=float)
        p_para = np.asarray(p_para, dtype=float)

        s_para = BZ_proj(Q_para - r_para, lattice)
        # eigenvalue of this W state
        tt = t_reg(p_para, E1, lattice, sigma_func_period) * t_reg(
            BZ_proj(Q_para - p_para, lattice), E - E1, lattice, sigma_func_period
        )

        reg_tt = (np.expm1(eta)+1)*tt # (np.expm1(eta)+1) is a more precise alternative to np.exp(eta) for eta close to 0.
        # self-energy terms
        Sigma1 = sigma_func_period(r_para[0], r_para[1])
        Sigma2 = sigma_func_period(s_para[0], s_para[1])
        r_para_norm = np.hypot(r_para[0], r_para[1])
        s_para_norm = np.hypot(s_para[0], s_para[1])
        # the follows are the coefficient terms in the quadratic equation
        # coefficient for E1tilde**2
        A = reg_tt - 1
        
        # coefficient for E1tilde
        B = (
            E * (1 - reg_tt)
            + (
                sigma_func_period(s_para[0], s_para[1])
                - sigma_func_period(r_para[0], r_para[1])
            )
            * reg_tt
            + np.conj(sigma_func_period(r_para[0], r_para[1]))
            - np.conj(sigma_func_period(s_para[0], s_para[1]))
        )
        # constant term
        C = (
            E * Sigma1 * reg_tt
            - Sigma1 * Sigma2 * reg_tt
            - E * lattice.omega_e
            + E * reg_tt * lattice.omega_e
            - Sigma1 * reg_tt * lattice.omega_e
            - Sigma2 * reg_tt * lattice.omega_e
            + lattice.omega_e**2
            - reg_tt * lattice.omega_e**2
            - E * np.conjugate(Sigma1)
            + lattice.omega_e * np.conjugate(Sigma1)
            + lattice.omega_e * np.conjugate(Sigma2)
            + np.conjugate(Sigma1) * np.conjugate(Sigma2)
        )


        if np.isclose(A,0.0):
            if np.isclose(B,0.0):
                if np.isclose(C,0.0):
                    print("The denominator vanishes for all E1tilde.")
                else:
                    print("No root for all E1tilde.")
                return None
            # single root
            else:
                root = np.array([- C / B])
            # two roots
        else:
            discriminant = B**2 - 4 * A * C
            discriminant = complex(discriminant)
            if np.isclose(discriminant,0.0): # a single double root
                root = np.array([-B/(2*A)])
            else: # quadratic formula
                root = np.array([
                    (-B + np.sqrt(discriminant)) / (2 * A),
                    (-B - np.sqrt(discriminant)) / (2 * A),
                ])

        # The imaginary part of the root for E1tilde never vanishes exactly due to numerical errors, but it should be discarded when it is small enough.
        mask1 = np.isclose(np.abs(np.imag(root)),0.0,rtol = imrtol, atol = imatol)
        root = root[mask1]
        root = np.real(root)

        # only keep the roots that satisfy the physical constraints of |r_para| \leq E1tilde \leq E-|s_para|
        mask2 = (r_para_norm <= root) & (root <= E-s_para_norm)
        root = root[mask2]

        val_list = []
        # convert the root in E1tilde to the corresponding q and Kz values
        for E1tilde in root:
            rz_arg = E1tilde**2 - np.hypot(r_para[0], r_para[1]) ** 2
            sz_arg = (E - E1tilde) ** 2 - np.hypot(s_para[0], s_para[1]) ** 2
            # discard roots that would produce NaN z-momenta
            if not np.isfinite(rz_arg) or not np.isfinite(sz_arg) or rz_arg < 0 or sz_arg < 0:
                continue
            rz = np.sqrt(rz_arg)
            sz = np.sqrt(sz_arg)

            q = (rz - sz) / 2
            Kz = rz + sz

            q_min, q_max = q_bounds(E, r_para, Q_para, lattice)
            # discard the roots outside of the finite interval
            if q <= q_min or q >= q_max:
                continue
            Kz = np.real(Kz)
            val_list.append([q, Kz])
        return val_list



"""
def peak_width_estimator_OLD(r_para, p_para, E1, E, Q_para,eta,lattice, sigma_func_period_numba, xatol=1e-10):
    from scipy.optimize import minimize_scalar

    root_list = _pole_loc(r_para, p_para, E1, E, Q_para, 0.0,lattice, sigma_func_period_numba)

    if not root_list:
        print("No pole on the real axis, no eta is needed.")
        return None
    q_root_list = sorted(root[0] for root in root_list)

    # If two roots are very close to each other, we treat them as a single double root.
    if len(q_root_list) == 2 and np.isclose(q_root_list[0], q_root_list[1]):
        q_root_list = [(q_root_list[0] + q_root_list[1]) / 2]

    def denom_abs_profile(q):
        return np.abs(_denom_BM(r_para, q, p_para, E1, E, Q_para, eta,lattice, sigma_func_period_numba)[1])

    q_min,q_max = q_bounds(E, r_para, Q_para, lattice)
    peak_width_list = []
    if len(q_root_list) == 1:
        # when there is a root for the unregularised denominator, eta might push the pole off the real axis. 
        # In this case, we need to re-estimate the peak location by finding the minimum of the absolute value of the denominator, and then calculate the peak width based on this estimated peak location.
        min_val = minimize_scalar(denom_abs_profile, bounds=(q_min, q_max), method="bounded", options={"xatol": xatol})
        threshold =  2 * min_val.fun
        q_L = brentq(lambda q: denom_abs_profile(q) - threshold, q_min, min_val.x, xtol=xatol, rtol=max(xatol, 1e-15))
        q_R = brentq(lambda q: denom_abs_profile(q) - threshold, min_val.x, q_max, xtol=xatol, rtol=max(xatol, 1e-15))
        peak_width = q_R - q_L
        peak_width_list.append(peak_width)

    if len(q_root_list) == 2:
        # In the regularised case with two roots, the new poles is still close to the original poles. So we use the midpoint of two original roots to split the regions for searching the left and right peaks.
        mid_point = (q_root_list[0] + q_root_list[1]) / 2
        min_val_L = minimize_scalar(denom_abs_profile, bounds=(q_min, mid_point), method="bounded", options={"xatol": xatol})
        min_val_R = minimize_scalar(denom_abs_profile, bounds=(mid_point, q_max), method="bounded", options={"xatol": xatol})

        # calculate the peak width for the left peak
        threshold_L = 2 * min_val_L.fun
        threshold_R = 2 * min_val_R.fun

        q_L1 = brentq(lambda q: denom_abs_profile(q) - threshold_L, q_min, min_val_L.x, xtol=xatol, rtol=max(xatol, 1e-15))
        q_R1 = brentq(lambda q: denom_abs_profile(q) - threshold_L, min_val_L.x, mid_point, xtol=xatol, rtol=max(xatol, 1e-15))
        peak_width_list.append(q_R1 - q_L1)

        # calculate the peak width for the right peak
        peak_val = 1 / minimize_scalar(denom_abs_profile, bounds=(mid_point, q_max), method="bounded", options={"xatol": xatol}).fun
        threshold = peak_val / 2
        q_L2 = brentq(lambda q: denom_abs_profile(q) - threshold_R, mid_point, min_val_R.x, xtol=xatol, rtol=max(xatol, 1e-15))
        q_R2 = brentq(lambda q: denom_abs_profile(q) - threshold_R, min_val_R.x, q_max, xtol=xatol, rtol=max(xatol, 1e-15))
        peak_width_list.append(q_R2 - q_L2)

    width_min = min(peak_width_list) # we pick the width of the smaller peak because if it is resolve by n_rs points. Then the larger peak is resolved by at least n_rs points.
    return width_min
"""

def peak_width_estimator(r_para, p_para, E1, E, Q_para,eta,n_points,lattice, sigma_func_period_numba, xatol=1e-10):
    from scipy.optimize import minimize_scalar

    root_list = _pole_loc(r_para, p_para, E1, E, Q_para, 0.0,lattice, sigma_func_period_numba)

    if not root_list:
        print("No pole on the real axis, no eta is needed.")
        return None

    q_root_list = sorted(root[0] for root in root_list)

    # If two roots are very close to each other, we treat them as a single double root.
    if len(q_root_list) == 2 and np.isclose(q_root_list[0], q_root_list[1]):
    #    q_root_list = [(q_root_list[0] + q_root_list[1]) / 2]
        raise RuntimeError(
            "Near-critical double root detected. "
            "The simple peak_width_estimator is not reliable here."
        )

    root_eta_list = _pole_loc(r_para, p_para, E1, E, Q_para, eta,lattice, sigma_func_period_numba)

    if root_eta_list:
        raise ValueError("A root was classified as real under radial regularisation. Check root imaginary tolerance, coefficient construction, or numerical unitarity.")


    def denom_abs_profile(q):
        return np.abs(_denom_BM_vec_OLD(r_para, q, p_para, E1, E, Q_para, eta,lattice, sigma_func_period_numba)[1])

    q_min, q_max = q_bounds(E, r_para, Q_para, lattice)
    # For a non-zero eta, the pole might be pushed off the real q axis.
    # In this case, we estimate the peak width by looking at the distance between two points in the level-set. 
    # The level-set is defined as the set of points where the absolute value of the denominator equals to a certain threshold. We choose the threshold to be twice the minimum value of the absolute value of the denominator, which is a common choice for estimating the full width at half maximum (FWHM) of a peak.

    # we first find the minimum of the absolute value of the denominator over the entire finite interval.
    global_min = minimize_scalar(denom_abs_profile, bounds=(q_min, q_max), method="bounded", options={"xatol": xatol})
    threshold = 2 * global_min.fun

    # we apply a grid over the entire interval to detect the points where the sign of the difference between the absolute value and the threshold changes, which indicates the crossing of the level-set.
    grid = np.linspace(q_min, q_max, n_points)
#    grid_val = [denom_abs_profile(q) - threshold for q in grid]
    grid_val = denom_abs_profile(grid) - threshold
    crossing_indices = np.flatnonzero(np.signbit(grid_val[:-1]) != np.signbit(grid_val[1:])) # compare the sign bit of an element with the signbit of the element after it. Return the indices where the signs of the two elements disagree.

    # After the crossing points are detected. We use brentq method to find their exact location.
    peak_width_list = []
    if len(crossing_indices) == 2:
        # when there is a root for the unregularised denominator, eta might push the pole off the real axis. 
        # In this case, we need to re-estimate the peak location by finding the minimum of the absolute value of the denominator, and then calculate the peak width based on this estimated peak location.
        q_L = brentq(lambda q: denom_abs_profile(q) - threshold, grid[crossing_indices[0]], grid[crossing_indices[0]+1], xtol=xatol, rtol=max(xatol, 1e-15))
        q_R = brentq(lambda q: denom_abs_profile(q) - threshold, grid[crossing_indices[1]], grid[crossing_indices[1]+1], xtol=xatol, rtol=max(xatol, 1e-15))
        peak_width = q_R - q_L
        peak_width_list.append(peak_width)

    elif len(crossing_indices) == 4:
        q_L1 = brentq(lambda q: denom_abs_profile(q) - threshold, grid[crossing_indices[0]], grid[crossing_indices[0]+1], xtol=xatol, rtol=max(xatol, 1e-15))
        q_R1 = brentq(lambda q: denom_abs_profile(q) - threshold, grid[crossing_indices[1]], grid[crossing_indices[1]+1], xtol=xatol, rtol=max(xatol, 1e-15))
        peak_width_list.append(q_R1 - q_L1)

        q_L2 = brentq(lambda q: denom_abs_profile(q) - threshold, grid[crossing_indices[2]], grid[crossing_indices[2]+1], xtol=xatol, rtol=max(xatol, 1e-15))
        q_R2 = brentq(lambda q: denom_abs_profile(q) - threshold, grid[crossing_indices[3]], grid[crossing_indices[3]+1], xtol=xatol, rtol=max(xatol, 1e-15))
        peak_width_list.append(q_R2 - q_L2)
    else:
        raise ValueError(f"The number of crossings for the level-set is {len(crossing_indices)}, which is not expected 2 or 4. The estimation might fail.")


    #print(f"The peak widths are {peak_width_list}.")
    width_min = min(peak_width_list) # we pick the width of the smaller peak because if it is resolve by n_rs points. Then the larger peak is resolved by at least n_rs points.
    return width_min


def W_disconnect(z,Zc,r_para,k_para,E1,E,Q_para,lattice):
    p_para = BZ_proj(Q_para - k_para,lattice)
    k_para_norm = np.linalg.norm(k_para)
    p_para_norm = np.linalg.norm(p_para)

    E2 = E - E1
    kz = np.sqrt(E1**2 - k_para_norm**2)
    pz = np.sqrt(E2**2 - p_para_norm**2)
    q = (kz - pz) /2
    Kz = kz + pz
    prefactor = (2*np.pi)**2 
        
    if np.allclose(r_para, k_para):
        term1 =  np.exp(1j*q*z)
    else: 
        term1= np.full_like(z, fill_value=0.0+0.0j,dtype=complex)

    if np.allclose(r_para, BZ_proj(p_para,lattice)):
        term2 =  np.exp(-1j*q*z)
    else:
        term2 = np.full_like(z, fill_value=0.0+0.0j,dtype=complex)

    return prefactor * np.exp(1j * Kz * Zc) * (term1 + term2)


__all__ = [
    "W_profile_BM",
    "solve_Kz",
    "W_k_sp_grid",
    "quad_FT",
    "peak_width_estimator",
    "W_disconnect"
]
