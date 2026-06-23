import numpy as np
from eigenstate_solving.eigen_eq_integrand import BZ_proj
from model.model import SquareLattice
from smatrix import t_reg


def _make_integrand_in_I_term(E:float,Q:np.ndarray,eta:float,tau:complex,lattice:SquareLattice,sigma_func_period):
    "This function returns a integrand in the I_{E1,k_para} term by assuming the lattice is deep subwavelength. Only zeroth order diffraction occcurs."

    def _integrand(x):
        ux,uy,uz = x
        r_para_x = lattice.q/2 * ux
        r_para_y = lattice.q/2 * uy
        r_para = np.array([r_para_x, r_para_y])
        r_para_norm = np.linalg.norm(r_para)

        s_para = BZ_proj(Q - r_para, lattice)
        s_para_norm = np.linalg.norm(s_para)

        width = E - r_para_norm - s_para_norm

        E1t = r_para_norm + width * uz
        E2t = E - E1t
        # this if condition represents two-photon light cone Heviside step function in the integrand.
        if width <= 0:
            return 0.0+0.0j
        Sgm_r = sigma_func_period(r_para[0],r_para[1])
        Sgm_s = sigma_func_period(s_para[0],s_para[1])

        num1 = np.imag(Sgm_r)
        denom1 = (E1t-lattice.omega_e-Sgm_r)**2
        num2 = np.imag(Sgm_s)
        denom2 = (E2t-lattice.omega_e-Sgm_s)**2

        fraction1 = num1/denom1
        fraction2 = num2/denom2

        tt = t_reg(r_para, E1t, lattice, sigma_func_period) * t_reg(s_para, E2t, lattice, sigma_func_period)
        eigenvalue_term = 1/(np.exp(eta)*tau - tt)

        return width * fraction1 * fraction2 * eigenvalue_term

    return _integrand



def I_term_integ_vegas(E:float,Q:np.ndarray,eta,tau:complex,nitn:int,neval:int,lattice:SquareLattice,sigma_func_period):

    import vegas
    integrand = _make_integrand_in_I_term(E, Q, eta,tau,lattice,sigma_func_period)
    # separate the real and imaginary part of the integrand for vegas integrator
    def integrand_re(x):
        return float(np.real(integrand(x)))
    def integrand_im(x):
        return float(np.imag(integrand(x)))

    integ_re = vegas.Integrator([[-1, 1], [-1, 1], [0, 1]])
    integ_im = vegas.Integrator([[-1, 1], [-1, 1], [0, 1]])

    integ_re(integrand_re, nitn=nitn, neval=neval)
    integ_im(integrand_im, nitn=nitn, neval=neval)
    result_re = integ_re(integrand_re, nitn=nitn, neval=neval)
    result_im = integ_im(integrand_im, nitn=nitn, neval=neval)

    result = result_re.mean + 1j * result_im.mean
    result = lattice.a**2/(2*np.pi) * result
    return result


def _make_integrand_in_I_term_batch(
    E: float,
    Q: np.ndarray,
    eta: float,
    tau: complex,
    lattice: SquareLattice,
    sigma_func_period,
):
    """Batch-compatible version of _make_integrand_in_I_term for Vegas."""
    base_integrand = _make_integrand_in_I_term(
        E, Q, eta, tau, lattice, sigma_func_period
    )
    Q_arr = np.asarray(Q, dtype=np.float64)
    q = float(lattice.q)
    half_q = q / 2.0
    tEQ = np.exp(eta) * tau

    def _integrand(x):
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            return base_integrand(x_arr)

        if x_arr.ndim != 2 or x_arr.shape[1] != 3:
            raise ValueError("Input must be shape (3,) or (n, 3)")

        r_para = np.empty((x_arr.shape[0], 2), dtype=np.float64)
        r_para[:, 0] = half_q * x_arr[:, 0]
        r_para[:, 1] = half_q * x_arr[:, 1]
        r_para_norm = np.linalg.norm(r_para, axis=1)

        s_para = ((Q_arr - r_para + half_q) % q) - half_q
        s_para_norm = np.linalg.norm(s_para, axis=1)

        width = E - r_para_norm - s_para_norm
        valid = width > 0.0
        out = np.zeros(x_arr.shape[0], dtype=np.complex128)
        if not np.any(valid):
            return out

        E1t = r_para_norm[valid] + width[valid] * x_arr[valid, 2]
        E2t = E - E1t
        r_valid = r_para[valid]
        s_valid = s_para[valid]

        Sgm_r = np.array(
            [sigma_func_period(float(kx), float(ky)) for kx, ky in r_valid],
            dtype=np.complex128,
        )
        Sgm_s = np.array(
            [sigma_func_period(float(kx), float(ky)) for kx, ky in s_valid],
            dtype=np.complex128,
        )

        fraction1 = np.imag(Sgm_r) / (E1t - lattice.omega_e - Sgm_r) ** 2
        fraction2 = np.imag(Sgm_s) / (E2t - lattice.omega_e - Sgm_s) ** 2
        tt = t_reg(r_valid, E1t, lattice, sigma_func_period) * t_reg(
            s_valid, E2t, lattice, sigma_func_period
        )
        eigenvalue_term = 1 / (tEQ - tt)

        out[valid] = width[valid] * fraction1 * fraction2 * eigenvalue_term
        return out

    return _integrand


def I_term_integ_vegas_batch(
    E: float,
    Q: np.ndarray,
    eta,
    tau: complex,
    nitn: int,
    neval: int,
    lattice: SquareLattice,
    sigma_func_period,
):
    import vegas

    integrand = _make_integrand_in_I_term_batch(
        E, Q, eta, tau, lattice, sigma_func_period
    )

    @vegas.lbatchintegrand
    def integrand_re(xbatch, integrand=integrand):
        xbatch_arr = np.ascontiguousarray(xbatch, dtype=np.float64)
        values = integrand(xbatch_arr)
        return np.ascontiguousarray(np.real(values), dtype=np.float64)

    @vegas.lbatchintegrand
    def integrand_im(xbatch, integrand=integrand):
        xbatch_arr = np.ascontiguousarray(xbatch, dtype=np.float64)
        values = integrand(xbatch_arr)
        return np.ascontiguousarray(np.imag(values), dtype=np.float64)

    integ_re = vegas.Integrator([[-1, 1], [-1, 1], [0, 1]])
    integ_im = vegas.Integrator([[-1, 1], [-1, 1], [0, 1]])

    integ_re(integrand_re, nitn=nitn, neval=neval)
    integ_im(integrand_im, nitn=nitn, neval=neval)
    result_re = integ_re(integrand_re, nitn=nitn, neval=neval)
    result_im = integ_im(integrand_im, nitn=nitn, neval=neval)

    result = result_re.mean + 1j * result_im.mean
    result = lattice.a**2 / (2 * np.pi) * result
    return result


__all__ = [
    "_make_integrand_in_I_term",
    "_make_integrand_in_I_term_batch",
    "I_term_integ_vegas",
    "I_term_integ_vegas_batch",
]
