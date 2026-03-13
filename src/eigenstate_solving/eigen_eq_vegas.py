import numpy as np
from model.model import SquareLattice
from typing import Callable
from .eigen_eq_integrand import _make_eigen_eq_integrand, _make_eigen_eq_integrand_numba
from smatrix.tau import tau_matrix_element
from scattering.filters import GH_filter_vectorized
import warnings
import math


def _should_restratify(train_result, integrator) -> bool:
    return (
        math.isfinite(float(train_result.mean))
        and math.isfinite(float(train_result.sdev))
        and abs(float(train_result.mean)) > 0.0
        and abs(float(train_result.sdev)) > 0.0
        and math.isfinite(float(getattr(integrator, "sum_wgt", np.nan)))
        and math.isfinite(float(getattr(integrator, "sum_sigf2", np.nan)))
    )


def eigen_eq_itr(
    Q: np.ndarray,
    E: float,
    lattice: SquareLattice,
    sigma_func_period: Callable,
    tEQ: float,
    *,
    neval: int = int(1e5),
    nitn: int = int(10),
):
    if abs(Q[0]) > lattice.q / 2 or abs(Q[1]) > lattice.q / 2:
        raise ValueError("Q is out of the first BZ")
    import vegas

    """integrate the eigenvalue equation using Vegas"""
    prefactor = tau_matrix_element(E, Q, lattice, sigma_func_period)
    gh_pairs = GH_filter_vectorized(E, Q, lattice)

    summand = 0.0 + 0.0j
    for gh_pair in gh_pairs:
        G, H = gh_pair
        eigen_eq_integrand = _make_eigen_eq_integrand(
            E, Q, G, H, lattice, sigma_func_period, tEQ
        )
        integ_re = vegas.Integrator([[-1, 1], [-1, 1], [0, 1]])
        integ_im = vegas.Integrator([[-1, 1], [-1, 1], [0, 1]])
        
        def integrand_re(x, eigen_eq_integrand=eigen_eq_integrand):
            return float(np.real(eigen_eq_integrand(x)))

        def integrand_im(x, eigen_eq_integrand=eigen_eq_integrand):
            return float(np.imag(eigen_eq_integrand(x)))
      
        
        integ_re(integrand_re, nitn=nitn, neval=neval)
        integ_im(integrand_im, nitn=nitn, neval=neval)
        result_re = integ_re(integrand_re, nitn=nitn, neval=neval)
        result_im = integ_im(integrand_im, nitn=nitn, neval=neval)
        summand += result_re.mean + 1j * result_im.mean

    return prefactor * summand


def eigen_eq_itr_batch(
    Q: np.ndarray,
    E: float,
    lattice: SquareLattice,
    sigma_func_period: Callable,
    tEQ: float,
    *,
    neval: int = int(1e6),
    nitn1: int = int(10),
    nitn2: int = int(10),
    q_threshold: float = 0.05,
):
    if abs(Q[0]) > lattice.q / 2 or abs(Q[1]) > lattice.q / 2:
        raise ValueError("Q is out of the first BZ")
    import vegas

    """Integrate the eigenvalue equation using Vegas batchmode."""
#    prefactor = tau_matrix_element(E, Q, lattice, sigma_func_period)
    # temporary fix to avoid prefactor calculation
    prefactor = 1.0
    gh_pairs = GH_filter_vectorized(E, Q, lattice)

    summand = 0.0 + 0.0j
    for gh_pair in gh_pairs:
        G, H = gh_pair
        eigen_eq_integrand = _make_eigen_eq_integrand_numba(
            E, Q, G, H, lattice, sigma_func_period, tEQ
        )
        integ_re = vegas.Integrator([[-1, 1], [-1, 1], [0, 1]])
        integ_im = vegas.Integrator([[-1, 1], [-1, 1], [0, 1]])

        @vegas.lbatchintegrand
        def integrand_re(xbatch, eigen_eq_integrand=eigen_eq_integrand):
            xbatch_arr = np.ascontiguousarray(xbatch, dtype=np.float64)
            values = eigen_eq_integrand(xbatch_arr)
            return np.ascontiguousarray(np.real(values), dtype=np.float64)

        @vegas.lbatchintegrand
        def integrand_im(xbatch, eigen_eq_integrand=eigen_eq_integrand):
            xbatch_arr = np.ascontiguousarray(xbatch, dtype=np.float64)
            values = eigen_eq_integrand(xbatch_arr)
            return np.ascontiguousarray(np.imag(values), dtype=np.float64)

        train_re = integ_re(integrand_re, nitn=nitn1, neval=neval)  # training step
        if _should_restratify(train_re, integ_re):
            vegas.restratify(integ_re, integrand_re, 2, verbose=True)
        result_re = integ_re(integrand_re, nitn=nitn2, neval=neval)
        train_im = integ_im(integrand_im, nitn=nitn1, neval=neval) # training step
        if _should_restratify(train_im, integ_im):
            vegas.restratify(integ_im, integrand_im, 2, verbose=True)
        result_im = integ_im(integrand_im, nitn=nitn2, neval=neval)
        if result_re.Q < q_threshold or result_im.Q < q_threshold:
            warnings.warn(
                f"Low Vegas Q: Q_re={result_re.Q:.3g}, Q_im={result_im.Q:.3g} at G={G}, H={H}",
                RuntimeWarning,
                stacklevel=2,
            )
        summand += result_re.mean + 1j * result_im.mean
        print(f"result_re.summary(): {result_re.summary()}")
        print(f"result_im.summary(): {result_im.summary()}")
        print(f"G={G}, H={H}")
    return prefactor * summand