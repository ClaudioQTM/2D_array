import sys
from pathlib import Path

# Add src to path so imports work when run from project root (must be before imports from src)
sys.path.insert(0, str(Path(__file__).parent / "src"))


import numpy as np
from model.defaults import alpha
from model.model import self_energy
# from smatrix.amplitudes import S_disconnected
from smatrix import create_self_energy_interpolator_numba, square_lattice, tau_matrix_element
from eigenstate_solving import eigen_eq_itr_batch
from joblib import Parallel, delayed

if __name__ == "__main__":
    # Load from file (comment out if computing fresh)
    sigma_data = np.load("data/sigma_grid0f1a.npz")
    kx = sigma_data["kx"]
    ky = sigma_data["ky"]
    sigma_grid = sigma_data["sigma_grid"]
    sigma_func_period_numba = create_self_energy_interpolator_numba(
        kx, ky, sigma_grid, lattice=square_lattice
    )
    """
    evaluated_omega = square_lattice.omega_e
    value = self_energy(600,600,square_lattice.a,square_lattice.d,evaluated_omega,alpha)
    value_BM = sigma_func_period_numba(600,600)
   # print(evaluated_omega)
    print(value)
    print(value_BM)
    """
    collective_lamb_shift = self_energy(
        0, 0, square_lattice.a, square_lattice.d, square_lattice.omega_e, alpha
    ).real
    """
    Q = np.array([0.0, 0.0], dtype=np.float64)

    E = square_lattice.omega_e+50
    k_para = np.array([60, 40])
    print(
        -2
        * square_lattice.a**2
        * self_energy(
            k_para[0], k_para[1], square_lattice.a, square_lattice.d, E, alpha
        ).imag
        * np.sqrt(E**2 - np.linalg.norm(k_para) ** 2)
        / E
    )
    print(np.abs(square_lattice.ge(coord_convert(k_para, E))) ** 2)
    """

    results = Parallel(n_jobs=6)(delayed(eigen_eq_itr_batch)(np.array([0,0]), 205, square_lattice, sigma_func_period_numba, np.exp(1j*phi),neval=int(5e6),tau_matrix_calculation=False) for phi in np.linspace(0, 2*np.pi, 18))
    results = np.asarray(results, dtype=np.complex128)
    results = tau_matrix_element(205, np.array([0,0]), square_lattice, sigma_func_period_numba) * results
    print(results)

    #    plot_integrand1(205, np.array([0,0]), np.array([0,0]), np.array([0,0]), 0.1, _make_eigen_eq_integrand, sigma_func_period_numba, square_lattice, np.exp(1j*np.pi))
    """
    integrand_reg = _make_eigen_eq_integrand(
        200,
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        square_lattice,
        sigma_func_period_numba,
        np.exp(1j * np.pi),
    )

    integrand = _make_eigen_eq_integrand_OLD(
        200,
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        square_lattice,
        sigma_func_period_numba,
        np.exp(1j * np.pi),
    )

    print(integrand_reg([0, 0, 0.01]))
    print(integrand([0, 0, 0.01]))
    """
#    print(legs(np.array([0,0]), square_lattice.omega_e, np.array([0,0]), square_lattice.omega_e, square_lattice, sigma_func_period_numba, direction="in"))
#    prop1 = sw_propagator(np.array([0,0]), square_lattice.omega_e, square_lattice, sigma_func_period_numba)*square_lattice.ge(np.array([0,0,square_lattice.omega_e]))
#    print(prop1**2)
