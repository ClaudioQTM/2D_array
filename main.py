import sys
from pathlib import Path

# Add src to path so imports work when run from project root (must be before imports from src)
sys.path.insert(0, str(Path(__file__).parent / "src"))


import numpy as np
from smatrix import (
    create_self_energy_interpolator_numba,
    square_lattice, S_disconnected,
)
from eigenstate_solving import eigen_eq_itr_batch
from joblib import Parallel, delayed
from smatrix.tau import tau_matrix_element
from model import self_energy
from model.defaults import alpha
#from scattering.filters import GH_filter_vectorized
from smatrix.amplitudes import S_disconnected


if __name__ == "__main__":
    # Load from file (comment out if computing fresh)
    sigma_data = np.load("data/sigma_grid0f4a.npz")
    kx = sigma_data["kx"]
    ky = sigma_data["ky"]
    sigma_grid = sigma_data["sigma_grid"]
    sigma_func_period_numba = create_self_energy_interpolator_numba(
        kx, ky, sigma_grid, lattice=square_lattice
    )
    collective_lamb_shift = self_energy(
        0, 0, square_lattice.a, square_lattice.d, square_lattice.omega_e, alpha
    ).real

#    for row in range(tmp.shape[0]):
#        for col in range(tmp.shape[1]):
#            print(tmp[row,col])
#print(GH_filter_vectorized(np.array([20,50]), 205, square_lattice))
    
    results = Parallel(n_jobs=6)(delayed(eigen_eq_itr_batch)(np.array([0,0]), 205, square_lattice, sigma_func_period_numba, np.exp(1j*phi),neval=int(5e6),tau_matrix_calculation=False) for phi in np.linspace(0, 2*np.pi, 6))
    results = np.asarray(results, dtype=np.complex128)
    results = tau_matrix_element(205, np.array([0,0]), square_lattice, sigma_func_period_numba) * results
    print(results)

#    integrand_tmp = _make_eigen_eq_integrand(250, np.array([0,0]), np.array([0,0]), np.array([0,0]), square_lattice, sigma_func_period_numba, np.exp(1j*np.pi/4))
#    print(integrand_tmp(np.array([0.0,0.5,0.5])))

#    print(plot_integrand1(205, np.array([0,0]), np.array([0,0]), np.array([0,0]), 0.5, sigma_func_period_numba, square_lattice, np.exp(1j*np.pi/4)))