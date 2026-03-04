import sys
from pathlib import Path

# Add src to path so imports work when run from project root (must be before imports from src)
sys.path.insert(0, str(Path(__file__).parent / "src"))


import numpy as np
from smatrix import (
    create_self_energy_interpolator_numba,
    square_lattice,
)
from eigenstate_solving import eigen_eq_itr_batch
#from scattering.filters import GH_filter_vectorized

if __name__ == "__main__":
    # Load from file (comment out if computing fresh)
    sigma_data = np.load("data/sigma_grid0f4a.npz")
    kx = sigma_data["kx"]
    ky = sigma_data["ky"]
    sigma_grid = sigma_data["sigma_grid"]
    sigma_func_period_numba = create_self_energy_interpolator_numba(
        kx, ky, sigma_grid, lattice=square_lattice
    )
#    collective_lamb_shift = self_energy(
#        0, 0, square_lattice.a, square_lattice.d, square_lattice.omega_e, alpha
#    ).real

#    for row in range(tmp.shape[0]):
#        for col in range(tmp.shape[1]):
#            print(tmp[row,col])
#print(GH_filter_vectorized(np.array([20,50]), 205, square_lattice))
    print(eigen_eq_itr_batch(np.array([20,50]), 205, square_lattice, sigma_func_period_numba, np.exp(1j*np.pi/4)))
#    print(_eigen_eq_integrand(205, np.array([20,50]), np.array([0,0]), 1, np.array([0,0]), np.array([0,0]), square_lattice,sigma_func_period_numba,np.exp(1j*np.pi/4)))