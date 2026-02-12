import numpy as np
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smatrix import parallel_self_energy_grid
from smatrix import square_lattice

if __name__ == "__main__":

    '''
    # Compute self-energy over 10x10 grid with omega=1
    kx, ky, sigma_grid = parallel_self_energy_grid(n_points=50, omega=square_lattice.omega_e, n_jobs=8, lattice=square_lattice)
    np.savez("data/sigma_grid0f6a.npz",kx=kx,ky=ky,sigma_grid=sigma_grid)
    '''

    kx, ky, omega, sigma_grid = parallel_self_energy_grid(n_points=30, omega=square_lattice.omega_e, n_jobs=6, lattice=square_lattice, dim=3, omega_cutoff=square_lattice.q/2-square_lattice.omega_e-0.001, omega_points=5)
    np.savez("data/sigma_grid0f4a_3D.npz",kx=kx,ky=ky,omega=omega,sigma_grid=sigma_grid)