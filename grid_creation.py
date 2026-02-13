import numpy as np
import sys
from pathlib import Path
from joblib import Parallel, delayed

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smatrix import square_lattice
from model import self_energy
from smatrix.defaults import alpha
from plots.plot_self_energy import plot_sigma_grid

def parallel_self_energy_grid(n_points, omega, n_jobs, lattice,dim,omega_cutoff=None,omega_points=None):
    k_max = float(lattice.q / 2)
    kx_grid = np.linspace(0, k_max, n_points)
    ky_grid = np.linspace(0, k_max, n_points)

    k_points = [(kx, ky) for kx in kx_grid for ky in ky_grid]
    if dim == 3:
        omega_grid = np.linspace(omega-omega_cutoff, omega+omega_cutoff, omega_points)
        results = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(self_energy)(kx, ky, lattice.a, lattice.d, omega, alpha) for (kx, ky) in k_points for omega in omega_grid
        )
        self_energy_grid = np.array(results, dtype=complex).reshape(n_points, n_points, omega_points)
        return kx_grid, ky_grid, omega_grid, self_energy_grid
    elif dim == 2:
        results = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(self_energy)(kx, ky, lattice.a, lattice.d, omega, alpha) for (kx, ky) in k_points
        )
        self_energy_grid = np.array(results, dtype=complex).reshape(n_points, n_points)
        return kx_grid, ky_grid, self_energy_grid
    else:
        raise ValueError(f"Invalid dimension: {dim}")

if __name__ == "__main__":


    kx, ky, sigma_grid = parallel_self_energy_grid(n_points=64, omega=square_lattice.omega_e, n_jobs=8, lattice=square_lattice, dim=2)
    np.savez("data/sigma_grid0f4a.npz",kx=kx,ky=ky,sigma_grid=sigma_grid)
    plot_sigma_grid(kx, ky, sigma_grid, save_plots=True, figsize=(16, 4))