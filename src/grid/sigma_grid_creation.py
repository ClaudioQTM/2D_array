import numpy as np
from joblib import Parallel, delayed

from smatrix import square_lattice
from model import self_energy
from model.defaults import alpha
from plots.plot_self_energy import plot_sigma_grid


def parallel_self_energy_grid(
    n_points, omega, n_jobs, lattice, dim, omega_start=None, omega_end=None,omega_points = None
):
    k_max = float(lattice.q / 2)
    kx_grid = np.linspace(0, k_max, n_points)
    ky_grid = np.linspace(0, k_max, n_points)

    k_points = [(kx, ky) for kx in kx_grid for ky in ky_grid]
    if dim == 3:
        if omega_start is None or omega_end is None or omega_points is None:
            raise ValueError(
                "omega_start, omega_end, omega_points are required when dim == 3"
            )

  
        omega_grid = np.linspace(
            float(omega_start),
            float(omega_end),
            omega_points,
        )
        results = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(self_energy)(kx, ky, lattice.a, lattice.d, omega_val, alpha)
            for (kx, ky) in k_points
            for omega_val in omega_grid
        )
        self_energy_grid = np.array(results, dtype=complex).reshape(
            n_points, n_points, omega_points
        )
        return kx_grid, ky_grid, omega_grid, self_energy_grid
    elif dim == 2:
        results = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(self_energy)(kx, ky, lattice.a, lattice.d, omega, alpha)
            for (kx, ky) in k_points
        )
        self_energy_grid = np.array(results, dtype=complex).reshape(n_points, n_points)
        return kx_grid, ky_grid, self_energy_grid
    else:
        raise ValueError(f"Invalid dimension: {dim}")


if __name__ == "__main__":
    kx, ky, omega_grid, sigma_grid = parallel_self_energy_grid(
        n_points=30,
        omega=None,
        n_jobs=6,
        lattice=square_lattice,
        dim=3,
        omega_start=square_lattice.omega_e,
        omega_end=2*square_lattice.omega_e,
        omega_points=15
    )
    np.savez("data/sigma_grid0f1a_3D.npz", kx=kx, ky=ky, sigma_grid=sigma_grid)
    plot_sigma_grid(kx, ky, sigma_grid, save_plots=False, figsize=(16, 4))