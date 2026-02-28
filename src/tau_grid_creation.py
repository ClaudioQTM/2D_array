"""
Change the name of data file to run the code for different sigma grid.
"""

import numpy as np
import matplotlib.pyplot as plt
from smatrix import (
    alpha,
    create_self_energy_interpolator_numba,
    self_energy,
    square_lattice,
    parallel_tau_matrix_grid,
)


if __name__ == "__main__":
    # Load from file (comment out if computing fresh)
    sigma_data_path = "data/sigma_grid0f4a.npz"
    data = np.load(sigma_data_path)
    kx = data["kx"]
    ky = data["ky"]
    sigma_grid = data["sigma_grid"]
    sigma_func_period_numba = create_self_energy_interpolator_numba(
        kx, ky, sigma_grid, lattice=square_lattice
    )
    collective_lamb_shift = self_energy(
        0, 0, square_lattice.a, square_lattice.d, square_lattice.omega_e, alpha
    ).real

    qx_grid, qy_grid, tau_grid = parallel_tau_matrix_grid(
        n_points=64,
        E=2 * (square_lattice.omega_e + collective_lamb_shift),
        n_jobs=6,
        lattice=square_lattice,
        sigma_func_period=sigma_func_period_numba,
    )
    tau_data_path = sigma_data_path.replace("sigma", "tau")
    np.savez(tau_data_path, qx_grid=qx_grid, qy_grid=qy_grid, tau_grid=tau_grid)

    # 2D plots of |tau| and phase(tau)
    tau_abs = 2 * np.abs(tau_grid) / float(square_lattice.gamma)
    tau_phase = np.angle(tau_grid)

    KX, KY = np.meshgrid(qx_grid, qy_grid, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axes[0].pcolormesh(KX, KY, tau_abs, shading="auto")

    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(KX, KY, tau_phase, shading="auto")

    fig.colorbar(im1, ax=axes[1])

    plt.show()
