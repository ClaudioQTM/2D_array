import sys
from pathlib import Path

# Add src to path so imports work when run from project root (must be before imports from src)
sys.path.insert(0, str(Path(__file__).parent / "src"))


import numpy as np

# from smatrix.amplitudes import S_disconnected
from eigenstate_solving.eigen_eq_integrand import BZ_proj
from smatrix import (
    create_self_energy_interpolator_numba,
    square_lattice,
)

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
    collective_lamb_shift = self_energy(
        0, 0, square_lattice.a, square_lattice.d, square_lattice.omega_e, alpha
    ).real
    Q = np.array([0.0, 0.0], dtype=np.float64)

    # Coarse scan settings.
    e_values = np.linspace(2*(square_lattice.omega_e + collective_lamb_shift)+1, 2*(square_lattice.omega_e + collective_lamb_shift)+80, 17, endpoint=False)
    phi_values = np.linspace(0.0, 2 * np.pi, 18, endpoint=False)
    neval_coarse = int(5e6)
    n_jobs = 6

    best_metric = np.inf
    best_E = None
    best_phi = None
    best_value = None

    for E in e_values:
        tau_factor = tau_matrix_element(
            E, Q, square_lattice, sigma_func_period_numba
        )
        results = Parallel(n_jobs=n_jobs)(
            delayed(eigen_eq_itr_batch)(
                Q,
                E,
                square_lattice,
                sigma_func_period_numba,
                np.exp(1j * phi),
                neval=neval_coarse,
                tau_matrix_calculation=False,
            )
            for phi in phi_values
        )
        values = tau_factor * np.asarray(results, dtype=np.complex128)
        distances = np.abs(values - 1.0)
        idx = int(np.argmin(distances))

        if distances[idx] < best_metric:
            best_metric = float(distances[idx])
            best_E = float(E)
            best_phi = float(phi_values[idx])
            best_value = values[idx]

        print(
            f"E={E:.6f} | best_phi={phi_values[idx]:.6f} | "
            f"value={values[idx]} | |value-1|={distances[idx]:.6e}"
        )

    print("\n=== Coarse scan global best ===")
    print(f"E_best = {best_E:.6f}")
    print(f"phi_best = {best_phi:.6f}")
    print(f"tau_factor * result = {best_value}")
    print(f"|tau_factor * result - 1| = {best_metric:.6e}")
    """
    """
    plot_integrand1(2*(square_lattice.omega_e + collective_lamb_shift)+60, np.array([0,0]), np.array([0,0]), np.array([0,0]), 0.2, sigma_func_period_numba, square_lattice, np.exp(1j*np.pi))
    """
    print(square_lattice.q)
    print(BZ_proj(np.array([499, 499]), square_lattice))
