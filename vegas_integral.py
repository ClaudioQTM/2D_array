from joblib import Parallel, delayed
import numpy as np

from model import self_energy, square_lattice
from model.defaults import alpha
from smatrix import create_self_energy_interpolator_numba, tau_matrix_element
from W_state.I_term import I_term_integ_vegas_batch

# import matplotlib.pyplot as plt
# from smatrix.amplitudes import S_disconnected

if __name__ == "__main__":
    # Load from file (comment out if computing fresh)
    sigma_data = np.load("data/sigma_grid0f1a.npz")
    kx = sigma_data["kx"]
    ky = sigma_data["ky"]
    sigma_grid = sigma_data["sigma_grid"]
    sigma_func_period_numba = create_self_energy_interpolator_numba(
        kx, ky, sigma_grid, lattice=square_lattice
    )
    collective_lamb_shift = self_energy(
        0, 0, square_lattice.a, square_lattice.d, square_lattice.omega_e, alpha
    ).real


    Q = np.array([0, 0])
    E = 2 * (square_lattice.omega_e + collective_lamb_shift) - 5
    eta = 0.001
    I_term = Parallel(n_jobs=6)(
        delayed(I_term_integ_vegas_batch)(
            E,
            Q,
            eta,
            np.exp(1j * phi),
            10,
            int(1e3),
            square_lattice,
            sigma_func_period_numba,
        )
        for phi in np.linspace(0, 2 * np.pi, 12)
    )
    I_term = np.asarray(I_term, dtype=np.complex128)
    I_term = tau_matrix_element(E, Q, square_lattice, sigma_func_period_numba) * I_term

    results = (-4j * np.pi**3 / square_lattice.a**4) * I_term
    print(results)
