"""
Entry point for running scattering integral calculations.
Run from project root: python run.py
"""

import sys
from pathlib import Path

# Add src to path so imports work when run from project root (must be before imports from src)
sys.path.insert(0, str(Path(__file__).parent / "src"))


import numpy as np
from smatrix import (
    alpha,
    create_self_energy_interpolator_numba,
    self_energy,
    square_lattice,
)
# from scattering.api import scattering_integral_vegas


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
    print(collective_lamb_shift)
    print(2 * square_lattice.omega_e + collective_lamb_shift)
    print(square_lattice.q)
    print(square_lattice.omega_e)
    #    print(abs(S_disconnected(np.array([kx,ky]),omg,np.array([kx,ky]),omg,square_lattice,sigma_func_period_numba)))
    #    print(t(np.array([kx,ky]),omg,square_lattice))
    #    print(abs(1- 1j/ square_lattice.a**2 * omg/np.sqrt(omg**2-kx**2-ky**2) * abs(square_lattice.ge(np.array([kx,ky,np.sqrt(omg**2-kx**2-ky**2)])))**2 / (omg - square_lattice.omega_e - self_energy(kx,ky,square_lattice.a,square_lattice.d,omg,alpha))))
    #    print(1- 1j/ square_lattice.a**2 * omg/np.sqrt(omg**2-kx**2-ky**2) * abs(square_lattice.ge(np.array([kx,ky,np.sqrt(omg**2-kx**2-ky**2)])))**2 / (omg - square_lattice.omega_e - self_energy(kx,ky,square_lattice.a,square_lattice.d,omg,alpha)))
    #    print(abs(tau_matrix_element_polar(omg, np.array([kx,ky]), square_lattice, sigma_func_period_numba, n_jobs=4))/square_lattice.gamma)

    #    print(2*abs(tau_matrix_element(omg, np.array([kx,ky]), square_lattice, sigma_func_period_numba))/square_lattice.gamma)

    """
    vegas_result = scattering_integral_vegas(np.array([0, 0]),
        square_lattice.omega_e,
        np.array([0, 0]),
        square_lattice.omega_e,
        square_lattice,
        _gaussian_in_state,
        sigma_func_period_numba,
        nitn1=3,
        nitn2=10,
        neval=5e4,
    )
    """
    """
    out_kx_grid = np.linspace(0, square_lattice.q/2, 4)
    out_ky_grid = out_kx_grid
    out_Ek_grid = np.linspace(0,2*square_lattice.omega_e, 4)
    out_px_grid = out_kx_grid
    out_py_grid = out_ky_grid
    out_Ep_grid = out_Ek_grid
    out_points = [(kx, ky, Ek, px, py, Ep) for kx in out_kx_grid for ky in out_ky_grid for Ek in out_Ek_grid for px in out_px_grid for py in out_py_grid for Ep in out_Ep_grid]

    outstate = Parallel(n_jobs = 6)(delayed(connected_amplitude)(
        np.array([kx, ky]),
        Ek,
        np.array([px, py]),
        Ep,
        square_lattice,
        _gaussian_in_state,
        sigma_func_period_numba,
        nitn1=5,nitn2=10,neval=5e4) for kx, ky, Ek, px, py, Ep in out_points)
    outstate = np.array(outstate)
    print(outstate.shape)
    """
