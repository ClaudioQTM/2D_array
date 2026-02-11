"""
Entry point for running scattering integral calculations.
Run from project root: python run.py
"""
import sys
from pathlib import Path

# Add src to path so imports work when run from project root (must be before imports from src)
sys.path.insert(0, str(Path(__file__).parent / "src"))


import numpy as np
from input_states import gaussian_in_state
from smatrix import (
    alpha,
    create_self_energy_interpolator_numba,
    legs,
    self_energy,
    square_lattice,
    t,
    tau_matrix_element,
    parallel_tau_matrix_grid,
)
from scattering import scattering_integral_vegas
import time

from joblib import Parallel, delayed
import matplotlib.pyplot as plt

if __name__ == "__main__":

    '''
    # Compute self-energy over 10x10 grid with omega=1
    kx, ky, sigma_grid = parallel_self_energy_grid(n_points=50, omega=square_lattice.omega_e, n_jobs=8, lattice=square_lattice)
    np.savez("data/sigma_grid0f6a.npz",kx=kx,ky=ky,sigma_grid=sigma_grid)
    '''
    # Load from file (comment out if computing fresh)
    data = np.load("data/sigma_grid0f6a.npz")
    kx = data["kx"]
    ky = data["ky"]
    sigma_grid = data["sigma_grid"]
    sigma_func_period_numba = create_self_energy_interpolator_numba(kx, ky, sigma_grid, lattice=square_lattice)
    collective_lamb_shift = self_energy(0,0,square_lattice.a,square_lattice.d,square_lattice.omega_e,alpha).real

    '''
    _gaussian_in_state = gaussian_in_state(
        q0=np.array([0.0, 0.0, square_lattice.omega_e + collective_lamb_shift + 0.01]),
        l0=np.array([0.0, 0.0, square_lattice.omega_e + collective_lamb_shift + 0.01]),
        sigma=np.pi/(3*square_lattice.a),
    )
    start_time = time.time()
    result = scattering_integral_vegas(
        np.array([0, 0]),
        square_lattice.omega_e + collective_lamb_shift + 0.01,
        np.array([0, 0]),
        square_lattice.omega_e + collective_lamb_shift + 0.01,
        square_lattice,
        _gaussian_in_state,
        nitn1=3,
        nitn2=10,
        neval=5e4,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(result)
    '''

#    print(t(np.array([0, 0]),square_lattice.omega_e,square_lattice,sigma_func_period))
#    print(abs(t(np.array([0.0, 0.0]),square_lattice.omega_e+0.05,square_lattice,sigma_func_period)))
#    print(self_energy(0,0,square_lattice.a,square_lattice.d,square_lattice.omega_e,square_lattice.omega_e,1e-4))
    #plt.plot(np.linspace(-2,2,100), abs(sw_propagator(np.array([0, 0]),square_lattice.omega_e + collective_lamb_shift + np.linspace(-2,2,100),square_lattice)))
    #plt.plot(np.linspace(-2,2,100), abs(S_disconnected(np.array([0, 0]),square_lattice.omega_e + collective_lamb_shift,np.array([0, 0]),square_lattice.omega_e + collective_lamb_shift + np.linspace(-2,2,100),square_lattice)))
    #plt.show()
    #print(self_energy(0,0,square_lattice.a,square_lattice.d,square_lattice.omega_e,square_lattice.omega_e,alpha).imag*2+square_lattice.gamma)
    #omg = 2*(square_lattice.omega_e+collective_lamb_shift)
#    omg_grid = np.linspace(square_lattice.omega_e, (square_lattice.omega_e+collective_lamb_shift), 2)

#    print(abs(square_lattice.ge(np.array([0.3,0.2,np.sqrt(omg**2-0.2**2)])))**2/square_lattice.a**2* omg/np.sqrt(omg**2-0.2**2))

#    print(-2*(-omg**2  * real_space_summation(square_lattice.a,square_lattice.d,np.array([0., 0.2]),omg).imag - omg**3/(6*np.pi)))
#    #print(k_space_summation(square_lattice.a,square_lattice.d,np.array([0.3, 0.2]),1,alpha))
#    print(abs(t(np.array([kx,ky]),omg,square_lattice,sigma_func_period)))
#    print(abs(1- 1j/ square_lattice.a**2 * omg/np.sqrt(omg**2-kx**2-ky**2) * abs(square_lattice.ge(np.array([kx,ky,np.sqrt(omg**2-kx**2-ky**2)])))**2 / (omg - square_lattice.omega_e - self_energy(kx,ky,square_lattice.a,square_lattice.d,omg,alpha))))

#    print(abs(tau_matrix_element_polar(omg, np.array([kx,ky]), square_lattice, sigma_func_period_numba, n_jobs=4))/square_lattice.gamma)

#    print(2*abs(tau_matrix_element(omg, np.array([kx,ky]), square_lattice, sigma_func_period_numba))/square_lattice.gamma)

    qx_grid, qy_grid, tau_grid = parallel_tau_matrix_grid(n_points=50, E=2*(square_lattice.omega_e + collective_lamb_shift), n_jobs=8, lattice=square_lattice, sigma_func_period=sigma_func_period_numba)
    np.savez("data/tau_grid0f6a.npz",qx_grid=qx_grid,qy_grid=qy_grid,tau_grid=tau_grid)

    # 2D plots of |tau| and phase(tau)
    tau_abs = 2*np.abs(tau_grid)/float(square_lattice.gamma)
    tau_phase = np.angle(tau_grid)

    KX, KY = np.meshgrid(qx_grid, qy_grid, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axes[0].pcolormesh(KX, KY, tau_abs, shading="auto")

    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(KX, KY, tau_phase, shading="auto")

    fig.colorbar(im1, ax=axes[1])

    plt.show()
'''
    _gaussian_in_state = gaussian_in_state(
        q0=np.array([0.0, 0.0, square_lattice.omega_e + collective_lamb_shift + 0.01]),
        l0=np.array([0.0, 0.0, square_lattice.omega_e + collective_lamb_shift + 0.01]),
        sigma=np.pi / (6 * square_lattice.a),
    )
    start_time = time.time()
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
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(vegas_result)
'''

