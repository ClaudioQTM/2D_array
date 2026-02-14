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
    self_energy,
    square_lattice,
    connected_amplitude,
    create_tau_interpolator_numba,
)
#from scattering.api import scattering_integral_vegas


if __name__ == "__main__":

    
    # Load from file (comment out if computing fresh)
    sigma_data = np.load("data/sigma_grid0f4a.npz")
    kx = sigma_data["kx"]
    ky = sigma_data["ky"]
    sigma_grid = sigma_data["sigma_grid"]
    sigma_func_period_numba = create_self_energy_interpolator_numba(kx, ky, sigma_grid, lattice=square_lattice)
    collective_lamb_shift = self_energy(0,0,square_lattice.a,square_lattice.d,square_lattice.omega_e,alpha).real

    tau_data = np.load("data/tau_grid0f4a.npz")
    qx = tau_data["qx_grid"]
    qy = tau_data["qy_grid"]
    tau_grid = tau_data["tau_grid"]
    tau_matrix = create_tau_interpolator_numba(qx, qy, tau_grid, lattice=square_lattice)
   

#    print(t(np.array([0, 0]),square_lattice.omega_e,square_lattice,sigma_func_period))
#    print(abs(t(np.array([0.0, 0.0]),square_lattice.omega_e+0.05,square_lattice,sigma_func_period)))
#    print(self_energy(0,0,square_lattice.a,square_lattice.d,square_lattice.omega_e,square_lattice.omega_e,1e-4))
    #plt.plot(np.linspace(-2,2,100), abs(sw_propagator(np.array([0, 0]),square_lattice.omega_e + collective_lamb_shift + np.linspace(-2,2,100),square_lattice)))
    #plt.plot(np.linspace(-2,2,100), abs(S_disconnected(np.array([0, 0]),square_lattice.omega_e + collective_lamb_shift,np.array([0, 0]),square_lattice.omega_e + collective_lamb_shift + np.linspace(-2,2,100),square_lattice)))
    #plt.show()
    #print(self_energy(0,0,square_lattice.a,square_lattice.d,square_lattice.omega_e,square_lattice.omega_e,alpha).imag*2+square_lattice.gamma)
    #omg = 2*(square_lattice.omega_e+collective_lamb_shift)
#    omg_grid = np.linspace(square_lattice.omega_e, (square_lattice.omega_e+collective_lamb_shift), 2)


    omg = 100.0
    kx = 0
    ky = 0

#    print(abs(S_disconnected(np.array([kx,ky]),omg,np.array([kx,ky]),omg,square_lattice,sigma_func_period_numba)))
#    print(t(np.array([kx,ky]),omg,square_lattice))
#    print(abs(1- 1j/ square_lattice.a**2 * omg/np.sqrt(omg**2-kx**2-ky**2) * abs(square_lattice.ge(np.array([kx,ky,np.sqrt(omg**2-kx**2-ky**2)])))**2 / (omg - square_lattice.omega_e - self_energy(kx,ky,square_lattice.a,square_lattice.d,omg,alpha))))
#    print(1- 1j/ square_lattice.a**2 * omg/np.sqrt(omg**2-kx**2-ky**2) * abs(square_lattice.ge(np.array([kx,ky,np.sqrt(omg**2-kx**2-ky**2)])))**2 / (omg - square_lattice.omega_e - self_energy(kx,ky,square_lattice.a,square_lattice.d,omg,alpha)))
#    print(abs(tau_matrix_element_polar(omg, np.array([kx,ky]), square_lattice, sigma_func_period_numba, n_jobs=4))/square_lattice.gamma)

#    print(2*abs(tau_matrix_element(omg, np.array([kx,ky]), square_lattice, sigma_func_period_numba))/square_lattice.gamma)

    

    _gaussian_in_state = gaussian_in_state(
        q0=np.array([0.0, 0.0, square_lattice.omega_e+collective_lamb_shift]),
        l0=np.array([0.0, 0.0, square_lattice.omega_e+collective_lamb_shift]),
        sigma= np.pi / (13 * square_lattice.a),
    )
    '''
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
    '''
    psi_c = connected_amplitude(np.array([0, 0]),square_lattice.omega_e+collective_lamb_shift,
    np.array([0, 0]),square_lattice.omega_e+collective_lamb_shift,
    square_lattice,
    tau_matrix,
    _gaussian_in_state,
    sigma_func_period_numba,
    nitn1=5,nitn2=10,neval=5e4)
    print(psi_c)


