"""
Entry point for running scattering integral calculations.
Run from project root: python run.py
"""
import sys
from pathlib import Path

# Add src to path so imports work when run from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from input_states import gaussian_in_state
from S_matrix import  square_lattice,t,self_energy,alpha,create_self_energy_interpolator_numba,parallel_self_energy_grid,real_space_summation,k_space_summation
#from scattering_integrals import scattering_integral_vegas
#import time
import matplotlib.pyplot as plt

if __name__ == "__main__":

    '''
    # Compute self-energy over 10x10 grid with omega=1
    kx, ky, sigma_grid = parallel_self_energy_grid(n_points=50, omega=square_lattice.omega_e, n_jobs=12, lattice=square_lattice)
    np.savez("data/sigma_grid0f4a.npz",kx=kx,ky=ky,sigma_grid=sigma_grid)
    '''
    # Load from file (comment out if computing fresh)
    data = np.load("data/sigma_grid0f4a.npz")
    kx = data["kx"]
    ky = data["ky"]
    sigma_grid = data["sigma_grid"]
    sigma_func_period = create_self_energy_interpolator_numba(kx, ky, sigma_grid, lattice=square_lattice)
    collective_lamb_shift = self_energy(0,0,square_lattice.a,square_lattice.d,square_lattice.omega_e,alpha,square_lattice).real

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
    print(square_lattice.ge(np.array([0.4, 0.0, np.sqrt(2**2-0.4**2)]))**2/square_lattice.a**2* 2/np.sqrt(2**2-0.4**2))
 
    print(-self_energy(0.4,0,square_lattice.a,square_lattice.d,2,alpha,square_lattice).imag*2)


    print(real_space_summation(square_lattice.a,square_lattice.d,np.array([0.4, 0.0]),2))
    print(k_space_summation(square_lattice.a,square_lattice.d,np.array([0.4, 0.0]),2,alpha))
