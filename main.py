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
from S_matrix import sigma_func_period, square_lattice, collective_lamb_shift,coord_convert,t
from scattering_integrals import scattering_integral_vegas
import time
#import matplotlib.pyplot as plt

if __name__ == "__main__":
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
    print(abs(t(np.array([0, 0]),square_lattice.omega_e,square_lattice)))
    print(sigma_func_period(0,0))
    print(square_lattice.a)
#    print(square_lattice.ge(np.array([0, 0, 1])))
    print(square_lattice.g(u=0,v=1,k_xy=np.array([0, 0]),k_z=1))
    print(square_lattice.g(u=0,v=-1,k_xy=np.array([0, 0]),k_z=1))
    print(square_lattice.g(u=1,v=1,k_xy=np.array([0, 0]),k_z=1))
    print(square_lattice.g(u=1,v=-1,k_xy=np.array([0, 0]),k_z=1))
    print(square_lattice.ge(np.array([0, 0, 1])))
#    print(self_energy(0,0,square_lattice.a,square_lattice.d,square_lattice.omega_e,square_lattice.omega_e,1e-4))
    print(coord_convert(np.array([0, 0]),square_lattice.omega_e))
    #plt.plot(np.linspace(-2,2,100), abs(sw_propagator(np.array([0, 0]),square_lattice.omega_e + collective_lamb_shift + np.linspace(-2,2,100),square_lattice)))
    #plt.plot(np.linspace(-2,2,100), abs(S_disconnected(np.array([0, 0]),square_lattice.omega_e + collective_lamb_shift,np.array([0, 0]),square_lattice.omega_e + collective_lamb_shift + np.linspace(-2,2,100),square_lattice)))
    #plt.show()




