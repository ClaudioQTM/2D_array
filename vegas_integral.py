import numpy as np
from smatrix import t_reg
from eigenstate_solving.eigen_eq_integrand import BZ_proj
from model import self_energy, square_lattice
from model.defaults import alpha
from smatrix import create_self_energy_interpolator_numba, tau_matrix_element
from W_state.I_term import I_term_integ_vegas_batch
from smatrix import L
from joblib import Parallel, delayed

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

    E = 2 * (square_lattice.omega_e + collective_lamb_shift)+ 0.25
    Q = np.array([0, 0])
    E1 = E / 2 + 1

    k_para = np.array([10, 10])
    tau = t_reg(k_para, E1, square_lattice, sigma_func_period_numba) * t_reg(BZ_proj(Q-k_para,square_lattice), E-E1, square_lattice, sigma_func_period_numba)

    #MEQ =  tau_matrix_element(E, Q, square_lattice, sigma_func_period_numba)
    #print("MEQ:",MEQ)
    n_test = 6
    eta_0 = 1e-3
    eta_list = [0.5**n * eta_0 for n in range(n_test)]
    neval_0 = int(5e6)
    neval_list = [2**n * neval_0 for n in range(n_test) ]
    regulator_neval_pairs = zip(eta_list,neval_list)

    I_term_list = Parallel(n_jobs=6)(delayed(I_term_integ_vegas_batch)(E,Q, eta, tau,10,neval,square_lattice,sigma_func_period_numba) for eta,neval in regulator_neval_pairs)
    print("I_term:",I_term_list)
"""
    
    
    C_term = 2 * L(k_para, E1, square_lattice, sigma_func_period_numba, "in",BM=True) * L(BZ_proj(Q-k_para,square_lattice), E-E1, square_lattice, sigma_func_period_numba, "in",BM=True) / (1 + 1j/2* (2*np.pi)**3 /square_lattice.a**4 * MEQ * I_term)
    
    print("C_term:", C_term)


    coefficient = -1j/2 * (2*np.pi)**3 / square_lattice.a**4 * MEQ * C_term 
    print(coefficient)
"""



