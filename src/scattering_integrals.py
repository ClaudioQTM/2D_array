import numpy as np
import warnings
import scipy.integrate as integrate
from input_states import gaussian_in_state
from model import *
from S_matrix import *
import vegas
# calculate the disconnected scattering amplitude

def disconnected_scattering_integral(q_para, Eq, l_para, El, in_state, lattice, sigma_func_period):
    return S_disconnected(q_para, Eq, l_para, El, lattice, sigma_func_period) * in_state(q_para, Eq, l_para, El)

# calculate the connected scattering amplitude

# define the filters for the triple lattice sums in front of the momentum integrals

def J_filter(k_para,p_para,lattice):
    COM_K_out = (k_para+p_para)/2
    
    # Extract meshgrid arrays
    J_grid_x, J_grid_y = lattice.lattice_grid
    J_x_flat = J_grid_x.ravel()
    J_y_flat = J_grid_y.ravel()
    
    # Filter: keep points where COM_K + J/2 is within the first BZ
    Jx_shifted = COM_K_out[0] + J_x_flat / 2
    Jy_shifted = COM_K_out[1] + J_y_flat / 2
    Jx_mask = (-float(lattice.q)/2 < Jx_shifted) & (Jx_shifted < float(lattice.q)/2)
    Jy_mask = (-float(lattice.q)/2 < Jy_shifted) & (Jy_shifted < float(lattice.q)/2)
    # Combine masks to keep (J_x, J_y) pairs where both conditions are satisfied
    J_mask = Jx_mask & Jy_mask
    J_x_filtered = J_x_flat[J_mask]
    J_y_filtered = J_y_flat[J_mask]
    return J_x_filtered, J_y_filtered


def GH_filter(COM_K,E,lattice=square_lattice):

    G_grid_x, G_grid_y = lattice.lattice_grid
    G_x_flat = G_grid_x.ravel()
    G_y_flat = G_grid_y.ravel()
    max_Delta_norm = np.sqrt((np.pi/lattice.a-abs(COM_K[0]))**2+(np.pi/lattice.a-abs(COM_K[1]))**2)

    first_mask = np.linalg.norm(COM_K + np.column_stack([G_x_flat, G_y_flat]), axis=1) <= E/c + np.sqrt(2)/2 * max_Delta_norm
    G_x_filtered = G_x_flat[first_mask]
    G_y_filtered = G_y_flat[first_mask]

    # Create all (G, H) pairs using meshgrid
    G_x_mesh, H_x_mesh = np.meshgrid(G_x_filtered, G_x_filtered, indexing='ij')
    G_y_mesh, H_y_mesh = np.meshgrid(G_y_filtered, G_y_filtered, indexing='ij')
    
    # Flatten to get all pairs
    G_x_pairs = G_x_mesh.ravel()
    G_y_pairs = G_y_mesh.ravel()
    H_x_pairs = H_x_mesh.ravel()
    H_y_pairs = H_y_mesh.ravel()

    # Apply second mask to all (G, H) pairs
    second_mask = np.linalg.norm(2*COM_K + np.column_stack([G_x_pairs + H_x_pairs, G_y_pairs + H_y_pairs]), axis=1) <= E/c
    G_x_filtered = G_x_pairs[second_mask]
    G_y_filtered = G_y_pairs[second_mask]
    H_x_filtered = H_x_pairs[second_mask]
    H_y_filtered = H_y_pairs[second_mask]

    return G_x_filtered, G_y_filtered, H_x_filtered, H_y_filtered





def _make_integrand_and_bounds(E, lattice, in_state, sigma_func_period):
    """Factory function to create integrand and D_bounds functions."""
    def integrand(D, Dpx, Dpy, COM_K, G, H):
        """Integrand with D as innermost variable so its bounds can depend on Dpx, Dpy.

        Supports scalar inputs or batched inputs (arrays) for D, Dpx, Dpy.
        """
        Dpx_arr = np.asarray(Dpx)
        Dpy_arr = np.asarray(Dpy)
        D_arr = np.asarray(D)

        # Build Dp with rightmost dimension 2 for vectorized computation
        Dp = np.stack([Dpx_arr, Dpy_arr], axis=-1)
        q_para = COM_K + Dp / 2 + G
        l_para = COM_K - Dp / 2 + H

        Eq = E / 2 + D_arr
        El = E / 2 - D_arr

        # Check if inside light cone; vectorized for arrays
        q_norm = np.linalg.norm(q_para, axis=-1)
        l_norm = np.linalg.norm(l_para, axis=-1)
        indicator = np.heaviside(E - np.linalg.norm(q_para + G, axis=-1) - np.linalg.norm(l_para + H, axis=-1), 0.5)

        # Skip per element: only evaluate for valid points
        mask = indicator != 0

        if np.ndim(mask) == 0:
            if not mask:
                return 0.0 + 0.0j

            qz = np.sqrt(Eq**2 - q_norm**2)
            lz = np.sqrt(El**2 - l_norm**2)
            jacobian = (Eq / qz) * (El / lz)
            return jacobian * legs(q_para, Eq, l_para, El, lattice, sigma_func_period, direction="in") * in_state(q_para, Eq, l_para, El)

        value = np.zeros_like(indicator, dtype=np.complex128)
        if np.any(mask):
            Eq_m = Eq[mask]
            El_m = El[mask]
            q_norm_m = q_norm[mask]
            l_norm_m = l_norm[mask]
            q_para_m = q_para[mask]
            l_para_m = l_para[mask]

            qz = np.sqrt(Eq_m**2 - q_norm_m**2)
            lz = np.sqrt(El_m**2 - l_norm_m**2)
            jacobian = (Eq_m / qz) * (El_m / lz)
            value[mask] = jacobian * legs(q_para_m, Eq_m, l_para_m, El_m, lattice, sigma_func_period, direction="in") * in_state(q_para_m, Eq_m, l_para_m, El_m)

        return value


    def D_bounds(Dpx, Dpy, COM_K, G, H):
        """D bounds depend on Dpx, Dpy through q_para, l_para.
        
        Works with both scalar and array inputs:
        - Scalar: Dpx, Dpy are floats -> returns [D_min, D_max] (scalars)
        - Array: Dpx, Dpy are arrays -> returns [D_min_arr, D_max_arr] (arrays)
        """
        # Convert to arrays to handle both scalar and array cases
        Dpx_arr = np.atleast_1d(Dpx)
        Dpy_arr = np.atleast_1d(Dpy)
        is_scalar = np.ndim(Dpx) == 0 and np.ndim(Dpy) == 0
        
        # Stack for vectorized computation: shape (n_samples, 2) or (1, 2) for scalar
        Dp = np.stack([Dpx_arr, Dpy_arr], axis=-1)
        
        # Broadcasting: COM_K and G/H are (2,), Dp is (..., 2)
        q_para = COM_K + Dp / 2 + G
        l_para = COM_K - Dp / 2 + H
        
        # Compute norms along last axis
        D_min = np.linalg.norm(q_para, axis=-1) - E / 2
        D_max = E / 2 - np.linalg.norm(l_para, axis=-1)
        
        # Return scalar if input was scalar, otherwise return arrays
        if is_scalar:
            return [D_min.item(), D_max.item()]
        else:
            return [D_min, D_max]
    
    return integrand, D_bounds

def _integrate_nquad(J_x, J_y, k_para, p_para, E, bound, lattice, integrand, D_bounds):
    """Perform integration using scipy.integrate.nquad (quadrature)."""
    total = 0.0 + 0.0j
    
    for j in range(len(J_x)):
        J = np.array([J_x[j], J_y[j]])
        COM_K = 0.5 * (k_para + p_para + J)
        G_x, G_y, H_x, H_y = GH_filter(COM_K, E, lattice)
        
        # Bounds for Dpx, Dpy (fixed, based on BZ constraints)
        Dpx_bounds = [abs(COM_K[0]) - bound, bound - abs(COM_K[0])]
        Dpy_bounds = [abs(COM_K[1]) - bound, bound - abs(COM_K[1])]
        
        for i in range(len(G_x)):
            G = np.array([G_x[i], G_y[i]])
            H = np.array([H_x[i], H_y[i]])
            
            # Integrate real and imaginary parts separately
            # nquad with D as innermost (first arg), so D bounds can depend on Dpx, Dpy
            result_real, _ = integrate.nquad(
                lambda D, Dpx, Dpy: integrand(D, Dpx, Dpy, COM_K, G, H).real,
                [
                    lambda Dpx, Dpy: D_bounds(Dpx, Dpy, COM_K, G, H),  # D bounds (innermost)
                    Dpx_bounds,  # Dpx bounds
                    Dpy_bounds,  # Dpy bounds (outermost)
                ],opts={'epsabs': 1e-3, 'epsrel': 1e-3, 'limit': 30}
            )
            
            result_imag, _ = integrate.nquad(
                lambda D, Dpx, Dpy: integrand(D, Dpx, Dpy, COM_K, G, H).imag,
                [
                    lambda Dpx, Dpy: D_bounds(Dpx, Dpy, COM_K, G, H),  # D bounds (innermost)
                    Dpx_bounds,  # Dpx bounds
                    Dpy_bounds,  # Dpy bounds (outermost)
                ],opts={'epsabs': 1e-3, 'epsrel': 1e-3, 'limit': 30}
            )
            
            total += result_real + 1j * result_imag
    
    return total




def _integrate_qmc(J_x, J_y, k_para, p_para, E, bound, lattice, integrand, D_bounds, m=13, seed=None):
    """Perform integration using Quasi-Monte Carlo (Sobol sequence).
    
    Parameters
    ----------
    m : int
        Sobol sequence uses 2^m samples (default: 13, i.e., 8192 samples)
    """
    from scipy.stats import qmc
    
    n_samples = 2**m  # Power of 2 for optimal Sobol balance properties
    total = 0.0 + 0.0j
    
    for j in range(len(J_x)):
        J = np.array([J_x[j], J_y[j]])
        COM_K = 0.5 * (k_para + p_para + J)
        G_x, G_y, H_x, H_y = GH_filter(COM_K, E, lattice)
        
        # Bounds for Dpx, Dpy (fixed, based on BZ constraints)
        Dpx_lo, Dpx_hi = abs(COM_K[0]) - bound, bound - abs(COM_K[0])
        Dpy_lo, Dpy_hi = abs(COM_K[1]) - bound, bound - abs(COM_K[1])
        
        for i in range(len(G_x)):
            G = np.array([G_x[i], G_y[i]])
            H = np.array([H_x[i], H_y[i]])
            
            # Generate Sobol sequence in [0,1]^3 with 2^m samples
            sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
            samples = sampler.random_base2(m)  # shape: (2^m, 3)
            
            # Vectorized transformation from [0,1] to actual bounds
            u1, u2, u3 = samples[:, 0], samples[:, 1], samples[:, 2]
            Dpx_arr = Dpx_lo + u1 * (Dpx_hi - Dpx_lo)
            Dpy_arr = Dpy_lo + u2 * (Dpy_hi - Dpy_lo)
            
            # Vectorized D_bounds calculation
            Dp_arr = np.stack([Dpx_arr, Dpy_arr], axis=1)  # shape: (n_samples, 2)
            q_para_arr = COM_K + Dp_arr / 2 + G  # broadcasting COM_K and G
            l_para_arr = COM_K - Dp_arr / 2 + H
            D_lo_arr = np.linalg.norm(q_para_arr, axis=1) - E / 2
            D_hi_arr = E / 2 - np.linalg.norm(l_para_arr, axis=1)
            
            # Filter valid samples
            valid_mask = D_hi_arr > D_lo_arr
            if not np.any(valid_mask):
                continue  # No valid samples
            
            # Transform D only for valid samples
            D_arr = D_lo_arr[valid_mask] + u3[valid_mask] * (D_hi_arr[valid_mask] - D_lo_arr[valid_mask])
            Dpx_valid = Dpx_arr[valid_mask]
            Dpy_valid = Dpy_arr[valid_mask]
            
            # Vectorized Jacobian
            jacobian_arr = (Dpx_hi - Dpx_lo) * (Dpy_hi - Dpy_lo) * (D_hi_arr[valid_mask] - D_lo_arr[valid_mask])
            
            # Evaluate integrand for all valid samples
            integral_sum = 0.0 + 0.0j
            for idx in range(len(D_arr)):
                f_val = integrand(D_arr[idx], Dpx_valid[idx], Dpy_valid[idx], COM_K, G, H)
                integral_sum += f_val * jacobian_arr[idx]
            
            # QMC estimate: average * volume (but volume is already in jacobian)
            total += integral_sum / n_samples
    
    return total


def _integrate_vegas(J_x, J_y, k_para, p_para, E, bound, lattice, integrand, D_bounds,
                     nitn1, nitn2, neval):
    """Perform integration using Vegas adaptive Monte Carlo.
    
    Parameters
    ----------
    nitn1 : int
        Number of iterations for initial Vegas adaptation
    nitn2 : int
        Number of iterations for second Vegas adaptation 
    neval : int
        Number of integrand evaluations per iteration 
    """
    import vegas
    
    total = 0.0 + 0.0j
    
    for j in range(len(J_x)):
        J = np.array([J_x[j], J_y[j]])
        COM_K = 0.5 * (k_para + p_para + J)
        G_x, G_y, H_x, H_y = GH_filter(COM_K, E, lattice)
        
        if len(G_x) == 0:
            continue
        
        # Bounds for Dpx, Dpy (fixed, based on BZ constraints)
        Dpx_lo, Dpx_hi = abs(COM_K[0]) - bound, bound - abs(COM_K[0])
        Dpy_lo, Dpy_hi = abs(COM_K[1]) - bound, bound - abs(COM_K[1])
        
        for i in range(len(G_x)):
            G = np.array([G_x[i], G_y[i]])
            H = np.array([H_x[i], H_y[i]])
            
            # Create Vegas batch integrands that map [0,1]^3 to actual domain
            @vegas.lbatchintegrand
            def vegas_integrand_real(xbatch):
                u1 = xbatch[:, 0]
                u2 = xbatch[:, 1]
                u3 = xbatch[:, 2]

                # Transform from [0,1] to actual bounds
                Dpx = Dpx_lo + u1 * (Dpx_hi - Dpx_lo)
                Dpy = Dpy_lo + u2 * (Dpy_hi - Dpy_lo)

                # D bounds depend on Dpx, Dpy
                D_lo, D_hi = D_bounds(Dpx, Dpy, COM_K, G, H)
                valid = D_hi > D_lo

                D = D_lo + u3 * (D_hi - D_lo)
                jacobian = (Dpx_hi - Dpx_lo) * (Dpy_hi - Dpy_lo) * (D_hi - D_lo)

                f_val = integrand(D, Dpx, Dpy, COM_K, G, H)
                result = f_val.real * jacobian
                return np.where(valid, result, 0.0)

            @vegas.lbatchintegrand
            def vegas_integrand_imag(xbatch):
                u1 = xbatch[:, 0]
                u2 = xbatch[:, 1]
                u3 = xbatch[:, 2]

                Dpx = Dpx_lo + u1 * (Dpx_hi - Dpx_lo)
                Dpy = Dpy_lo + u2 * (Dpy_hi - Dpy_lo)

                D_lo, D_hi = D_bounds(Dpx, Dpy, COM_K, G, H)
                valid = D_hi > D_lo

                D = D_lo + u3 * (D_hi - D_lo)
                jacobian = (Dpx_hi - Dpx_lo) * (Dpy_hi - Dpy_lo) * (D_hi - D_lo)

                f_val = integrand(D, Dpx, Dpy, COM_K, G, H)
                result = f_val.imag * jacobian
                return np.where(valid, result, 0.0)
            
            # Vegas integration over [0,1]^3
            vegas_integ_re = vegas.Integrator([[0, 1], [0, 1], [0, 1]])
            
            # Integrate real part
            vegas_integ_re(vegas_integrand_real, nitn=nitn1, neval=neval)
            result_real = vegas_integ_re(vegas_integrand_real, nitn=nitn2, neval=neval)
            real_iters = nitn1 + nitn2
            while result_real.Q <= 0.1 and real_iters <= 20:
                result_real = vegas_integ_re(vegas_integrand_real, nitn=nitn2, neval=neval)
                real_iters += nitn2
            if result_real.Q <= 0.1:
                warnings.warn(
                    f"VEGAS real-part Q stayed <= 0.1 after {real_iters} iterations (Q={result_real.Q}).",
                    RuntimeWarning,
                )
            
            # Reset and integrate imaginary part
            vegas_integ_im = vegas.Integrator([[0, 1], [0, 1], [0, 1]])
            vegas_integ_im(vegas_integrand_imag, nitn=nitn1, neval=neval)
            result_imag = vegas_integ_im(vegas_integrand_imag, nitn=nitn2, neval=neval)
            imag_iters = 1
            while result_imag.Q <= 0.1 and imag_iters <= 20:
                result_imag = vegas_integ_im(vegas_integrand_imag, nitn=nitn2, neval=neval)
                imag_iters += nitn2
            if result_imag.Q <= 0.1:
                warnings.warn(
                    f"VEGAS imag-part Q stayed <= 0.1 after {imag_iters} iterations (Q={result_imag.Q}).",
                    RuntimeWarning,
                )
            total += result_real.mean + 1j * result_imag.mean
    
    return total


def scattering_integral_nquad(k_para, Ek, p_para, Ep, lattice, in_state, sigma_func_period):
    """Compute scattering integral using quadrature (nquad)."""
    E = Ek + Ep
    bound = np.pi / lattice.a
    
    # Get filtered J values
    J_x, J_y = J_filter(k_para, p_para, lattice)
    
    # Create integrand and bounds functions
    integrand, D_bounds = _make_integrand_and_bounds(E, lattice, in_state, sigma_func_period)
    
    return _integrate_nquad(J_x, J_y, k_para, p_para, E, bound, lattice, integrand, D_bounds)


def scattering_integral_qmc(k_para, Ek, p_para, Ep, lattice, in_state, sigma_func_period, m=13, seed=None):
    """Compute scattering integral using Quasi-Monte Carlo (Sobol sequence).
    
    Parameters
    ----------
    k_para, Ek, p_para, Ep : array, float
        Incoming photon momenta and energies
    lattice : SquareLattice
        Lattice object
    in_state : callable
        Input state function
    m : int
        Sobol sequence uses 2^m samples per (J, G, H) combination.
        Default: 13 (8192 samples). Common values: 10 (1024), 12 (4096), 14 (16384)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    complex
        The scattering integral value
    """
    E = Ek + Ep
    bound = np.pi / lattice.a
    
    # Get filtered J values
    J_x, J_y = J_filter(k_para, Ek, p_para, Ep, lattice)
    
    # Create integrand and bounds functions
    integrand, D_bounds = _make_integrand_and_bounds(E, lattice, in_state, sigma_func_period)
    
    return _integrate_qmc(J_x, J_y, k_para, p_para, E, bound, lattice, integrand, D_bounds, m, seed)


def scattering_integral_vegas(k_para, Ek, p_para, Ep, lattice, in_state, sigma_func_period, nitn1=3, nitn2=10, neval=5e4):
    """Compute scattering integral using Vegas adaptive Monte Carlo.
    
    Parameters
    ----------
    k_para, Ek, p_para, Ep : array, float
        Incoming photon momenta and energies
    lattice : SquareLattice
        Lattice object
    in_state : callable
        Input state function
    nitn1 : int
        Number of iterations for initial Vegas adaptation (default: 3)
    nitn2 : int
        Number of iterations for second Vegas adaptation (default: 10)
    neval : int
        Number of integrand evaluations per iteration (default: 50000)
    
    Returns
    -------
    complex
        The scattering integral value
    """
    E = Ek + Ep
    bound = np.pi / lattice.a
    
    # Get filtered J values
    J_x, J_y = J_filter(k_para, p_para, lattice)
    
    # Create integrand and bounds functions
    integrand, D_bounds = _make_integrand_and_bounds(E, lattice, in_state, sigma_func_period)

    return _integrate_vegas(
        J_x,
        J_y,
        k_para,
        p_para,
        E,
        bound,
        lattice,
        integrand,
        D_bounds,
        nitn1,
        nitn2,
        neval,
    )

