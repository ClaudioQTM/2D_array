import numpy as np
import warnings
import scipy.integrate as integrate
from input_states import gaussian_in_state
from model import *
from S_matrix import *
import vegas
try:
    from numba import njit
except Exception:  # pragma: no cover - fallback when numba is unavailable
    njit = None
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





def _build_numba_integrand_kernel(E, lattice, in_state, sigma_func_period):
    """
    Try to build a Numba kernel for the hot integrand path.

    This collapses several Python-level pieces used by `integrand` into one
    `njit`-compiled kernel:
      - kinematics (q, l from D, Dpx, Dpy and COM_K, G, H)
      - light-cone / indicator check
      - Jacobian factors
      - `legs`-like coupling using a simplified, inlined version of `ge` and
        `sw_propagator`
      - Gaussian two-photon input state for `gaussian_in_state`

    If anything about the inputs makes this unsafe or non-Numba-friendly
    (e.g. different in_state type, weird shapes, or sigma_func not usable
    in nopython mode), we return None and the caller falls back to the
    original Python integrand instead.
    """
    if njit is None:
        return None
    if not isinstance(in_state, gaussian_in_state):
        return None

    try:
        # Grab parameters from the specific gaussian_in_state instance and
        # lattice, and normalise them into simple ndarray / float scalars
        # that Numba can close over.
        q0 = np.asarray(in_state.q0, dtype=np.float64)
        l0 = np.asarray(in_state.l0, dtype=np.float64)
        sigma_arr = np.asarray(in_state.sigma, dtype=np.float64)
        if q0.shape != (3,) or l0.shape != (3,):
            return None
        if sigma_arr.ndim == 0:
            sigma_vec = np.full(3, float(sigma_arr), dtype=np.float64)
        else:
            sigma_flat = sigma_arr.reshape(-1)
            if sigma_flat.size != 3:
                return None
            sigma_vec = sigma_flat.astype(np.float64)
        if np.any(sigma_vec <= 0):
            return None

        # Lattice / model parameters that are treated as constants inside
        # the kernel.
        d_vec = np.asarray(lattice.d, dtype=np.complex128).reshape(3)
        omega_e = float(lattice.omega_e)
        c_val = float(c)
        eps0 = float(epsilon_0)
        e_half = 0.5 * float(E)
        e_total = float(E)
        sqrt2_inv = 1.0 / np.sqrt(2.0)
        norm_pref = (2.0 * np.pi) ** (-1.5) * (np.prod(sigma_vec) ** (-1.0))
    except Exception:
        return None

    @njit(cache=True)
    def _ge_single(kx, ky, kz, d):
        """
        Lightweight stand‑in for `square_lattice.ge` for a single k-vector.

        It reconstructs the two polarisation vectors and their couplings to
        the dipole `d`, then returns sqrt(sum_u,v |g(u,v)|^2).
        """
        kxy2 = kx * kx + ky * ky
        kxy_norm = np.sqrt(kxy2)
        if kxy_norm < 1e-14:
            return 0.0

        total = 0.0
        for v in (-1.0, 1.0):
            kz_signed = v * kz
            k_norm = np.sqrt(kxy2 + kz_signed * kz_signed)
            if k_norm < 1e-14:
                continue

            e1x = ky / kxy_norm
            e1y = -kx / kxy_norm
            e1z = 0.0

            denom = k_norm * kxy_norm
            e2x = -(-kz_signed * kx) / denom
            e2y = -(-kz_signed * ky) / denom
            e2z = -(kxy2) / denom

            p0x = sqrt2_inv * (e1x + 1j * e2x)
            p0y = sqrt2_inv * (e1y + 1j * e2y)
            p0z = sqrt2_inv * (e1z + 1j * e2z)
            coup0 = (np.conjugate(p0x) * d[0] +
                     np.conjugate(p0y) * d[1] +
                     np.conjugate(p0z) * d[2])

            p1x = sqrt2_inv * (e1x - 1j * e2x)
            p1y = sqrt2_inv * (e1y - 1j * e2y)
            p1z = sqrt2_inv * (e1z - 1j * e2z)
            coup1 = (np.conjugate(p1x) * d[0] +
                     np.conjugate(p1y) * d[1] +
                     np.conjugate(p1z) * d[2])

            disp = c_val * k_norm
            pref = np.sqrt(disp / (2.0 * eps0))
            g0 = pref * coup0
            g1 = pref * coup1
            total += (g0.real * g0.real + g0.imag * g0.imag)
            total += (g1.real * g1.real + g1.imag * g1.imag)

        return np.sqrt(total)

    @njit(cache=True)
    def _gaussian_state_value(qx, qy, Eq, lx, ly, El):
        """
        Evaluate the product of two 3D Gaussians in k-space corresponding
        to the `gaussian_in_state` object used at construction time.

        This is specialised to the 2D + dispersion -> 3D branch that is
        relevant for the scattering integrals (no generic shape handling).
        """
        qz2 = (Eq / c_val) * (Eq / c_val) - (qx * qx + qy * qy)
        lz2 = (El / c_val) * (El / c_val) - (lx * lx + ly * ly)
        if qz2 <= 0.0 or lz2 <= 0.0:
            return 0.0

        qz = np.sqrt(qz2)
        lz = np.sqrt(lz2)

        dq0 = (qx - q0[0]) / sigma_vec[0]
        dq1 = (qy - q0[1]) / sigma_vec[1]
        dq2 = (qz - q0[2]) / sigma_vec[2]
        dl0 = (lx - l0[0]) / sigma_vec[0]
        dl1 = (ly - l0[1]) / sigma_vec[1]
        dl2 = (lz - l0[2]) / sigma_vec[2]

        exp_arg = -0.25 * (dq0 * dq0 + dq1 * dq1 + dq2 * dq2 + dl0 * dl0 + dl1 * dl1 + dl2 * dl2)
        return norm_pref * np.exp(exp_arg)

    @njit(cache=True)
    def _integrand_kernel(D, Dpx, Dpy, COM_K, G, H):
        """
        Core Numba-accelerated integrand.

        Inputs:
          - D, Dpx, Dpy: batches of scalar integration variables
          - COM_K, G, H: 2D vectors defining kinematics for this (J, G, H)
        For each sample it:
          1. Builds q and l from COM_K, Dp and G/H.
          2. Applies the light-cone / support indicator.
          3. Computes Jacobian factors.
          4. Evaluates a simplified `legs` (ge * ge * propagators).
          5. Multiplies by the Gaussian in-state value.
        """
        n = D.shape[0]
        out = np.zeros(n, dtype=np.complex128)

        for idx in range(n):
            dpx = Dpx[idx]
            dpy = Dpy[idx]
            d = D[idx]

            qx = COM_K[0] + 0.5 * dpx + G[0]
            qy = COM_K[1] + 0.5 * dpy + G[1]
            lx = COM_K[0] - 0.5 * dpx + H[0]
            ly = COM_K[1] - 0.5 * dpy + H[1]

            Eq = e_half + d
            El = e_half - d

            q_norm2 = qx * qx + qy * qy
            l_norm2 = lx * lx + ly * ly

            indicator_arg = (
                e_total
                - np.sqrt((qx + G[0]) * (qx + G[0]) + (qy + G[1]) * (qy + G[1]))
                - np.sqrt((lx + H[0]) * (lx + H[0]) + (ly + H[1]) * (ly + H[1]))
            )
            if indicator_arg < 0.0:
                continue

            qz2_jac = Eq * Eq - q_norm2
            lz2_jac = El * El - l_norm2
            if qz2_jac <= 0.0 or lz2_jac <= 0.0:
                continue

            qz2 = (Eq / c_val) * (Eq / c_val) - q_norm2
            lz2 = (El / c_val) * (El / c_val) - l_norm2
            if qz2 <= 0.0 or lz2 <= 0.0:
                continue

            qz = np.sqrt(qz2)
            lz = np.sqrt(lz2)

            ge_q = _ge_single(qx, qy, qz, d_vec)
            ge_l = _ge_single(lx, ly, lz, d_vec)
            if ge_q == 0.0 or ge_l == 0.0:
                continue

            sigma_q = sigma_func_period(qx, qy)
            sigma_l = sigma_func_period(lx, ly)
            sw_q = 1.0 / (Eq - omega_e - sigma_q)
            sw_l = 1.0 / (El - omega_e - sigma_l)
            legs_val = ge_q * ge_l * sw_q * sw_l

            in_state_val = _gaussian_state_value(qx, qy, Eq, lx, ly, El)
            if in_state_val == 0.0:
                continue

            jacobian = (Eq / np.sqrt(qz2_jac)) * (El / np.sqrt(lz2_jac))
            out[idx] = jacobian * legs_val * in_state_val

        return out

    try:
        _ = _integrand_kernel(
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        )
    except Exception:
        return None

    return _integrand_kernel


def _make_integrand_and_bounds(E, lattice, in_state, sigma_func_period):
    """Factory function to create integrand and D_bounds functions."""
    integrand_kernel_numba = _build_numba_integrand_kernel(E, lattice, in_state, sigma_func_period)

    def integrand(D, Dpx, Dpy, COM_K, G, H):
        """Integrand with D as innermost variable so its bounds can depend on Dpx, Dpy.

        Supports scalar inputs or batched inputs (arrays) for D, Dpx, Dpy.
        """
        if integrand_kernel_numba is not None:
            is_scalar = np.ndim(D) == 0 and np.ndim(Dpx) == 0 and np.ndim(Dpy) == 0
            D_arr = np.atleast_1d(np.asarray(D, dtype=np.float64))
            Dpx_arr = np.atleast_1d(np.asarray(Dpx, dtype=np.float64))
            Dpy_arr = np.atleast_1d(np.asarray(Dpy, dtype=np.float64))
            COM_K_arr = np.asarray(COM_K, dtype=np.float64)
            G_arr = np.asarray(G, dtype=np.float64)
            H_arr = np.asarray(H, dtype=np.float64)
            out = integrand_kernel_numba(D_arr, Dpx_arr, Dpy_arr, COM_K_arr, G_arr, H_arr)
            if is_scalar:
                return out[0]
            return out

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


def _build_vegas_transform_kernel(E):
    """
    Build a Numba kernel for the Vegas variable transform: [0,1]^3 -> (Dpx, Dpy, D).

    Vegas samples uniformly in the unit cube. For each sample `u=(u1,u2,u3)` we:
      - Map u1,u2 to the Brillouin-zone constrained ranges of Dpx, Dpy
      - Compute the *D* bounds (D_lo, D_hi) implied by those Dpx/Dpy via
        |q_para| and |l_para| constraints
      - Map u3 into D in [D_lo, D_hi]
      - Return the Jacobian for the full transform so the integrand can be
        evaluated as an average over the unit cube.

    We keep this transform separate from the physics integrand so it can be
    reused by both the real and imaginary Vegas batch-integrands, and because
    it is pure scalar arithmetic that Numba optimizes very well.

    Returns
    -------
    callable or None
        If Numba is available, returns a compiled function:
          (Dpx, Dpy, D, D_lo, D_hi, jacobian, valid) = f(xbatch, ...)
        Otherwise returns None (caller falls back to NumPy/Python code).
    """
    if njit is None:
        return None

    E_half = 0.5 * float(E)

    @njit(cache=True)
    def _transform(xbatch, Dpx_lo, Dpx_hi, Dpy_lo, Dpy_hi, COM_K, G, H):
        # Allocate outputs once per batch call (Vegas provides xbatch with shape (n, 3)).
        n = xbatch.shape[0]
        Dpx = np.empty(n, dtype=np.float64)
        Dpy = np.empty(n, dtype=np.float64)
        D = np.empty(n, dtype=np.float64)
        D_lo = np.empty(n, dtype=np.float64)
        D_hi = np.empty(n, dtype=np.float64)
        jacobian = np.empty(n, dtype=np.float64)
        valid = np.empty(n, dtype=np.bool_)

        # Constant part of the Jacobian from (u1,u2) -> (Dpx,Dpy).
        vol_Dpx = Dpx_hi - Dpx_lo
        vol_Dpy = Dpy_hi - Dpy_lo
        prefactor = vol_Dpx * vol_Dpy

        for idx in range(n):
            # Unpack the unit-cube coordinates for this sample.
            u1 = xbatch[idx, 0]
            u2 = xbatch[idx, 1]
            u3 = xbatch[idx, 2]

            # Linear maps from [0,1] to [lo,hi] for Dpx and Dpy.
            dpx = Dpx_lo + u1 * vol_Dpx
            dpy = Dpy_lo + u2 * vol_Dpy
            Dpx[idx] = dpx
            Dpy[idx] = dpy

            # Compute the transverse momenta q_para and l_para (only their norms are needed here).
            qx = COM_K[0] + 0.5 * dpx + G[0]
            qy = COM_K[1] + 0.5 * dpy + G[1]
            lx = COM_K[0] - 0.5 * dpx + H[0]
            ly = COM_K[1] - 0.5 * dpy + H[1]

            # D bounds for this (Dpx,Dpy) choice:
            #   D_lo = |q_para| - E/2
            #   D_hi = E/2 - |l_para|
            dlo = np.sqrt(qx * qx + qy * qy) - E_half
            dhi = E_half - np.sqrt(lx * lx + ly * ly)
            D_lo[idx] = dlo
            D_hi[idx] = dhi

            # Map u3 into D in [D_lo, D_hi]. Note: if dhi <= dlo this interval is invalid.
            D[idx] = dlo + u3 * (dhi - dlo)

            # Full Jacobian: (Dpx range) * (Dpy range) * (D range for this sample).
            jacobian[idx] = prefactor * (dhi - dlo)
            valid[idx] = dhi > dlo

        return Dpx, Dpy, D, D_lo, D_hi, jacobian, valid

    return _transform

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
    vegas_transform_kernel = _build_vegas_transform_kernel(E)
    
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
            COM_K64 = np.asarray(COM_K, dtype=np.float64)
            G64 = np.asarray(G, dtype=np.float64)
            H64 = np.asarray(H, dtype=np.float64)
            
            # Create Vegas batch integrands that map [0,1]^3 to actual domain
            @vegas.lbatchintegrand
            def vegas_integrand_real(xbatch):
                if vegas_transform_kernel is not None:
                    Dpx, Dpy, D, _, _, jacobian, valid = vegas_transform_kernel(
                        np.asarray(xbatch, dtype=np.float64),
                        Dpx_lo,
                        Dpx_hi,
                        Dpy_lo,
                        Dpy_hi,
                        COM_K64,
                        G64,
                        H64,
                    )
                else:
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
                if vegas_transform_kernel is not None:
                    Dpx, Dpy, D, _, _, jacobian, valid = vegas_transform_kernel(
                        np.asarray(xbatch, dtype=np.float64),
                        Dpx_lo,
                        Dpx_hi,
                        Dpy_lo,
                        Dpy_hi,
                        COM_K64,
                        G64,
                        H64,
                    )
                else:
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

