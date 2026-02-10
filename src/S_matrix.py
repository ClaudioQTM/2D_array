from model import *
import numpy as np
from scipy.interpolate import RectBivariateSpline
from joblib import Parallel, delayed
from numba import njit
from scipy import integrate


alpha = 1e-4
polar_vec1 = np.array([1,1j,0])/np.sqrt(2)
polar_vec2 = np.array([1,-1j,0])/np.sqrt(2)
field = EMField()

square_lattice = SquareLattice(a=0.4*2*np.pi, omega_e=1, dipole_vector=np.array([1,1j,0])/np.sqrt(2), field=field)
collective_lamb_shift = self_energy(0,0,square_lattice.a,square_lattice.d,square_lattice.omega_e,alpha).real

def parallel_self_energy_grid(n_points, omega, n_jobs,lattice):

    k_max = float(lattice.q/2)
    # Create k-space grid
    kx_grid = np.linspace(0, k_max, n_points)
    ky_grid = np.linspace(0, k_max, n_points)
    
    # Create list of all (i, j, kx, ky) pairs for parallel computation
    k_points = [(i, j, kx, ky) for i, kx in enumerate(kx_grid) for j, ky in enumerate(ky_grid)]
    
    
    # Parallel computation using joblib
    results = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(self_energy)(kx, ky, lattice.a, lattice.d, omega, alpha)
        for i, j, kx, ky in k_points
    )
    
    # Reshape results into 2D grid
    self_energy_grid = np.array(results, dtype=complex).reshape(n_points, n_points)
    
    return kx_grid, ky_grid, self_energy_grid


def create_self_energy_interpolator(kx_grid, ky_grid, sigma_grid,kx=3, ky=3):
    """
    kx, ky : int
        Degree of the spline (1=linear, 3=cubic). Default is cubic.
    
    Returns:
    --------
    sigma_interp : callable
        Function that takes (kx, ky) and returns interpolated self-energy.
        Can accept scalars or arrays.
    """
    # Interpolate real and imaginary parts separately
    real_spline = RectBivariateSpline(kx_grid, ky_grid, sigma_grid.real, kx=kx, ky=ky)
    imag_spline = RectBivariateSpline(kx_grid, ky_grid, sigma_grid.imag, kx=kx, ky=ky)
    
    def sigma_interp(kx_val, ky_val, grid=False):
        
        real_part = real_spline(kx_val, ky_val, grid=grid)
        imag_part = imag_spline(kx_val, ky_val, grid=grid)
        return real_part + 1j * imag_part
    

    def sigma_func_period(kx, ky, lattice):
        kx_bz = (kx+lattice.q/2) % lattice.q-lattice.q/2
        # map kx_bz into the first quadrant
        kx_bz = abs(kx_bz)
        ky_bz = (ky+lattice.q/2) % lattice.q-lattice.q/2
        # map kx_bz into the first quadrant
        ky_bz = abs(ky_bz)
        return sigma_interp(kx_bz, ky_bz)
    return sigma_func_period


def create_self_energy_interpolator_numba(kx_grid, ky_grid, sigma_grid, lattice):
    """
    Create a Numba-compatible periodic self-energy interpolator using bilinear interpolation.
    
    Parameters:
    -----------
    kx_grid, ky_grid : array-like
        1D arrays of grid points (assumed uniform spacing)
    sigma_grid : complex array
        2D array of self-energy values on the grid
    lattice : SquareLattice
        Lattice object (only lattice.q is used)
    
    Returns:
    --------
    sigma_func_period_numba : callable
        Numba-compiled function that takes (kx, ky) and returns interpolated self-energy.
        Note: This version does NOT take lattice as argument (it's baked in).
    """
    # Convert grids to float64 arrays (Numba compatible)
    kx_arr = np.ascontiguousarray(kx_grid, dtype=np.float64)
    ky_arr = np.ascontiguousarray(ky_grid, dtype=np.float64)
    real_grid = np.ascontiguousarray(sigma_grid.real, dtype=np.float64)
    imag_grid = np.ascontiguousarray(sigma_grid.imag, dtype=np.float64)
    
    # Extract lattice parameter as plain float
    q = float(lattice.q)
    
    # Grid parameters
    kx_min, kx_max = kx_arr[0], kx_arr[-1]
    ky_min, ky_max = ky_arr[0], ky_arr[-1]
    nx, ny = len(kx_arr), len(ky_arr)
    dx = kx_arr[1] - kx_arr[0]
    dy = ky_arr[1] - ky_arr[0]
    
    @njit(cache=True)
    def bilinear_interp(x, y, x_min, y_min, dx, dy, nx, ny, z_grid):
        """Bilinear interpolation on a uniform grid."""
        # Clamp to grid bounds
        x = max(x_min, min(x, x_min + (nx - 1) * dx - 1e-10))
        y = max(y_min, min(y, y_min + (ny - 1) * dy - 1e-10))
        
        # Find cell indices
        ix = int((x - x_min) / dx)
        iy = int((y - y_min) / dy)
        
        ix = max(0, min(ix, nx - 2))
        iy = max(0, min(iy, ny - 2))
        
        # Local coordinates in [0, 1]
        tx = (x - (x_min + ix * dx)) / dx
        ty = (y - (y_min + iy * dy)) / dy
        
        # Bilinear interpolation
        z00 = z_grid[ix, iy]
        z10 = z_grid[ix + 1, iy]
        z01 = z_grid[ix, iy + 1]
        z11 = z_grid[ix + 1, iy + 1]
        
        return (z00 * (1 - tx) * (1 - ty) +
                z10 * tx * (1 - ty) +
                z01 * (1 - tx) * ty +
                z11 * tx * ty)
    
    @njit(cache=True)
    def sigma_func_period_numba(kx, ky):
        """Numba-compiled periodic self-energy function."""
        # Map to first BZ centered at origin
        kx_bz = (kx + q / 2) % q - q / 2
        # Map to first quadrant (use symmetry)
        kx_bz = abs(kx_bz)
        
        ky_bz = (ky + q / 2) % q - q / 2
        ky_bz = abs(ky_bz)
        
        # Interpolate real and imaginary parts
        real_part = bilinear_interp(kx_bz, ky_bz, kx_min, ky_min, dx, dy, nx, ny, real_grid)
        imag_part = bilinear_interp(kx_bz, ky_bz, kx_min, ky_min, dx, dy, nx, ny, imag_grid)
        
        return real_part + 1j * imag_part
    
    return sigma_func_period_numba





def tau_matrix_element(E, Q, lattice,sigma_func_period):
    """Compute tau matrix element via 2D integration over the Brillouin zone."""
    Qx, Qy = Q
    bound = np.pi / lattice.a
    
    def integrand(qx, qy):
        sigma1 = sigma_func_period(qx, qy)
        sigma2 = sigma_func_period(Qx - qx, Qy - qy)
        return 1 / (E - 2*lattice.omega_e - sigma1 - sigma2)
    
    # Use nquad instead of dblquad to allow setting subdivision limit
    integration_opts = {'limit': 50}
    
    re_integral, _ = integrate.nquad(
        lambda qx, qy: integrand(qx, qy).real,
        [[-bound, bound], [-bound, bound]],
        opts=integration_opts
    )
    
    im_integral, _ = integrate.nquad(
        lambda qx, qy: integrand(qx, qy).imag,
        [[-bound, bound], [-bound, bound]],
        opts=integration_opts
    )
    
    
    Pi = (lattice.a / (2*np.pi))**2 * (re_integral + 1j * im_integral)
    return -1 / Pi




'''
# Plot Delta/gamma vs a/lambda
a_values = np.linspace(0.15*2*np.pi, 0.99*2*np.pi,20)

# Parallel computation using joblib
self_energy_values = Parallel(n_jobs=-1, verbose=10)(
    delayed(self_energy)(0, 0, aa, lattice.d, lattice.omega_e, lattice.omega_e, alpha)
    for aa in a_values
)

# Convert to numpy array for easier handling
self_energy_array = np.array(self_energy_values)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(a_values/(2*np.pi), -self_energy_array.real/float(lattice.gamma), label='Real part', linewidth=2)
#plt.plot(a_values, self_energy_array.imag, label='Imaginary part', linewidth=2)
plt.xlabel('$a/\\lambda$', fontsize=12)
plt.ylabel('Self-energy/$\\gamma$', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''






def t(k_para, E, lattice):   
    k = coord_convert(k_para, E)
    if k.ndim == 1:
        kz = k[2]
    else:
        kz = k[:, 2]
    prefactor = -1j * np.linalg.norm(lattice.d)**2 / lattice.a**2 * E / kz
    numerator = abs(lattice.ge(k))**2
    denominator =  E - lattice.omega_e - self_energy(k[0],k[1],lattice.a,lattice.d,E,alpha)
    fraction_part = numerator / denominator
    return 1 + prefactor * fraction_part

def S_disconnected(q_para,Eq,l_para,El,lattice):
    direct_term = t(q_para,Eq,lattice) * t(l_para,El,lattice)
    return direct_term


def coord_convert(k_para, E):
    """
    Convert 2D parallel momentum [kx, ky] to 3D Cartesian [kx, ky, kz] using
    dispersion |k| = E/c, so kz = sqrt((E/c)^2 - kx^2 - ky^2).

    Accepts single vectors or batches (e.g. from np.linspace grids).
    """
    k_arr = np.asarray(k_para, dtype=np.float64)
    c_val = float(c)

    # Single momentum: k_para shape (2,) -> return (3,)
    if k_arr.ndim == 1:
        kz = np.sqrt((E / c_val)**2 - np.linalg.norm(k_arr)**2)
        E_arr = np.asarray(E, dtype=np.float64)
        if kz.ndim == 0:
            return np.concatenate([k_arr, [kz]])
        else:
            k_para_arr = np.broadcast_to(k_arr, (kz.shape[0], 2))
            return np.column_stack([k_para_arr, kz])
            

    # Batch: k_para shape (n, 2) or (2, n) -> return (n, 3)
    if k_arr.ndim == 2:
        if k_arr.shape[1] == 2:
            k_para_norm = k_arr
        elif k_arr.shape[0] == 2:
            k_para_norm = k_arr.T
        else:
            raise ValueError("k_para must have shape (n, 2) or (2, n).")

        # E can be scalar (same energy for all) or length-n array
        E_arr = np.asarray(E, dtype=np.float64)
        if E_arr.ndim == 0:
            E_vec = np.full(k_para_norm.shape[0], float(E_arr))
        else:
            E_vec = E_arr.reshape(-1)
            if E_vec.shape[0] != k_para_norm.shape[0]:
                raise ValueError("E must have the same length as k_para.")

        kz = np.sqrt((E_vec / c_val)**2 - np.sum(k_para_norm**2, axis=1))
        return np.column_stack([k_para_norm, kz])

    raise ValueError("k_para must be a 1D or 2D array.")



def sw_propagator(k_para, E, lattice,sigma_func_period):
    k_arr = np.asarray(k_para, dtype=np.float64)
    E_arr = np.asarray(E, dtype=np.float64)

    if k_arr.ndim == 1:
        sigma_val = sigma_func_period(k_arr[0], k_arr[1])
        denom = E_arr - lattice.omega_e - sigma_val
        return 1 / denom

    if k_arr.ndim == 2:
        if k_arr.shape[1] == 2:
            k_para_norm = k_arr
        elif k_arr.shape[0] == 2:
            k_para_norm = k_arr.T
        else:
            raise ValueError("k_para must have shape (n, 2) or (2, n).")

        if E_arr.ndim == 0:
            E_vec = np.full(k_para_norm.shape[0], float(E_arr))
        else:
            E_vec = E_arr.reshape(-1)
            if E_vec.shape[0] != k_para_norm.shape[0]:
                raise ValueError("E must have the same length as k_para.")

        sigma_vals = np.array(
            [sigma_func_period(kx, ky) for kx, ky in k_para_norm],
            dtype=complex,
        )
        denom = E_vec - lattice.omega_e - sigma_vals
        return 1 / denom

    raise ValueError("k_para must be a 1D or 2D array.")



def legs(q_para, Eq, l_para, El, lattice, sigma_func_period, direction="in"):
    q = coord_convert(q_para, Eq)
    l = coord_convert(l_para, El)
    if direction == 'in':
        coupling = lattice.ge(q) * lattice.ge(l)
    elif direction == 'out':
        coupling = np.conj(lattice.ge(q)) * np.conj(lattice.ge(l))
    else:
        raise ValueError(f"Invalid direction: {direction}")
    return coupling * sw_propagator(q_para, Eq, lattice,sigma_func_period) * sw_propagator(l_para, El, lattice,sigma_func_period)




def tau_matrix_element_polar(E, Q, lattice, sigma_func_period, n_jobs=4):
    """Compute tau matrix element via 2D integration over the square BZ in polar coords.
    
    Square region: |kx| <= pi/a, |ky| <= pi/a (edge length 2*pi/a), centered at (0,0).
    In polar coordinates (k_abs, theta), the radial cutoff depends on theta:
        k_abs <= min((pi/a)/|cos(theta)|, (pi/a)/|sin(theta)|).
    
    Parameters:
    -----------
    n_jobs : int
        Number of parallel jobs (default 4 for the 4 integrals).
    """
    # Check condition: only execute if lattice.omega_e < np.pi/(lattice.a)
    if not (lattice.omega_e < np.pi / lattice.a):
        raise ValueError(f"Function requires lattice.omega_e < np.pi/(lattice.a). "
                         f"Got omega_e={lattice.omega_e:.6e}, pi/a={np.pi/lattice.a:.6e}")
    
    Qx, Qy = Q
    bound = np.pi / lattice.a
    k_LC = lattice.omega_e / float(c)
    
    def integrand(k_abs, theta):
        sigma1 = sigma_func_period(k_abs*np.cos(theta), k_abs*np.sin(theta))
        sigma2 = sigma_func_period(Qx - k_abs*np.cos(theta), Qy - k_abs*np.sin(theta))
        return 1 / (E - 2*lattice.omega_e - sigma1 - sigma2)

    def k_abs_range(theta):
        """Integration range outside the light cone within 1st BZ."""
        r_x = bound / abs(np.cos(theta)) if abs(np.cos(theta)) > 1e-12 else np.inf
        r_y = bound / abs(np.sin(theta)) if abs(np.sin(theta)) > 1e-12 else np.inf
        return [k_LC, min(r_x, r_y)]

    # Define the four integrand functions
    def integrand_real(k_abs, theta):
        return (integrand(k_abs, theta) * k_abs).real
    
    
    def integrand_imag(k_abs, theta):
        return (integrand(k_abs, theta) * k_abs).imag
    


    def run_nquad(func, ranges, opts):
        """Helper for parallel nquad calls."""
        result, _ = integrate.nquad(func, ranges, opts=opts)
        return result

    integration_opts = {'limit': 150,'epsabs': 1.49e-04,'epsrel': 1.49e-04}
    
    # Run all four integrals in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_nquad)(func, ranges, integration_opts)
        for func, ranges in [
            (integrand_real, [[0.0, k_LC], [0.0, 2*np.pi]]),
            (integrand_real, [k_abs_range, [0.0, 2*np.pi]]),
            (integrand_imag, [[0.0, k_LC], [0.0, 2*np.pi]]),
            (integrand_imag, [k_abs_range, [0.0, 2*np.pi]]),
        ]
    )
    
    re_integral_LC, re_integral_rest, im_integral_LC, im_integral_rest = results
    
    Pi = (lattice.a / (2*np.pi))**2 * ((re_integral_LC + re_integral_rest) + 1j * (im_integral_LC + im_integral_rest))
    return -1 / Pi
