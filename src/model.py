import numpy as np
import mpmath as mp

c = mp.mpf('1')
epsilon_0 = mp.mpf('1')
hbar = mp.mpf('1')
mu_0 = mp.mpf('1')


class EMField:
    """define properties of a 3D electromagnetic field"""
    def __init__(self,polar_vec):
        self.polar_vec = polar_vec

    @staticmethod
    def DispRel(k_xy, k_z):
        k_xy = np.asarray(k_xy)
        k_z = np.asarray(k_z)
        if k_xy.ndim == 1:
            return c * np.sqrt(k_z**2 + np.linalg.norm(k_xy)**2)
        if k_xy.ndim == 2:
            if k_xy.shape[1] == 2:
                k_xy_norm = k_xy
            elif k_xy.shape[0] == 2:
                k_xy_norm = k_xy.T
            else:
                raise ValueError("k_xy must have shape (n, 2) or (2, n).")
            n = k_xy_norm.shape[0]
            if k_z.ndim == 0:
                k_z_vec = np.full(n, float(k_z))
            else:
                k_z_vec = k_z.reshape(-1)
                if k_z_vec.shape[0] != n:
                    raise ValueError("k_z must have the same length as k_xy.")
            return float(c) * np.sqrt(k_z_vec**2 + np.sum(k_xy_norm**2, axis=1))
        raise ValueError("k_xy must be a 1D or 2D array.")

    @staticmethod
    def GreenTensor(k_xy,z,E):
        k_xy = np.asarray(k_xy)
        z_arr = np.asarray(z)
        if k_xy.ndim == 1:
            kz = np.sqrt(E**2 - np.linalg.norm(k_xy)**2)
            k = np.concatenate([k_xy, [np.sign(z) * kz]])
            Q = np.identity(3) - np.outer(k, k) / np.linalg.norm(k) ** 2
            return 1j / (2 * kz) * np.exp(1j * kz * abs(z)) * Q

        if k_xy.ndim == 2:
            if k_xy.shape[1] == 2:
                k_xy_norm = k_xy
            elif k_xy.shape[0] == 2:
                k_xy_norm = k_xy.T
            else:
                raise ValueError("k_xy must have shape (n, 2) or (2, n).")
            n = k_xy_norm.shape[0]
            if z_arr.ndim == 0:
                z_vec = np.full(n, float(z_arr))
            else:
                z_vec = z_arr.reshape(-1)
                if z_vec.shape[0] != n:
                    raise ValueError("z must have the same length as k_xy.")

            kz = np.sqrt(E**2 - np.sum(k_xy_norm**2, axis=1))
            sign_z = np.sign(z_vec)
            k = np.column_stack([k_xy_norm, sign_z * kz])
            k_norm = np.linalg.norm(k, axis=1)
            outer = k[:, :, None] * k[:, None, :]
            Q = np.identity(3)[None, :, :] - outer / (k_norm[:, None, None] ** 2)
            prefactor = 1j / (2 * kz) * np.exp(1j * kz * np.abs(z_vec))
            return prefactor[:, None, None] * Q

        raise ValueError("k_xy must be a 1D or 2D array.")

    @staticmethod
    def GreenTensor_r(r,E):
        r_norm = np.linalg.norm(r)
        r_norm = mp.mpf(r_norm)
        k = E/c
        A = 1+(mp.j*k*r_norm-1)/(k**2*r_norm**2)
        B = -1+(3-3*mp.j*k*r_norm)/(k**2*r_norm**2)

        I = np.identity(3)
        I = mp.matrix(I)
        rr = np.outer(r,r)/r_norm**2
        rr = mp.matrix(rr)

        return mp.exp(mp.j*k*r_norm)/(4*mp.pi*r_norm)*(A*I+B*rr)






class SquareLattice:
    """Define properties of a atomic square lattice."""

    def __init__(self, a, omega_e, dipole_vector, field,grid_cutoff=50):
        self.a = a
        self.omega_e = omega_e
        self.d = dipole_vector
        self.q = 2 * mp.pi / a
        self.field = field
        self.gamma = np.linalg.norm(self.d)**2*self.omega_e**3/(3*np.pi*epsilon_0*hbar*c**3)  # population decay rate
        # The center of this grid is not the first BZ. I defined this for the convenience of the summations.
        J1, J2 = np.meshgrid(np.arange(-grid_cutoff, grid_cutoff + 1), np.arange(-grid_cutoff, grid_cutoff + 1))
        self.lattice_grid = (float(self.q) * J1, float(self.q) * J2)
    def g(self, u, v, k_xy, k_z):
        if v not in (-1, 1):
            raise ValueError(f"v must be +1 or -1, got {v}")
        k_xy = np.asarray(k_xy)
        k_z = np.asarray(k_z)
        disp = self.field.DispRel(k_xy, v * k_z)
     
        coupling = np.vdot(self.field.polar_vec[u], self.d)
        if k_xy.ndim == 1:
            return np.sqrt(float(disp)/ (2*float(epsilon_0)))  * coupling
        return np.sqrt(disp/ (2*float(epsilon_0)))  * coupling

    def ge(self, k):
        k = np.asarray(k)
        if k.ndim == 1:
            k_xy = k[:2]
            k_z = k[2]
            value = sum(sum(abs(self.g(u, v, k_xy, k_z))**2 for u in (0, 1)) for v in (-1, 1))
            return np.sqrt(value)
        if k.ndim == 2:
            if k.shape[1] == 3:
                k_norm = k
            elif k.shape[0] == 3:
                k_norm = k.T
            else:
                raise ValueError("k must have shape (n, 3) or (3, n).")
            k_xy = k_norm[:, :2]
            k_z = k_norm[:, 2]
            total = np.zeros(k_norm.shape[0], dtype=np.float64)
            for v in (-1, 1):
                for u in (0, 1):
                    total += abs(self.g(u, v, k_xy, k_z)) ** 2
            return np.sqrt(total)
        raise ValueError("k must be a 1D or 2D array.")



def real_space_summation(a, d, k_xy, omega):

    def summand(m, n, k_xy, omega):
        k = omega / c
        # lattice vector
        r_mn_norm = a * mp.sqrt(m**2 + n**2)
        d_norm = np.linalg.norm(d)
        d_norm = mp.mpf(d_norm)
        # compute two inner products between the dipole vector and the lattice vector
        drrd = mp.fabs(m * a * d[0] + n * a * d[1])**2
        A = 1 + (mp.j * k * r_mn_norm - 1) / (k**2 * r_mn_norm**2)
        B = -1 + (3 - 3 * mp.j * k * r_mn_norm)/(k**2 * r_mn_norm**2)
        dGd = mp.exp(mp.j * k * r_mn_norm)/(4 * mp.pi * r_mn_norm) * (A * d_norm**2 + B * drrd / r_mn_norm**2)
        power = -mp.j * (m*k_xy[0] * a+ n * k_xy[1] * a)
        return mp.exp(power) * dGd

    rsp_sum = mp.nsum(lambda m, n: 0 if (m == 0 and n == 0) else  summand(m,n,k_xy,omega), [-mp.inf, mp.inf], [-mp.inf, mp.inf])
    return rsp_sum

def k_space_summation(a, d, k_xy, omega, alpha):
    k = omega/c
    lam = 2*mp.pi/omega
    q = 2*mp.pi/a

    def kG_squared(m, n):
        kGx = k_xy[0] + m * q
        kGy = k_xy[1] + n * q
        
        return kGx**2 + kGy**2

    def summand(m,n):
        kG2 = kG_squared(m, n)
        k_z = mp.sqrt(k**2 - kG2)

        return mp.exp(-alpha*kG2)*(1-kG2/(2*k**2))/k_z

    
    G00 = 1/(16*mp.sqrt(mp.pi*alpha))*mp.exp(-alpha*k**2)*(1-1/(2*k**2*alpha))
    G00 = np.linalg.norm(d)**2*G00

    RegLatticeSum = mp.j/(2*a**2)*mp.nsum(lambda m,n: summand(m,n), [-mp.inf, mp.inf], [-mp.inf, mp.inf])
    RegLatticeSum = np.linalg.norm(d)**2*RegLatticeSum
    
    FinitePart = RegLatticeSum - G00 -np.linalg.norm(d)**2*mp.j/(3*lam)

    return FinitePart



def self_energy(k_x,k_y,a, d, omega_e,omega, alpha):
    k_xy = np.array([k_x, k_y])
    prefactor = mu_0*omega_e**2/hbar
    return complex(prefactor*k_space_summation(a, d, k_xy, omega, alpha))

