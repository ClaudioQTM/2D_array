from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is on sys.path so we can import `smatrix`, `input_states`, etc.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smatrix import square_lattice, create_self_energy_interpolator_numba
from input_states import gaussian_in_state
import numpy as np
import vegas
import time

if __name__ == "__main__":
    q0 = np.array([0, 0, square_lattice.omega_e])
    l0 = np.array([0, 0, square_lattice.omega_e])
    sigma = np.pi / (6 * square_lattice.a)
    cut_off = 4 * sigma
    q_up = q0 + cut_off
    q_low = q0 - cut_off
    l_up = q_up
    l_low = q_low

    # Instantiate the Gaussian input state once and reuse it in the integrands.
    _gaussian_in_state = gaussian_in_state(q0=q0, l0=l0, sigma=sigma)
    # Load from file (comment out if computing fresh)
    data = np.load("data/sigma_grid0f4a.npz")
    kx = data["kx"]
    ky = data["ky"]
    sigma_grid = data["sigma_grid"]
    sigma_func_period = create_self_energy_interpolator_numba(kx, ky, sigma_grid, lattice=square_lattice)

    def test_integrand(x):
        """Integrand for Vegas: x = [qx, qy, qz, lx, ly, lz]."""
        qx, qy, qz, lx, ly, lz = x[0], x[1], x[2], x[3], x[4], x[5]
        q = np.array([qx, qy, qz], dtype=float)
        l = np.array([lx, ly, lz], dtype=float)

        # Energies from full 3D momenta (c = 1 in this model).
        Eq = np.linalg.norm(q)
        El = np.linalg.norm(l)

        # Parallel components [kx, ky] used by gaussian_in_state.
        q_para = q[:2]
        l_para = l[:2]

        val = t(q_para, Eq, square_lattice, sigma_func_period)*_gaussian_in_state(q_para, Eq, l_para, El)
        return val**2


    @vegas.rbatchintegrand
    def test_integrand2(x):
        """Vectorised version: x has shape (6,) or (n, 6) with [qx, qy, qz, lx, ly, lz]."""
        x = np.asarray(x)
        is_scalar = x.ndim == 1
        if is_scalar:
            x = x.reshape(1, -1)
        else:
            if x.shape[0] == 6 and x.shape[1] != 6:
                x = x.T
            elif x.shape[1] != 6:
                raise ValueError(f"Expected shape (n, 6) or (6, n); got {x.shape}.")

        qx, qy, qz, lx, ly, lz = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        Eq = np.sqrt(qz**2 + qx**2 + qy**2)
        El = np.sqrt(lz**2 + lx**2 + ly**2)
        q_para = np.stack([qx, qy], axis=1)
        l_para = np.stack([lx, ly], axis=1)

        val = _gaussian_in_state(q_para, Eq, l_para, El)
        val = val**2
        if is_scalar:
            return val[0]
        return val



    # 6D integration bounds: [qx, qy, qz, lx, ly, lz]
    bounds = [
        [q_low[0], q_up[0]], [q_low[1], q_up[1]], [q_low[2], q_up[2]],
        [l_low[0], l_up[0]], [l_low[1], l_up[1]], [l_low[2], l_up[2]],
    ]
    t0 = time.perf_counter()
    norm_integ = vegas.Integrator(bounds)
    vegas.Integrator(bounds)(test_integrand2, nitn=5, neval=5e4)
    result = vegas.Integrator(bounds)(test_integrand2, nitn=10, neval=5e4)
    elapsed = time.perf_counter() - t0
    print(f"Time taken: {elapsed:.2f} seconds")
    #result2 = vegas.Integrator(bounds)(test_integrand2, nitn=10, neval=25000)
    print("test_integrand: ", result.summary())
    #print("test_integrand2:", result2.summary())

