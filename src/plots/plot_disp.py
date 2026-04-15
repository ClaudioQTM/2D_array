import sys
from pathlib import Path

# Find project root (folder containing pyproject.toml), then add src/
project_root = None
p = Path.cwd().resolve()
for d in [p, *p.parents]:
    if (d / "pyproject.toml").exists():
        project_root = d
        sys.path.insert(0, str(d / "src"))
        break
if project_root is None:
    raise RuntimeError("Could not find project root (pyproject.toml)")

import numpy as np
import pandas as pd
#import plotly.express as px
#import webbrowser
from joblib import Parallel, delayed

from eigenstate_solving.eigen_eq_integrand import BZ_proj
from model import SquareLattice, field
from smatrix import create_self_energy_interpolator_numba, t_reg

square_lattice01 = SquareLattice(
    a_lmd_ratio=0.1,
    omega_e=100.0,
    dipole_unit_vector=np.array([1, 1j, 0]) / np.sqrt(2),
    gamma=1,
    field=field,
    grid_cutoff=50,
)
sigma_path = project_root / "data" / "sigma_grid0f1a.npz"
if not sigma_path.exists():
    raise FileNotFoundError(f"Missing data file: {sigma_path}")

sigma_data = np.load(sigma_path)
kx = sigma_data["kx"]
ky = sigma_data["ky"]
sigma_grid = sigma_data["sigma_grid"]
sigma_func_period_numba = create_self_energy_interpolator_numba(
    kx, ky, sigma_grid, lattice=square_lattice01
)

collective_lamb_shift = sigma_func_period_numba(0,0)
# Uniformly sample the product $t(k_\parallel,E_1)t(Q_\parallel,E-E_1)$ in the region $||\vec{k}_\parallel||< 2\omega_e$
eps = 1e-3  # avoid zero-division at integration boundaries
n_energy_points = 10000
E = 2 * (square_lattice01.omega_e + collective_lamb_shift)
n_Q_samples = 100
n_k_samples = 50000


def _sample_for_k(
    k_para: np.ndarray,
    Q_para: np.ndarray,
    E: float,
    eps: float,
    n_energy_points: int,
):
    """Accept k_para as shape (2,) or batch shape (N, 2)."""
    k_batch = np.atleast_2d(k_para)

    # Vectorized projection for all k points at once.
    l_batch = BZ_proj(Q_para - k_batch, square_lattice01)

    k_norm = np.linalg.norm(k_batch, axis=1)
    l_norm = np.linalg.norm(l_batch, axis=1)
    width = E - k_norm - l_norm

    values = []

    valid_idx = np.where(width > 0)[0]  # return the array of indices
    for idx in valid_idx:
        k_i = k_batch[idx]
        l_i = l_batch[idx]

        E1_grid = np.linspace(
            k_norm[idx] + eps,
            E - l_norm[idx] - eps,
            n_energy_points,
        )
        values.extend(
            [
                t_reg(k_i, E1, square_lattice01, sigma_func_period_numba)
                * t_reg(l_i, E - E1, square_lattice01, sigma_func_period_numba)
                for E1 in E1_grid
            ]
        )

    return np.array(values, dtype=complex)


def disp_for_Q(E, Q_para, n_samples):
    k_para_list = []

    # uniformly sampling in a circle
    for i in range(0, n_samples):
        U = np.random.uniform(0, 1.0)
        theta = np.random.uniform(0, 2 * np.pi)

        kx = E * np.sqrt(U) * np.cos(theta)
        ky = E * np.sqrt(U) * np.sin(theta)
        k_para_list.append(np.array([kx, ky]))

    k_para_list = np.array(k_para_list)

    samples = _sample_for_k(k_para_list, Q_para, E, eps, n_energy_points)

    samples = np.array(samples)
    return samples


def _generate_Q_list(E, n_samples):
    Q_para_list = []
    for i in range(0, n_samples):
        U = np.random.uniform(0, 1.0)
        theta = np.random.uniform(0, 2 * np.pi)

        kx = E * np.sqrt(U) * np.cos(theta)
        ky = E * np.sqrt(U) * np.sin(theta)
        Q_para_list.append(np.array([kx, ky]))
    return Q_para_list


Q_para_list = _generate_Q_list(E, n_Q_samples)
samples_array = Parallel(6, backend="loky")(
    delayed(disp_for_Q)(E, Q_para, n_samples=n_k_samples) for Q_para in Q_para_list
)

# Build cloud points from samples_array and Q_list.
# x = Qx, y = Qy, z = arg(sample)
# Keep this conservative to avoid Plotly transport/render failures.
max_points_per_Q = 12_000
max_total_points = 120_000

Qx_all = []
Qy_all = []
phase_all = []

for Q_para, samples in zip(Q_para_list, samples_array):
    if samples.size == 0:
        continue

    # Downsample each Q bucket for faster Plotly rendering.
    if samples.size > max_points_per_Q:
        idx = np.random.choice(samples.size, size=max_points_per_Q, replace=False)
        samples_use = samples[idx]
    else:
        samples_use = samples

    phase = np.angle(samples_use)
    Qx_all.append(np.full(phase.shape, Q_para[0]))
    Qy_all.append(np.full(phase.shape, Q_para[1]))
    phase_all.append(phase)

Qx_all = np.concatenate(Qx_all)
Qy_all = np.concatenate(Qy_all)
phase_all = np.concatenate(phase_all)

finite_mask = np.isfinite(Qx_all) & np.isfinite(Qy_all) & np.isfinite(phase_all)
Qx_all = Qx_all[finite_mask]
Qy_all = Qy_all[finite_mask]
phase_all = phase_all[finite_mask]

if phase_all.size == 0:
    raise RuntimeError("All sampled points are non-finite; nothing to plot.")

# Global downsample to keep browser-side WebGL stable.
if phase_all.size > max_total_points:
    idx = np.random.choice(phase_all.size, size=max_total_points, replace=False)
    Qx_all = Qx_all[idx]
    Qy_all = Qy_all[idx]
    phase_all = phase_all[idx]

df = pd.DataFrame(
    {
        "Qx": Qx_all,
        "Qy": Qy_all,
        "phase": phase_all,
    }
)

df.to_csv("data/plot_disp_points.csv", index=False)

