import matplotlib.pyplot as plt
import numpy as np

try:
    from .eigen_eq_integrand import _make_eigen_eq_integrand
except ImportError:
    # Allow running this file directly as a script from the repo root.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eigenstate_solving.eigen_eq_integrand import _make_eigen_eq_integrand

def plot_integrand1(E,Q,G,H,D_normalized,sigma_func_period,lattice,tEQ):
    integrand = _make_eigen_eq_integrand(E, Q, G, H, lattice, sigma_func_period, tEQ)

    transformed_qx = np.linspace(-1,1, 150)
    transformed_qy = np.linspace(-1,1, 150)

    integrand_values = np.zeros((len(transformed_qx), len(transformed_qy)), dtype=complex)
    for i,qx in enumerate(transformed_qx):
        for j,qy in enumerate(transformed_qy):
            x = np.array([qx, qy, D_normalized])
            integrand_values[i,j] = integrand(x)

    real_plot = np.real(integrand_values)
    imag_plot = np.imag(integrand_values)

    qx_mesh, qy_mesh = np.meshgrid(transformed_qx, transformed_qy, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    heat_re = axes[0].contourf(qx_mesh, qy_mesh, real_plot, levels=60, cmap="viridis")
    fig.colorbar(heat_re, ax=axes[0], label="Re[integrand]")
    axes[0].set_title("Real part heatmap")
    axes[0].set_xlabel("transformed qx")
    axes[0].set_ylabel("transformed qy")

    heat_im = axes[1].contourf(qx_mesh, qy_mesh, imag_plot, levels=60, cmap="viridis")
    fig.colorbar(heat_im, ax=axes[1], label="Im[integrand]")
    axes[1].set_title("Imaginary part heatmap")
    axes[1].set_xlabel("transformed qx")
    axes[1].set_ylabel("transformed qy")

    plt.show()
    return fig, axes, integrand_values
    

__all__ = ["plot_integrand1"]

