import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sys
from pathlib import Path

# Ensure the top-level `src` directory (one level above `plots/`) is on sys.path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smatrix import alpha, self_energy, square_lattice


def _sigma_real_for_a(a: float, omega_e: float) -> float:
    """Helper to compute Re[self_energy] for given lattice constant a."""
    sigma = self_energy(
        0.0,
        0.0,
        a,
        square_lattice.d,
        omega_e,
        alpha,
        summation_type="k",
    )
    return float(np.real(sigma))


def main():
    """
    1D scan of Re[Σ(k_xy = 0, omega = omega_e)] as a function of a / lambda_e.

    We keep:
    - k_xy = (0, 0)
    - omega = square_lattice.omega_e
    - dipole moment d = square_lattice.d (fixed by the default gamma, omega_e)

    and vary the lattice spacing a via the ratio a / lambda_e in [0.2, 0.99].
    """
    omega_e = square_lattice.omega_e

    # Use the same speed of light as in the model (c = 1 in natural units),
    # so that lambda_e matches the lattice definition.
    lambda_e = 2 * np.pi / (omega_e)  # since c = 1 in model.py

    ratios = np.linspace(0.1, 0.999, 10)
    a_values = ratios * lambda_e

    # Parallel evaluation over the grid of `a` values
    sigma_real = Parallel(n_jobs=8)(
        delayed(_sigma_real_for_a)(float(a), float(omega_e)) for a in a_values
    )
    sigma_real = np.asarray(sigma_real, dtype=float)
    sigma_real = sigma_real/square_lattice.gamma
    plt.figure(figsize=(6, 4))
    plt.plot(ratios, sigma_real, marker="o", markersize=3, linestyle="-")
    plt.xlabel(r"$a / \lambda_e$")
    plt.ylabel(r"$\mathrm{Re}\,\Sigma(k_{xy}=0,\omega=\omega_e)$")
    plt.title("Real part of self-energy vs $a/\\lambda_e$")
    plt.xlim(0, 1)
    plt.ylim(-3, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

