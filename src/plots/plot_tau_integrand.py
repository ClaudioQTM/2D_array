"""Plot the tau-matrix integrand over the Brillouin zone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.defaults import collective_lamb_shift  # noqa: E402
from smatrix import create_self_energy_interpolator_numba, square_lattice  # noqa: E402


def _make_integrand_grid(E, Qx, Qy, sigma_func_period, lattice, n_points):
    bound = np.pi / lattice.a
    qx_grid = np.linspace(-bound, bound, n_points)
    qy_grid = np.linspace(-bound, bound, n_points)
    integrand_grid = np.empty((n_points, n_points), dtype=np.complex128)
    denominator_grid = np.empty((n_points, n_points), dtype=np.complex128)

    for ix, qx in enumerate(qx_grid):
        for iy, qy in enumerate(qy_grid):
            sigma1 = sigma_func_period(qx, qy)
            sigma2 = sigma_func_period(Qx - qx, Qy - qy)
            denominator = E - 2 * lattice.omega_e - sigma1 - sigma2
            denominator_grid[ix, iy] = denominator
            integrand_grid[ix, iy] = 1.0 / (denominator)

    return qx_grid, qy_grid, integrand_grid, denominator_grid


def find_singularity_candidates(qx_grid, qy_grid, denominator_grid, top_k=12):
    """
    Return unique candidate singularity points where |denominator| is smallest.

    A true singularity satisfies denominator == 0 (both real and imaginary parts).
    On a finite grid, we report the nearest sampled points.
    """
    abs_den = np.abs(denominator_grid)
    flat_order = np.argsort(abs_den.ravel())
    min_spacing = max(1, int(0.02 * len(qx_grid)))

    candidates = []
    for flat_idx in flat_order:
        ix, iy = np.unravel_index(flat_idx, abs_den.shape)

        keep = True
        for old_ix, old_iy, *_ in candidates:
            if abs(ix - old_ix) <= min_spacing and abs(iy - old_iy) <= min_spacing:
                keep = False
                break

        if keep:
            den = denominator_grid[ix, iy]
            candidates.append(
                (
                    ix,
                    iy,
                    float(qx_grid[ix]),
                    float(qy_grid[iy]),
                    float(np.abs(den)),
                    float(den.real),
                    float(den.imag),
                )
            )
            if len(candidates) >= top_k:
                break

    return candidates


def plot_tau_integrand(
    qx_grid,
    qy_grid,
    integrand_grid,
    E,
    Qx,
    Qy,
    save_path=None,
    show=True,
    singularity_candidates=None,
):
    QX, QY = np.meshgrid(qx_grid, qy_grid, indexing="ij")

    real_part = integrand_grid.real
    imag_part = integrand_grid.imag
    magnitude = np.abs(integrand_grid)
    phase = np.angle(integrand_grid)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    im0 = axes[0].contourf(QX, QY, real_part, levels=80, cmap="RdBu_r")
    axes[0].set_title("Re[integrand]")
    axes[0].set_xlabel(r"$q_x$")
    axes[0].set_ylabel(r"$q_y$")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(QX, QY, imag_part, levels=80, cmap="RdBu_r")
    axes[1].set_title("Im[integrand]")
    axes[1].set_xlabel(r"$q_x$")
    axes[1].set_ylabel(r"$q_y$")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(QX, QY, magnitude, levels=80, cmap="viridis")
    axes[2].set_title("|integrand|")
    axes[2].set_xlabel(r"$q_x$")
    axes[2].set_ylabel(r"$q_y$")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].contourf(QX, QY, phase, levels=80, cmap="twilight")
    axes[3].set_title("arg(integrand)")
    axes[3].set_xlabel(r"$q_x$")
    axes[3].set_ylabel(r"$q_y$")
    axes[3].set_aspect("equal")
    plt.colorbar(im3, ax=axes[3])

    if singularity_candidates:
        cand_qx = [x[2] for x in singularity_candidates]
        cand_qy = [x[3] for x in singularity_candidates]
        for ax in axes:
            ax.scatter(
                cand_qx,
                cand_qy,
                s=28,
                c="white",
                edgecolors="black",
                linewidths=0.8,
                marker="o",
                label="near-singular",
            )
        axes[2].legend(loc="upper right", fontsize=8)

    fig.suptitle(
        rf"tau integrand: $E={E:.8g}$, $Q=({Qx:.4g}, {Qy:.4g})$",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _parse_args():
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_sigma = repo_root / "data" / "sigma_grid0f2a.npz"
    default_out = repo_root / "src" / "plot" / "tau_integrand.png"

    parser = argparse.ArgumentParser(
        description="Plot the integrand used in tau_matrix_element."
    )
    parser.add_argument(
        "--sigma-data",
        type=Path,
        default=default_sigma,
        help="Path to .npz file with arrays: kx, ky, sigma_grid.",
    )
    parser.add_argument(
        "--E",
        type=float,
        default=None,
        help="Total two-photon energy. Default uses 2*(omega_e + collective_lamb_shift).",
    )
    parser.add_argument("--Qx", type=float, default=0.0, help="Total momentum Qx.")
    parser.add_argument("--Qy", type=float, default=0.0, help="Total momentum Qy.")
    parser.add_argument(
        "--n-points",
        type=int,
        default=201,
        help="Grid points per momentum axis for qx,qy.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figure to --output path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help="Output path when --save is given.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (useful for remote/headless runs).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of near-singular candidate points to report/mark.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if not args.sigma_data.exists():
        raise FileNotFoundError(f"Sigma data file not found: {args.sigma_data}")

    sigma_data = np.load(args.sigma_data)
    kx = sigma_data["kx"]
    ky = sigma_data["ky"]
    sigma_grid = sigma_data["sigma_grid"]

    sigma_func_period = create_self_energy_interpolator_numba(
        kx, ky, sigma_grid, lattice=square_lattice
    )

    E = args.E
    if E is None:
        E = 2 * (square_lattice.omega_e + collective_lamb_shift)

    qx_grid, qy_grid, integrand_grid, denominator_grid = _make_integrand_grid(
        E=E,
        Qx=args.Qx,
        Qy=args.Qy,
        sigma_func_period=sigma_func_period,
        lattice=square_lattice,
        n_points=args.n_points,
    )
    singularity_candidates = find_singularity_candidates(
        qx_grid=qx_grid,
        qy_grid=qy_grid,
        denominator_grid=denominator_grid,
        top_k=args.top_k,
    )

    print("\nNear-singular candidates (smallest |denominator|):")
    print(" idx    qx           qy           |den|         Re(den)       Im(den)")
    print("-----------------------------------------------------------------------")
    for idx, (_ix, _iy, qx, qy, abs_den, re_den, im_den) in enumerate(
        singularity_candidates, start=1
    ):
        print(
            f"{idx:>3d}  {qx:>11.6f}  {qy:>11.6f}  {abs_den:>11.4e}  {re_den:>11.4e}  {im_den:>11.4e}"
        )

    save_path = args.output if args.save else None
    plot_tau_integrand(
        qx_grid=qx_grid,
        qy_grid=qy_grid,
        integrand_grid=integrand_grid,
        E=E,
        Qx=args.Qx,
        Qy=args.Qy,
        save_path=save_path,
        show=not args.no_show,
        singularity_candidates=singularity_candidates,
    )


if __name__ == "__main__":
    main()
