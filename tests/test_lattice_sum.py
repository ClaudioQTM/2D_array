import numpy as np
from joblib import Parallel, delayed

from model import self_energy, real_space_summation, k_space_summation
from smatrix import square_lattice, alpha


def test_self_energy_reflection_symmetry(tolerance: float = 1e-10, n_tests: int = 20):
    """
    Verify that the single-particle self-energy is reflection-symmetric in the 1st BZ.

    Checks, for random k-points in the first Brillouin zone of `square_lattice`:
        σ(kx, ky) == σ(kx, -ky)  (x-axis reflection)
        σ(kx, ky) == σ(-kx, ky)  (y-axis reflection)
    """
    lattice = square_lattice
    rng = np.random.default_rng(42)

    # First Brillouin zone: |kx|, |ky| <= q/2 = pi/a
    k_max = float(np.pi / lattice.a) * 0.9  # stay inside the BZ to avoid edge artefacts

    for _ in range(n_tests):
        kx = rng.uniform(0.0, k_max)
        ky = rng.uniform(0.0, k_max)

        sigma_orig = self_energy(kx, ky, lattice.a, lattice.d, lattice.omega_e, alpha)
        sigma_x_ref = self_energy(kx, -ky, lattice.a, lattice.d, lattice.omega_e, alpha)
        sigma_y_ref = self_energy(-kx, ky, lattice.a, lattice.d, lattice.omega_e, alpha)

        diff_x = abs(sigma_orig - sigma_x_ref)
        diff_y = abs(sigma_orig - sigma_y_ref)

        assert diff_x < tolerance, (
            f"x-reflection symmetry broken at k=({kx:.4f},{ky:.4f}): "
            f"|σ(kx,ky)-σ(kx,-ky)| = {diff_x:.3e} ≥ {tolerance:.1e}"
        )
        assert diff_y < tolerance, (
            f"y-reflection symmetry broken at k=({kx:.4f},{ky:.4f}): "
            f"|σ(kx,ky)-σ(-kx,ky)| = {diff_y:.3e} ≥ {tolerance:.1e}"
        )


def _summation_diff_for_k(kx: float, ky: float, a: float, d, omega: float, alpha_val: float):
    """Worker: compute real-space and k-space sums and their difference for a single k."""
    k_xy = np.array([kx, ky], dtype=float)
    real_val = real_space_summation(a, d, k_xy, omega)
    k_val = k_space_summation(a, d, k_xy, omega, alpha_val)
    diff = abs(complex(real_val) - complex(k_val))
    return kx, ky, diff


def test_real_vs_k_space_summation_first_BZ(n_tests: int = 50, tolerance: float = 1e-2):
    """
    Check that real-space and k-space lattice sums agree on random points in the 1st BZ.

    We draw `n_tests` random `k_xy` in the first Brillouin zone of the default
    `square_lattice` and compare `real_space_summation` to `k_space_summation`
    using the same lattice spacing `a`, dipole `d`, transition frequency
    `omega_e` and regularisation parameter `alpha`.
    """
    rng = np.random.default_rng(42)

    a = float(square_lattice.a)
    d = square_lattice.d
    omega = float(square_lattice.omega_e)

    # First Brillouin zone for the square lattice: |kx|, |ky| <= q/2 = pi/a
    k_max = float(np.pi / a)

    # Always include standard high-symmetry points in the BZ, then add random points.
    hs = k_max
    high_symmetry = np.array(
        [
            [0.0, 0.0],   # Γ
            [hs, 0.0],    # X
            [0.0, hs],    # Y
            [-hs, 0.0],
            [0.0, -hs],
            [hs, hs],     # M
            [hs, -hs],
            [-hs, hs],
            [-hs, -hs],
        ],
        dtype=float,
    )
    n_hs = high_symmetry.shape[0]
    n_random = max(n_tests - n_hs, 0)

    random_points = rng.uniform(-k_max, k_max, size=(n_random, 2)) if n_random > 0 else np.empty((0, 2), dtype=float)
    # Total set of test points (high-symmetry first, then random).
    k_points = np.vstack([high_symmetry, random_points])

    results = Parallel(n_jobs=6)(
        delayed(_summation_diff_for_k)(kx, ky, a, d, omega, alpha)
        for kx, ky in k_points
    )

    for kx, ky, diff in results:
        assert diff < tolerance, (
            f"real_space_summation and k_space_summation disagree at "
            f"k=({kx:.4f}, {ky:.4f}): |Δ| = {diff:.3e} ≥ {tolerance:.1e}"
        )

'''

import numpy as np

from model import self_energy, real_space_summation, k_space_summation
from smatrix import square_lattice, alpha



def test_self_energy_reflection_symmetry(lattice, alpha=1e-4, tolerance=1e-10, n_tests=20):
    """
    Test whether self_energy is symmetric with respect to reflection along x and y axes
    inside the 1st Brillouin Zone.
    
    Tests:
    - x-axis reflection: σ(kx, ky) = σ(kx, -ky)
    - y-axis reflection: σ(kx, ky) = σ(-kx, ky)
    
    Parameters:
        lattice: SquareLattice instance
        alpha: regularization parameter for k-space sum
        tolerance: maximum allowed difference (default 1e-10)
        n_tests: number of random test points
    
    Returns:
        dict with test results
    """
    np.random.seed(42)  # reproducibility
    
    # k range within the first Brillouin zone: |k| < pi/a
    k_max = float(np.pi / lattice.a) * 0.9  # stay within BZ
    
    results = {
        'x_reflection': {'passed': 0, 'failed': 0, 'diffs': []},
        'y_reflection': {'passed': 0, 'failed': 0, 'diffs': []},
        'failures': []
    }
    
    print(f"Testing self-energy reflection symmetry in 1st BZ (tolerance = {tolerance})...")
    print("="*80)
    
    for i in range(n_tests):
        # Random k_xy within Brillouin zone
        kx = np.random.uniform(0, k_max)  # Use positive kx to test reflection
        ky = np.random.uniform(0, k_max)  # Use positive ky to test reflection
        
        # Compute self-energy at various reflected points
        sigma_orig = self_energy(kx, ky, lattice.a, lattice.d, lattice.omega_e, lattice.omega_e, alpha)
        sigma_x_ref = self_energy(kx, -ky, lattice.a, lattice.d, lattice.omega_e, lattice.omega_e, alpha)  # x-axis reflection
        sigma_y_ref = self_energy(-kx, ky, lattice.a, lattice.d, lattice.omega_e, lattice.omega_e, alpha)  # y-axis reflection
        
        # Check x-axis reflection symmetry: σ(kx, ky) = σ(kx, -ky)
        diff_x = abs(sigma_orig - sigma_x_ref)
        results['x_reflection']['diffs'].append(diff_x)
        status_x = "PASS" if diff_x < tolerance else "FAIL"
        if diff_x < tolerance:
            results['x_reflection']['passed'] += 1
        else:
            results['x_reflection']['failed'] += 1
            results['failures'].append({
                'type': 'x_reflection',
                'kx': kx, 'ky': ky,
                'sigma_orig': sigma_orig, 'sigma_ref': sigma_x_ref,
                'diff': diff_x
            })
        
        # Check y-axis reflection symmetry: σ(kx, ky) = σ(-kx, ky)
        diff_y = abs(sigma_orig - sigma_y_ref)
        results['y_reflection']['diffs'].append(diff_y)
        status_y = "PASS" if diff_y < tolerance else "FAIL"
        if diff_y < tolerance:
            results['y_reflection']['passed'] += 1
        else:
            results['y_reflection']['failed'] += 1
            results['failures'].append({
                'type': 'y_reflection',
                'kx': kx, 'ky': ky,
                'sigma_orig': sigma_orig, 'sigma_ref': sigma_y_ref,
                'diff': diff_y
            })
        
        print(f"Test {i+1:3d}: k=({kx:.4f}, {ky:.4f})")
        print(f"         σ(kx, ky)  = {sigma_orig:.6e}")
        print(f"         σ(kx,-ky)  = {sigma_x_ref:.6e}  diff={diff_x:.2e} [{status_x}]")
        print(f"         σ(-kx,ky)  = {sigma_y_ref:.6e}  diff={diff_y:.2e} [{status_y}]")
        print()
    
    print("="*80)
    print("SUMMARY:")
    print("-"*80)
    print(f"X-axis reflection (kx, ky) -> (kx, -ky):")
    print(f"  Passed: {results['x_reflection']['passed']}/{n_tests} ({100*results['x_reflection']['passed']/n_tests:.1f}%)")
    if results['x_reflection']['diffs']:
        print(f"  Mean diff: {np.mean(results['x_reflection']['diffs']):.2e}")
        print(f"  Max diff:  {np.max(results['x_reflection']['diffs']):.2e}")
    
    print(f"\nY-axis reflection (kx, ky) -> (-kx, ky):")
    print(f"  Passed: {results['y_reflection']['passed']}/{n_tests} ({100*results['y_reflection']['passed']/n_tests:.1f}%)")
    if results['y_reflection']['diffs']:
        print(f"  Mean diff: {np.mean(results['y_reflection']['diffs']):.2e}")
        print(f"  Max diff:  {np.max(results['y_reflection']['diffs']):.2e}")
    
    print("="*80)
    
    # Overall verdict
    total_passed = results['x_reflection']['passed'] + results['y_reflection']['passed']
    total_tests = 2 * n_tests
    if total_passed == total_tests:
        print("\n✓ SYMMETRY CONFIRMED: self_energy has reflection symmetry along both x and y axes!")
    else:
        print(f"\n✗ SYMMETRY BROKEN: {total_tests - total_passed} tests failed")
        if results['failures']:
            print("\nFirst few failures:")
            for f in results['failures'][:5]:
                print(f"  {f['type']}: k=({f['kx']:.4f}, {f['ky']:.4f}), diff={f['diff']:.2e}")
    
    return results



def test_real_vs_k_space_summation_first_BZ(n_tests: int = 20, tolerance: float = 1e-2):
    """
    Check that real-space and k-space lattice sums agree on random points in the 1st BZ.

    We draw `n_tests` random `k_xy` in the first Brillouin zone of the default
    `square_lattice` and compare `real_space_summation` to `k_space_summation`
    using the same lattice spacing `a`, dipole `d`, transition frequency
    `omega_e` and regularisation parameter `alpha`.
    """
    rng = np.random.default_rng(42)

    a = float(square_lattice.a)
    d = square_lattice.d
    omega = float(square_lattice.omega_e)

    # First Brillouin zone for the square lattice: |kx|, |ky| <= q/2 = pi/a
    k_max = float(np.pi / a)

    for _ in range(n_tests):
        kx, ky = rng.uniform(-k_max, k_max, size=2)
        k_xy = np.array([kx, ky], dtype=float)

        real_val = real_space_summation(a, d, k_xy, omega)
        k_val = k_space_summation(a, d, k_xy, omega, alpha)

        diff = abs(complex(real_val) - complex(k_val))
        assert diff < tolerance, (
            f"real_space_summation and k_space_summation disagree at "
            f"k=({kx:.4f}, {ky:.4f}): |Δ| = {diff:.3e} ≥ {tolerance:.1e}"
        )



def plot_sigma_grid(kx_grid, ky_grid, sigma_grid, save_plots=False, figsize=(14, 6)):
    """
    Create 2D color plots for the real and imaginary parts of sigma_grid.
    
    Parameters:
    -----------
    kx_grid : array-like
        1D array of kx values
    ky_grid : array-like
        1D array of ky values
    sigma_grid : array-like
        2D complex array of self-energy values
    save_plots : bool
        Whether to save plots to files (default: False)
    figsize : tuple
        Figure size (width, height) in inches (default: (14, 6))
    """
    # Create meshgrid for plotting
    KX, KY = np.meshgrid(kx_grid, ky_grid)
    
    # Extract real and imaginary parts
    real_part = sigma_grid.real
    imag_part = sigma_grid.imag
    magnitude = np.abs(sigma_grid)
    phase = np.angle(sigma_grid)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # 1. Real part
    ax1 = axes[0]
    im1 = ax1.contourf(KX, KY, real_part, levels=50, cmap='RdBu_r')
    ax1.set_xlabel('$k_x$', fontsize=12)
    ax1.set_ylabel('$k_y$', fontsize=12)
    ax1.set_title('Real Part of $\\sigma$', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Re($\\sigma$)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Imaginary part
    ax2 = axes[1]
    im2 = ax2.contourf(KX, KY, imag_part, levels=50, cmap='RdBu_r')
    ax2.set_xlabel('$k_x$', fontsize=12)
    ax2.set_ylabel('$k_y$', fontsize=12)
    ax2.set_title('Imaginary Part of $\\sigma$', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='Im($\\sigma$)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Magnitude
    ax3 = axes[2]
    im3 = ax3.contourf(KX, KY, magnitude, levels=50, cmap='viridis')
    ax3.set_xlabel('$k_x$', fontsize=12)
    ax3.set_ylabel('$k_y$', fontsize=12)
    ax3.set_title('Magnitude of $\\sigma$', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label='|$\\sigma$|')
    ax3.grid(True, alpha=0.3)
    
    # 4. Phase
    ax4 = axes[3]
    im4 = ax4.contourf(KX, KY, phase, levels=50, cmap='hsv')
    ax4.set_xlabel('$k_x$', fontsize=12)
    ax4.set_ylabel('$k_y$', fontsize=12)
    ax4.set_title('Phase of $\\sigma$', fontsize=14, fontweight='bold')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4, label='Phase (rad)')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Self-Energy Grid Visualization', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_plots:
        filename = 'sigma_grid_visualization.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("SIGMA_GRID STATISTICS")
    print("="*60)
    print(f"Real part - Min: {real_part.min():.6e}, Max: {real_part.max():.6e}, Mean: {real_part.mean():.6e}")
    print(f"Imaginary part - Min: {imag_part.min():.6e}, Max: {imag_part.max():.6e}, Mean: {imag_part.mean():.6e}")
    print(f"Magnitude - Min: {magnitude.min():.6e}, Max: {magnitude.max():.6e}, Mean: {magnitude.mean():.6e}")
    print(f"Phase - Min: {phase.min():.4f} rad, Max: {phase.max():.4f} rad")
    print("="*60 + "\n")
    
    return fig


def visualize_integrand(E, Q, lattice, n_points=200, save_plots=False):
    """
    Visualize and analyze the real and imaginary parts of the integrand.
    
    Parameters:
    -----------
    E : float
        Energy parameter
    Q : array-like
        Momentum vector [Qx, Qy]
    lattice : SquareLattice
        Lattice object
    n_points : int
        Number of grid points for visualization (default: 200)
    save_plots : bool
        Whether to save plots to files (default: False)
    """
    Qx, Qy = Q
    bound = np.pi / lattice.a
    
    def integrand(qx, qy):
        sigma1 = sigma_func_period(qx, qy, lattice)
        sigma2 = sigma_func_period(Qx - qx, Qy - qy, lattice)
        return 1 / (E - 2*lattice.omega_e - sigma1 - sigma2)
    
    # Create 2D grid
    qx_grid = np.linspace(-bound, bound, n_points)
    qy_grid = np.linspace(-bound, bound, n_points)
    QX, QY = np.meshgrid(qx_grid, qy_grid)
    
    # Evaluate integrand on grid (vectorized for speed)
    print("Evaluating integrand on grid...")
    try:
        # Try vectorized evaluation (faster)
        sigma1_grid = sigma_func_period(QX, QY, lattice)
        sigma2_grid = sigma_func_period(Qx - QX, Qy - QY, lattice)
        denominator_grid = E - 2*lattice.omega_e - sigma1_grid - sigma2_grid
        integrand_values = 1 / denominator_grid
    except (TypeError, ValueError):
        # Fallback to loop-based evaluation if vectorized fails
        print("Vectorized evaluation failed, using loop-based evaluation...")
        integrand_values = np.zeros_like(QX, dtype=complex)
        for i, qx in enumerate(qx_grid):
            for j, qy in enumerate(qy_grid):
                integrand_values[j, i] = integrand(qx, qy)
    
    # Extract real and imaginary parts
    real_part = integrand_values.real
    imag_part = integrand_values.imag
    magnitude = np.abs(integrand_values)
    phase = np.angle(integrand_values)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Real part heatmap
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.contourf(QX, QY, real_part, levels=50, cmap='RdBu_r')
    ax1.set_xlabel('$q_x$', fontsize=12)
    ax1.set_ylabel('$q_y$', fontsize=12)
    ax1.set_title('Real Part of Integrand', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Real')
    ax1.grid(True, alpha=0.3)
    
    # 2. Imaginary part heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.contourf(QX, QY, imag_part, levels=50, cmap='RdBu_r')
    ax2.set_xlabel('$q_x$', fontsize=12)
    ax2.set_ylabel('$q_y$', fontsize=12)
    ax2.set_title('Imaginary Part of Integrand', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='Imaginary')
    ax2.grid(True, alpha=0.3)
    
    # 3. Magnitude heatmap
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.contourf(QX, QY, magnitude, levels=50, cmap='viridis')
    ax3.set_xlabel('$q_x$', fontsize=12)
    ax3.set_ylabel('$q_y$', fontsize=12)
    ax3.set_title('Magnitude of Integrand', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label='|Integrand|')
    ax3.grid(True, alpha=0.3)
    
    # 4. Phase heatmap
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.contourf(QX, QY, phase, levels=50, cmap='hsv')
    ax4.set_xlabel('$q_x$', fontsize=12)
    ax4.set_ylabel('$q_y$', fontsize=12)
    ax4.set_title('Phase of Integrand', fontsize=14, fontweight='bold')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4, label='Phase (rad)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Cross-section along qx=constant
    ax5 = plt.subplot(2, 3, 5)
#    qy_idx = n_points // 2  # Index corresponding to qx=0
    qy_idx = 10
    qx_fixed = qx_grid[qy_idx]
    
    # Plot integrand
    ax5.plot(qy_grid, real_part[qy_idx, :], 'b-', label='Integrand Real', linewidth=2)
    ax5.plot(qy_grid, imag_part[qy_idx, :], 'r-', label='Integrand Imag', linewidth=2)
    ax5.set_xlabel(f'$q_y$ (at $q_x={qx_fixed:.2f}$)', fontsize=12)
    ax5.set_ylabel('Integrand Value', fontsize=12, color='black')
    ax5.tick_params(axis='y', labelcolor='black')
    
    # Secondary y-axis for self-energy
    ax5_twin = ax5.twinx()
    sigma1_line = sigma1_grid[qy_idx, :] if len(sigma1_grid.shape) > 1 else np.array([sigma_func_period(qx_fixed, qy, lattice) for qy in qy_grid])
    sigma2_line = sigma2_grid[qy_idx, :] if len(sigma2_grid.shape) > 1 else np.array([sigma_func_period(Qx - qx_fixed, Qy - qy, lattice) for qy in qy_grid])
    # Plot sigma2 first, then sigma1 on top (so green is visible when they overlap)
    ax5_twin.plot(qy_grid, sigma2_line.real, 'm--', label=r'$\Sigma_2$ Real', linewidth=2.5, alpha=0.7)
    ax5_twin.plot(qy_grid, sigma2_line.imag, 'm:', label=r'$\Sigma_2$ Imag', linewidth=2.5, alpha=0.7)
    ax5_twin.plot(qy_grid, sigma1_line.real, 'g-', label=r'$\Sigma_1$ Real', linewidth=1.5, alpha=1.0)
    ax5_twin.plot(qy_grid, sigma1_line.imag, 'g--', label=r'$\Sigma_1$ Imag', linewidth=1.5, alpha=1.0)
    ax5_twin.set_ylabel('Self-energy $\\Sigma$', fontsize=12, color='green')
    ax5_twin.tick_params(axis='y', labelcolor='green')
    
    ax5.set_title(f'Cross-section along $q_x={qx_fixed:.2f}$', fontsize=14, fontweight='bold')
    # Combine legends from both axes
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Cross-section along qy=constant
    ax6 = plt.subplot(2, 3, 6)
    qx_idx = n_points // 2  # Index corresponding to qy=0
    qy_fixed = qy_grid[qx_idx]
    
    # Plot integrand
    ax6.plot(qx_grid, real_part[:, qx_idx], 'b-', label='Integrand Real', linewidth=2)
    ax6.plot(qx_grid, imag_part[:, qx_idx], 'r-', label='Integrand Imag', linewidth=2)
    ax6.set_xlabel(f'$q_x$ (at $q_y={qy_fixed:.2f}$)', fontsize=12)
    ax6.set_ylabel('Integrand Value', fontsize=12, color='black')
    ax6.tick_params(axis='y', labelcolor='black')
    
    # Secondary y-axis for self-energy
    ax6_twin = ax6.twinx()
    sigma1_line_x = sigma1_grid[:, qx_idx] if len(sigma1_grid.shape) > 1 else np.array([sigma_func_period(qx, qy_fixed, lattice) for qx in qx_grid])
    sigma2_line_x = sigma2_grid[:, qx_idx] if len(sigma2_grid.shape) > 1 else np.array([sigma_func_period(Qx - qx, Qy - qy_fixed, lattice) for qx in qx_grid])
    # Plot sigma2 first, then sigma1 on top (so green is visible when they overlap)
    ax6_twin.plot(qx_grid, sigma2_line_x.real, 'm--', label=r'$\Sigma_2$ Real', linewidth=2.5, alpha=0.7)
    ax6_twin.plot(qx_grid, sigma2_line_x.imag, 'm:', label=r'$\Sigma_2$ Imag', linewidth=2.5, alpha=0.7)
    ax6_twin.plot(qx_grid, sigma1_line_x.real, 'g-', label=r'$\Sigma_1$ Real', linewidth=1.5, alpha=1.0)
    ax6_twin.plot(qx_grid, sigma1_line_x.imag, 'g--', label=r'$\Sigma_1$ Imag', linewidth=1.5, alpha=1.0)
    ax6_twin.set_ylabel('Self-energy $\\Sigma$', fontsize=12, color='green')
    ax6_twin.tick_params(axis='y', labelcolor='green')
    
    ax6.set_title(f'Cross-section along $q_y={qy_fixed:.2f}$', fontsize=14, fontweight='bold')
    # Combine legends from both axes
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    plt.suptitle(f'Integrand Analysis: E={E:.4f}, Q=({Qx:.4f}, {Qy:.4f})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'integrand_analysis_E{E:.4f}_Q{Qx:.4f}_{Qy:.4f}.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to integrand_analysis_E{E:.4f}_Q{Qx:.4f}_{Qy:.4f}.png")
    
    plt.show()
    
    # Print analysis statistics
    print("\n" + "="*60)
    print("INTEGRAND ANALYSIS STATISTICS")
    print("="*60)
    print(f"Real part - Min: {real_part.min():.6e}, Max: {real_part.max():.6e}, Mean: {real_part.mean():.6e}")
    print(f"Imaginary part - Min: {imag_part.min():.6e}, Max: {imag_part.max():.6e}, Mean: {imag_part.mean():.6e}")
    print(f"Magnitude - Min: {magnitude.min():.6e}, Max: {magnitude.max():.6e}, Mean: {magnitude.mean():.6e}")
    print(f"Phase - Min: {phase.min():.4f} rad, Max: {phase.max():.4f} rad")
    
    # Check for singularities (very large values)
    large_threshold = 1e6
    large_mask = magnitude > large_threshold
    n_singular = np.sum(large_mask)
    if n_singular > 0:
        print(f"\nWARNING: Found {n_singular} points with magnitude > {large_threshold:.2e}")
        print("These may indicate singularities or near-singularities in the integrand.")
    
    # Check for NaN or Inf values
    n_nan = np.sum(np.isnan(integrand_values))
    n_inf = np.sum(np.isinf(integrand_values))
    if n_nan > 0 or n_inf > 0:
        print(f"\nWARNING: Found {n_nan} NaN values and {n_inf} Inf values in the integrand.")
    
    print("="*60 + "\n")
    
    return {
        'qx_grid': qx_grid,
        'qy_grid': qy_grid,
        'real_part': real_part,
        'imag_part': imag_part,
        'magnitude': magnitude,
        'phase': phase,
        'integrand_values': integrand_values,
        'sigma1_grid': sigma1_grid,
        'sigma2_grid': sigma2_grid
    }



def visualize_sigma_func_period(sigma_func, lattice, n_points=200, save=False, figsize=(14, 10)):
    """Plot sigma_func on a 2D grid and its 1D cuts along kx=0 and ky=0."""
    bound = float(lattice.q / 2)
    kx_grid = np.linspace(-bound, bound, n_points)
    ky_grid = np.linspace(-bound, bound, n_points)
    KX, KY = np.meshgrid(kx_grid, ky_grid)

    sigma_vals = sigma_func(KX, KY, lattice)
    real_part = sigma_vals.real
    imag_part = sigma_vals.imag
    magnitude = np.abs(sigma_vals)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    im0 = axes[0, 0].contourf(KX, KY, real_part, levels=50, cmap='RdBu_r')
    axes[0, 0].set_title('Re($\\sigma$)')
    axes[0, 0].set_xlabel('$k_x$')
    axes[0, 0].set_ylabel('$k_y$')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].contourf(KX, KY, imag_part, levels=50, cmap='RdBu_r')
    axes[0, 1].set_title('Im($\\sigma$)')
    axes[0, 1].set_xlabel('$k_x$')
    axes[0, 1].set_ylabel('$k_y$')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 1])

    center_idx = n_points // 2
    axes[1, 0].plot(ky_grid, real_part[:, center_idx], label='Re')
    axes[1, 0].plot(ky_grid, imag_part[:, center_idx], label='Im')
    axes[1, 0].set_title('$\\sigma(k_x=0, k_y)$')
    axes[1, 0].set_xlabel('$k_y$')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(kx_grid, real_part[center_idx, :], label='Re')
    axes[1, 1].plot(kx_grid, imag_part[center_idx, :], label='Im')
    axes[1, 1].set_title('$\\sigma(k_x, k_y=0)$')
    axes[1, 1].set_xlabel('$k_x$')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('sigma_func_period_visualization.png', dpi=300, bbox_inches='tight')
    return {
        'kx_grid': kx_grid,
        'ky_grid': ky_grid,
        'sigma_vals': sigma_vals,
        'fig': fig
    }


visualize_sigma_func_period(sigma_func_period, square_lattice)

visualize_integrand(2*(square_lattice.omega_e+collective_lamb_shift),np.array([0,0]),square_lattice)





def visualize_polar_integrand(E, Q, lattice, n_k=100, n_theta=100, save_plots=False):
    """
    Visualize the polar integrand in the two integration regions:
    1. Light cone region: k_abs ∈ [0, k_LC]
    2. Rest region: k_abs ∈ [k_LC, k_abs_max(theta)]
    
    Parameters:
    -----------
    E : float
        Energy parameter
    Q : array-like
        Momentum vector [Qx, Qy]
    lattice : SquareLattice
        Lattice object
    n_k : int
        Number of k_abs points (default: 100)
    n_theta : int
        Number of theta points (default: 100)
    save_plots : bool
        Whether to save plots to files (default: False)
    """
    Qx, Qy = Q
    bound = np.pi / lattice.a
    k_LC = lattice.omega_e / float(c)
    
    def integrand(k_abs, theta):
        sigma1 = sigma_func_period(k_abs*np.cos(theta), k_abs*np.sin(theta), lattice)
        sigma2 = sigma_func_period(Qx - k_abs*np.cos(theta), Qy - k_abs*np.sin(theta), lattice)
        return 1 / (E - 2*lattice.omega_e - sigma1 - sigma2)
    
    def k_abs_max(theta):
        """Max radius before crossing the square BZ boundary."""
        c_val = abs(np.cos(theta))
        s_val = abs(np.sin(theta))
        r_x = bound / c_val if c_val > 1e-12 else np.inf
        r_y = bound / s_val if s_val > 1e-12 else np.inf
        return min(r_x, r_y)
    
    # Create grids
    theta_grid = np.linspace(0, 2*np.pi, n_theta)
    k_max_overall = max(k_abs_max(t) for t in theta_grid)
    k_grid = np.linspace(0, k_max_overall * 1.05, n_k)
    
    THETA, K = np.meshgrid(theta_grid, k_grid)
    
    # Compute boundary curves
    k_bz_boundary = np.array([k_abs_max(t) for t in theta_grid])
    
    # Evaluate integrand on grid (with Jacobian k_abs)
    print("Evaluating polar integrand on grid...")
    integrand_values = np.zeros_like(K, dtype=complex)
    for i, k in enumerate(k_grid):
        for j, theta in enumerate(theta_grid):
            if k <= k_abs_max(theta) and k > 0:
                integrand_values[i, j] = integrand(k, theta) * k  # Include Jacobian
            else:
                integrand_values[i, j] = np.nan
    
    real_part = integrand_values.real
    imag_part = integrand_values.imag
    magnitude = np.abs(integrand_values)
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # ========== Row 1: Polar coordinates (k_abs vs theta) ==========
    
    # 1. Real part in (k, theta) coordinates
    ax1 = plt.subplot(2, 3, 1)
    vmax_real = np.nanpercentile(np.abs(real_part), 95)
    im1 = ax1.pcolormesh(theta_grid, k_grid, real_part, cmap='RdBu_r', 
                          vmin=-vmax_real, vmax=vmax_real, shading='auto')
    ax1.axhline(y=k_LC, color='yellow', linestyle='--', linewidth=2, label=f'Light cone $k_{{LC}}={k_LC:.3f}$')
    ax1.plot(theta_grid, k_bz_boundary, 'g-', linewidth=2, label='BZ boundary')
    ax1.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax1.set_ylabel(r'$k$ (= $|k|$)', fontsize=12)
    ax1.set_title('Real part: $\\mathrm{Re}[f(k,\\theta) \\cdot k]$', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    plt.colorbar(im1, ax=ax1)
    
    # 2. Imaginary part in (k, theta) coordinates
    ax2 = plt.subplot(2, 3, 2)
    vmax_imag = np.nanpercentile(np.abs(imag_part), 95)
    im2 = ax2.pcolormesh(theta_grid, k_grid, imag_part, cmap='RdBu_r',
                          vmin=-vmax_imag, vmax=vmax_imag, shading='auto')
    ax2.axhline(y=k_LC, color='yellow', linestyle='--', linewidth=2, label=f'Light cone')
    ax2.plot(theta_grid, k_bz_boundary, 'g-', linewidth=2, label='BZ boundary')
    ax2.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax2.set_ylabel(r'$k$', fontsize=12)
    ax2.set_title('Imag part: $\\mathrm{Im}[f(k,\\theta) \\cdot k]$', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    plt.colorbar(im2, ax=ax2)
    
    # 3. Magnitude in (k, theta) coordinates
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.pcolormesh(theta_grid, k_grid, np.log10(magnitude + 1e-15), cmap='viridis', shading='auto')
    ax3.axhline(y=k_LC, color='yellow', linestyle='--', linewidth=2, label=f'Light cone')
    ax3.plot(theta_grid, k_bz_boundary, 'g-', linewidth=2, label='BZ boundary')
    ax3.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax3.set_ylabel(r'$k$', fontsize=12)
    ax3.set_title('$\\log_{10}|f(k,\\theta) \\cdot k|$', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    plt.colorbar(im3, ax=ax3, label='log₁₀|integrand|')
    
    # ========== Row 2: Cartesian view and line cuts ==========
    
    # 4. Cartesian view (kx, ky) with regions marked
    ax4 = plt.subplot(2, 3, 4)
    KX = K * np.cos(THETA)
    KY = K * np.sin(THETA)
    im4 = ax4.pcolormesh(KX, KY, real_part, cmap='RdBu_r',
                          vmin=-vmax_real, vmax=vmax_real, shading='auto')
    # Draw light cone circle
    theta_circle = np.linspace(0, 2*np.pi, 200)
    ax4.plot(k_LC * np.cos(theta_circle), k_LC * np.sin(theta_circle), 
             'y--', linewidth=2, label='Light cone')
    # Draw BZ boundary (square)
    ax4.plot([-bound, bound, bound, -bound, -bound], 
             [-bound, -bound, bound, bound, -bound], 
             'g-', linewidth=2, label='BZ boundary')
    ax4.set_xlabel(r'$k_x$', fontsize=12)
    ax4.set_ylabel(r'$k_y$', fontsize=12)
    ax4.set_title('Real part in $(k_x, k_y)$', fontsize=14, fontweight='bold')
    ax4.set_aspect('equal')
    ax4.legend(loc='upper right', fontsize=9)
    plt.colorbar(im4, ax=ax4)
    
    # 5. Line cut at theta = 0 (along kx axis)
    ax5 = plt.subplot(2, 3, 5)
    theta_idx = 0  # theta = 0
    k_max_at_theta0 = k_abs_max(0)
    valid_k = k_grid <= k_max_at_theta0
    ax5.plot(k_grid[valid_k], real_part[valid_k, theta_idx], 'b-', linewidth=2, label='Real')
    ax5.plot(k_grid[valid_k], imag_part[valid_k, theta_idx], 'r-', linewidth=2, label='Imag')
    ax5.axvline(x=k_LC, color='yellow', linestyle='--', linewidth=2, label=f'$k_{{LC}}$')
    ax5.axvline(x=k_max_at_theta0, color='green', linestyle='-', linewidth=2, label='BZ edge')
    ax5.fill_between([0, k_LC], ax5.get_ylim()[0], ax5.get_ylim()[1], alpha=0.2, color='cyan', label='LC region')
    ax5.set_xlabel(r'$k$', fontsize=12)
    ax5.set_ylabel('Integrand × k', fontsize=12)
    ax5.set_title(r'Line cut at $\theta = 0$ (along $k_x$)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Line cut at theta = pi/4 (diagonal)
    ax6 = plt.subplot(2, 3, 6)
    theta_idx_diag = n_theta // 8  # approximately pi/4
    theta_val = theta_grid[theta_idx_diag]
    k_max_at_diag = k_abs_max(theta_val)
    valid_k_diag = k_grid <= k_max_at_diag
    ax6.plot(k_grid[valid_k_diag], real_part[valid_k_diag, theta_idx_diag], 'b-', linewidth=2, label='Real')
    ax6.plot(k_grid[valid_k_diag], imag_part[valid_k_diag, theta_idx_diag], 'r-', linewidth=2, label='Imag')
    ax6.axvline(x=k_LC, color='yellow', linestyle='--', linewidth=2, label=f'$k_{{LC}}$')
    ax6.axvline(x=k_max_at_diag, color='green', linestyle='-', linewidth=2, label='BZ edge')
    ax6.set_xlabel(r'$k$', fontsize=12)
    ax6.set_ylabel('Integrand × k', fontsize=12)
    ax6.set_title(f'Line cut at $\\theta = {theta_val:.2f}$ rad ($\\approx \\pi/4$)', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('polar_integrand_visualization.png', dpi=150, bbox_inches='tight')
        print("Saved polar_integrand_visualization.png")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("POLAR INTEGRAND ANALYSIS")
    print("="*60)
    print(f"Light cone radius k_LC = ω_e/c = {k_LC:.6f}")
    print(f"BZ boundary (at θ=0): {bound:.6f}")
    print(f"BZ boundary (at θ=π/4): {k_abs_max(np.pi/4):.6f}")
    print(f"\nLight cone region: k ∈ [0, {k_LC:.4f}]")
    print(f"Rest region: k ∈ [k_LC, k_max(θ)]")
    print(f"\nReal part - max: {np.nanmax(real_part):.4e}, min: {np.nanmin(real_part):.4e}")
    print(f"Imag part - max: {np.nanmax(imag_part):.4e}, min: {np.nanmin(imag_part):.4e}")
    print("="*60)
    
    return fig


# Example usage:
visualize_polar_integrand(2*(square_lattice.omega_e+collective_lamb_shift), np.array([0,0]), square_lattice)


visualize_sigma_func_period(sigma_func_period, square_lattice)

visualize_integrand(2*(square_lattice.omega_e+collective_lamb_shift), np.array([0,0]), square_lattice)





'''