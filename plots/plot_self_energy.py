import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
    # Create meshgrid for plotting (using 'ij' indexing to match sigma_grid shape)
    KX, KY = np.meshgrid(kx_grid, ky_grid, indexing='ij')
    
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
        # Save in the plots directory
        plots_dir = Path(__file__).parent
        filename = plots_dir / 'sigma_grid_visualization.png'
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


if __name__ == "__main__":
    # Load data from npz file
    data_path = Path(__file__).parent.parent / "data" / "sigma_grid0f4a_patched.npz"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file exists.")
    else:
        print(f"Loading data from {data_path}")
        data = np.load(data_path)
        kx = data["kx"]
        ky = data["ky"]
        sigma_grid = data["sigma_grid"]
        
        print(f"kx shape: {kx.shape}")
        print(f"ky shape: {ky.shape}")
        print(f"sigma_grid shape: {sigma_grid.shape}")
        print(f"sigma_grid dtype: {sigma_grid.dtype}")
        
        # Create plots
        plot_sigma_grid(kx, ky, sigma_grid, save_plots=True, figsize=(16, 4))