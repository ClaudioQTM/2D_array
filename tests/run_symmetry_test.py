#%%
"""
Test script to check self-energy reflection symmetry along x and y axes in 1st BZ.
Run this from your Anaconda environment.
"""

from model import self_energy, EMField, SquareLattice
import numpy as np
from joblib import Parallel, delayed


def compute_self_energy_point(kx, ky, a, d, omega_e, alpha):
    """Helper function for parallel computation of self-energy."""
    return self_energy(kx, ky, a, d, omega_e, omega_e, alpha)


def test_self_energy_reflection_symmetry_parallel(lattice, alpha=1e-4, tolerance=1e-10, n_tests=50, n_jobs=-1):
    """
    Test whether self_energy is symmetric with respect to reflection along x and y axes
    inside the 1st Brillouin Zone. Uses parallel evaluation for speed.
    
    Tests:
    - x-axis reflection: sigma(kx, ky) = sigma(kx, -ky)
    - y-axis reflection: sigma(kx, ky) = sigma(-kx, ky)
    
    Parameters:
        lattice: SquareLattice instance
        alpha: regularization parameter
        tolerance: maximum allowed difference
        n_tests: number of random test points
        n_jobs: number of parallel jobs (-1 = use all cores)
    """
    np.random.seed(42)  # reproducibility
    
    # k range within the first Brillouin zone: |k| <= pi/a (including boundary)
    k_max = float(np.pi / lattice.a)  # include BZ boundary
    
    # Generate all random k-points
    kx_vals = np.random.uniform(0, k_max, n_tests)
    ky_vals = np.random.uniform(0, k_max, n_tests)
    
    # Build list of all points to evaluate:
    # For each test point (kx, ky), we need: (kx, ky), (kx, -ky), (-kx, ky)
    all_points = []
    for i in range(n_tests):
        kx, ky = kx_vals[i], ky_vals[i]
        all_points.append((kx, ky))      # original
        all_points.append((kx, -ky))     # x-axis reflection
        all_points.append((-kx, ky))     # y-axis reflection
    
    print(f"Testing self-energy reflection symmetry in 1st BZ")
    print(f"  n_tests = {n_tests}, tolerance = {tolerance}")
    print(f"  Total self-energy evaluations: {len(all_points)}")
    print(f"  Using {n_jobs} parallel jobs (-1 = all cores)")
    print("="*80)
    print("Computing self-energy values in parallel...")
    
    # Parallel computation of all self-energy values
    results_parallel = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(compute_self_energy_point)(kx, ky, lattice.a, lattice.d, lattice.omega_e, alpha)
        for kx, ky in all_points
    )
    
    # Parse results
    results = {
        'x_reflection': {'passed': 0, 'failed': 0, 'diffs': []},
        'y_reflection': {'passed': 0, 'failed': 0, 'diffs': []},
        'failures': []
    }
    
    print("\n" + "="*80)
    print("DETAILED RESULTS:")
    print("-"*80)
    
    for i in range(n_tests):
        kx, ky = kx_vals[i], ky_vals[i]
        idx = i * 3
        sigma_orig = results_parallel[idx]
        sigma_x_ref = results_parallel[idx + 1]
        sigma_y_ref = results_parallel[idx + 2]
        
        # Check x-axis reflection symmetry
        diff_x = abs(sigma_orig - sigma_x_ref)
        results['x_reflection']['diffs'].append(diff_x)
        status_x = "PASS" if diff_x < tolerance else "FAIL"
        if diff_x < tolerance:
            results['x_reflection']['passed'] += 1
        else:
            results['x_reflection']['failed'] += 1
            results['failures'].append({
                'type': 'x_reflection', 'kx': kx, 'ky': ky,
                'sigma_orig': sigma_orig, 'sigma_ref': sigma_x_ref, 'diff': diff_x
            })
        
        # Check y-axis reflection symmetry
        diff_y = abs(sigma_orig - sigma_y_ref)
        results['y_reflection']['diffs'].append(diff_y)
        status_y = "PASS" if diff_y < tolerance else "FAIL"
        if diff_y < tolerance:
            results['y_reflection']['passed'] += 1
        else:
            results['y_reflection']['failed'] += 1
            results['failures'].append({
                'type': 'y_reflection', 'kx': kx, 'ky': ky,
                'sigma_orig': sigma_orig, 'sigma_ref': sigma_y_ref, 'diff': diff_y
            })
        
        print(f"Test {i+1:3d}: k=({kx:.4f}, {ky:.4f})")
        print(f"         sigma(kx, ky)  = {sigma_orig:.6e}")
        print(f"         sigma(kx,-ky)  = {sigma_x_ref:.6e}  diff={diff_x:.2e} [{status_x}]")
        print(f"         sigma(-kx,ky)  = {sigma_y_ref:.6e}  diff={diff_y:.2e} [{status_y}]")
    
    print("\n" + "="*80)
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
        print("\n[PASS] SYMMETRY CONFIRMED: self_energy has reflection symmetry along both x and y axes!")
    else:
        print(f"\n[FAIL] SYMMETRY BROKEN: {total_tests - total_passed} tests failed")
        if results['failures']:
            print("\nFirst few failures:")
            for f in results['failures'][:5]:
                print(f"  {f['type']}: k=({f['kx']:.4f}, {f['ky']:.4f}), diff={f['diff']:.2e}")
    
    return results


if __name__ == "__main__":
    # Create a lattice instance for testing
    test_field = EMField()
    test_lattice = SquareLattice(
        a=0.4*2*np.pi, 
        omega_e=1, 
        dipole_vector=np.array([1, 1j, 0])/np.sqrt(2), 
        field=test_field
    )

    # Run the symmetry test with 50 samples per axis, using all CPU cores
    print("Testing self-energy reflection symmetry (PARALLEL)...")
    print()
    results = test_self_energy_reflection_symmetry_parallel(
        test_lattice, 
        alpha=1e-4, 
        tolerance=1e-8, 
        n_tests=50,
        n_jobs=-1  # use all available CPU cores
    )
