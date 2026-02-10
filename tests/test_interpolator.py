#%%
import pytest

# This module contains long-running benchmark-style helpers and functions that
# are not structured as pytest unit tests (they take non-fixture parameters).
pytest.skip("test_interpolator is a benchmark script, not a pytest unit test module", allow_module_level=True)

from joblib import Parallel, delayed
import numpy as np

from smatrix import (
    create_self_energy_interpolator,
    create_self_energy_interpolator_numba,
    square_lattice,
)



def generate_boundary_test_points(lattice, n_edge_points=1000):
    """
    Generate systematic test points focusing on BZ boundaries and special points.
    
    Returns:
    --------
    list of tuples: (kx, ky, description)
    """
    q = float(lattice.q)  # q = 2*pi/a, BZ spans [-q/2, q/2]
    bound = q / 2  # BZ boundary
    eps = 1e-8  # Small offset for near-boundary tests
    
    test_points = []
    
    # === 1. High-symmetry points ===
    # Gamma point (origin)
    test_points.append((0.0, 0.0, "Gamma (0,0)"))
    
    # X points (center of BZ edges)
    test_points.append((bound, 0.0, "X (q/2, 0)"))
    test_points.append((-bound, 0.0, "X (-q/2, 0)"))
    test_points.append((0.0, bound, "X (0, q/2)"))
    test_points.append((0.0, -bound, "X (0, -q/2)"))
    
    # M points (corners of BZ)
    test_points.append((bound, bound, "M (q/2, q/2)"))
    test_points.append((-bound, bound, "M (-q/2, q/2)"))
    test_points.append((bound, -bound, "M (q/2, -q/2)"))
    test_points.append((-bound, -bound, "M (-q/2, -q/2)"))
    
    # === 2. Points along BZ edges ===
    edge_vals = np.linspace(-bound, bound, n_edge_points)
    for val in edge_vals:
        # Right edge (kx = q/2)
        test_points.append((bound, val, f"Right edge (q/2, {val:.4f})"))
        # Left edge (kx = -q/2)
        test_points.append((-bound, val, f"Left edge (-q/2, {val:.4f})"))
        # Top edge (ky = q/2)
        test_points.append((val, bound, f"Top edge ({val:.4f}, q/2)"))
        # Bottom edge (ky = -q/2)
        test_points.append((val, -bound, f"Bottom edge ({val:.4f}, -q/2)"))
    
    # === 3. Points just inside boundary (eps inside) ===
    test_points.append((bound - eps, 0.0, "Just inside right edge"))
    test_points.append((-bound + eps, 0.0, "Just inside left edge"))
    test_points.append((0.0, bound - eps, "Just inside top edge"))
    test_points.append((0.0, -bound + eps, "Just inside bottom edge"))
    test_points.append((bound - eps, bound - eps, "Just inside M corner"))
    
    # === 4. Points just outside boundary (should wrap via periodicity) ===
    test_points.append((bound + eps, 0.0, "Just outside right edge"))
    test_points.append((-bound - eps, 0.0, "Just outside left edge"))
    test_points.append((0.0, bound + eps, "Just outside top edge"))
    test_points.append((0.0, -bound - eps, "Just outside bottom edge"))
    test_points.append((bound + eps, bound + eps, "Just outside M corner"))
    
    # === 5. Points in 2nd BZ (test periodicity) ===
    # These should map back to equivalent points in 1st BZ
    test_points.append((q + 0.1, 0.0, "2nd BZ right (q+0.1, 0)"))
    test_points.append((-q - 0.1, 0.0, "2nd BZ left (-q-0.1, 0)"))
    test_points.append((0.0, q + 0.1, "2nd BZ top (0, q+0.1)"))
    test_points.append((q + 0.1, q + 0.1, "2nd BZ corner"))
    test_points.append((1.5*q, 0.5*q, "Far outside BZ (1.5q, 0.5q)"))
    
    # === 6. Diagonal high-symmetry line (Gamma-M) ===
    diag_vals = np.linspace(0, bound, 20)
    for val in diag_vals:
        test_points.append((val, val, f"Gamma-M diagonal ({val:.4f}, {val:.4f})"))
    
    # === 7. Points along Gamma-X line ===
    for val in np.linspace(0, bound, 20):
        test_points.append((val, 0.0, f"Gamma-X line ({val:.4f}, 0)"))
    
    # === 8. Grid boundary tests (near interpolation grid edges) ===
    # The interpolation grid goes from 0 to q/2, so test near those bounds
    grid_eps = 1e-6
    test_points.append((grid_eps, grid_eps, "Near grid origin"))
    test_points.append((bound - grid_eps, bound - grid_eps, "Near grid corner"))
    
    return test_points


def test_numba_vs_original(kx_grid, ky_grid, sigma_grid, lattice, n_random=1000, tolerance=1e-2, n_jobs=-1):
    """
    Comprehensive test that Numba version agrees with original version,
    especially on BZ boundaries and special points.
    
    Parameters:
    -----------
    kx_grid, ky_grid : arrays
        Grid points used for interpolation
    sigma_grid : complex array
        Self-energy values on grid
    lattice : SquareLattice
        Lattice object
    n_random : int
        Number of random test points (in addition to systematic boundary tests)
    tolerance : float
        Maximum allowed relative difference
    n_jobs : int
        Number of parallel jobs (-1 for all CPUs)
    
    Returns:
    --------
    dict : Test results with separate sections for boundary and random tests
    """
    # Create both interpolators
    sigma_original = create_self_energy_interpolator(kx_grid, ky_grid, sigma_grid, lattice, kx=1, ky=1)
    sigma_numba = create_self_energy_interpolator_numba(kx_grid, ky_grid, sigma_grid, lattice)
    
    q = float(lattice.q)
    
    def evaluate_point(kx, ky):
        """Evaluate both interpolators at a point and compare."""
        val_orig = sigma_original(kx, ky, lattice)
        val_numba = sigma_numba(kx, ky)
        
        # Handle array output from original
        if hasattr(val_orig, '__len__'):
            val_orig = complex(val_orig.flatten()[0])
        
        abs_diff = abs(val_orig - val_numba)
        rel_diff = abs_diff / (abs(val_orig) + 1e-15)
        
        return val_orig, val_numba, abs_diff, rel_diff
    
    # ========== PART 1: Systematic Boundary Tests ==========
    print("=" * 80)
    print("PART 1: SYSTEMATIC BOUNDARY AND SPECIAL POINT TESTS")
    print("=" * 80)
    
    boundary_points = generate_boundary_test_points(lattice, n_edge_points=30)
    n_boundary = len(boundary_points)
    
    print(f"Testing {n_boundary} boundary/special points...")
    print(f"  BZ boundary: q/2 = {q/2:.6f}")
    print(f"  Tolerance: {tolerance}")
    print("-" * 80)
    
    boundary_results = {
        'passed': 0,
        'failed': 0,
        'max_rel_diff': 0.0,
        'max_abs_diff': 0.0,
        'failures': []
    }
    
    # Run boundary tests (sequential for detailed output)
    for i, (kx, ky, desc) in enumerate(boundary_points):
        val_orig, val_numba, abs_diff, rel_diff = evaluate_point(kx, ky)
        
        boundary_results['max_abs_diff'] = max(boundary_results['max_abs_diff'], abs_diff)
        boundary_results['max_rel_diff'] = max(boundary_results['max_rel_diff'], rel_diff)
        
        if rel_diff < tolerance:
            boundary_results['passed'] += 1
            status = "PASS"
        else:
            boundary_results['failed'] += 1
            boundary_results['failures'].append({
                'kx': kx, 'ky': ky, 'desc': desc,
                'orig': val_orig, 'numba': val_numba,
                'abs_diff': abs_diff, 'rel_diff': rel_diff
            })
            status = "FAIL"
        
        # Print special points and failures
        if "Gamma" in desc or "X (" in desc or "M (" in desc or status == "FAIL":
            print(f"  {desc:40s} | rel_diff={rel_diff:.2e} [{status}]")
    
    print("-" * 80)
    print(f"Boundary tests: {boundary_results['passed']}/{n_boundary} passed")
    print(f"  Max absolute diff: {boundary_results['max_abs_diff']:.2e}")
    print(f"  Max relative diff: {boundary_results['max_rel_diff']:.2e}")
    
    if boundary_results['failures']:
        print(f"\n  FAILURES ({len(boundary_results['failures'])}):")
        for f in boundary_results['failures'][:10]:  # Show first 10 failures
            print(f"    {f['desc']}: k=({f['kx']:.6f}, {f['ky']:.6f})")
            print(f"      orig={f['orig']:.6e}, numba={f['numba']:.6e}, rel_diff={f['rel_diff']:.2e}")
    
    # ========== PART 2: Periodicity Tests ==========
    print("\n" + "=" * 80)
    print("PART 2: PERIODICITY TESTS")
    print("=" * 80)
    print("Verifying f(k) = f(k + G) for reciprocal lattice vectors G")
    print("-" * 80)
    
    periodicity_results = {'passed': 0, 'failed': 0, 'failures': []}
    
    # Test that equivalent points give same values
    test_k_points = [(0.1, 0.2), (0.3, 0.1), (-0.2, 0.15), (q/4, q/4)]
    G_vectors = [(q, 0), (0, q), (q, q), (-q, 0), (0, -q), (-q, -q), (2*q, 0), (0, 2*q)]
    
    for kx, ky in test_k_points:
        val_base_orig, val_base_numba, _, _ = evaluate_point(kx, ky)
        
        for Gx, Gy in G_vectors:
            kx_shifted, ky_shifted = kx + Gx, ky + Gy
            val_shift_orig, val_shift_numba, _, _ = evaluate_point(kx_shifted, ky_shifted)
            
            # Check original periodicity
            diff_orig = abs(val_base_orig - val_shift_orig) / (abs(val_base_orig) + 1e-15)
            # Check numba periodicity
            diff_numba = abs(val_base_numba - val_shift_numba) / (abs(val_base_numba) + 1e-15)
            
            if diff_orig < tolerance and diff_numba < tolerance:
                periodicity_results['passed'] += 1
                status = "PASS"
            else:
                periodicity_results['failed'] += 1
                periodicity_results['failures'].append({
                    'k': (kx, ky), 'G': (Gx, Gy),
                    'diff_orig': diff_orig, 'diff_numba': diff_numba
                })
                status = "FAIL"
                print(f"  k=({kx:.3f},{ky:.3f}) + G=({Gx:.3f},{Gy:.3f}): "
                      f"diff_orig={diff_orig:.2e}, diff_numba={diff_numba:.2e} [{status}]")
    
    total_periodicity = periodicity_results['passed'] + periodicity_results['failed']
    print(f"\nPeriodicity tests: {periodicity_results['passed']}/{total_periodicity} passed")
    
    # ========== PART 3: Random Tests (Parallel) ==========
    print("\n" + "=" * 80)
    print("PART 3: RANDOM POINT TESTS (PARALLEL)")
    print("=" * 80)
    
    np.random.seed(42)
    k_range = q * 1.5
    kx_test = np.random.uniform(-k_range, k_range, n_random)
    ky_test = np.random.uniform(-k_range, k_range, n_random)
    
    print(f"Testing {n_random} random points in parallel...")
    print(f"  k range: [-{k_range:.4f}, {k_range:.4f}]")
    print(f"  n_jobs: {n_jobs}")
    print("-" * 80)
    
    def evaluate_single_point_parallel(i, kx, ky):
        """Wrapper for parallel evaluation."""
        val_orig, val_numba, abs_diff, rel_diff = evaluate_point(kx, ky)
        return {
            'index': i, 'kx': kx, 'ky': ky,
            'val_orig': val_orig, 'val_numba': val_numba,
            'abs_diff': abs_diff, 'rel_diff': rel_diff
        }
    
    # Run in parallel
    parallel_results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(evaluate_single_point_parallel)(i, kx_test[i], ky_test[i])
        for i in range(n_random)
    )
    
    random_results = {
        'passed': 0,
        'failed': 0,
        'max_rel_diff': 0.0,
        'max_abs_diff': 0.0,
        'failures': []
    }
    
    for res in parallel_results:
        random_results['max_abs_diff'] = max(random_results['max_abs_diff'], res['abs_diff'])
        random_results['max_rel_diff'] = max(random_results['max_rel_diff'], res['rel_diff'])
        
        if res['rel_diff'] < tolerance:
            random_results['passed'] += 1
        else:
            random_results['failed'] += 1
            random_results['failures'].append(res)
    
    print(f"\nRandom tests: {random_results['passed']}/{n_random} passed")
    print(f"  Max absolute diff: {random_results['max_abs_diff']:.2e}")
    print(f"  Max relative diff: {random_results['max_rel_diff']:.2e}")
    
    if random_results['failures']:
        print(f"\n  First 5 failures:")
        for f in random_results['failures'][:5]:
            print(f"    k=({f['kx']:.6f}, {f['ky']:.6f}): rel_diff={f['rel_diff']:.2e}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    total_passed = boundary_results['passed'] + periodicity_results['passed'] + random_results['passed']
    total_tests = n_boundary + total_periodicity + n_random
    total_failed = boundary_results['failed'] + periodicity_results['failed'] + random_results['failed']
    
    print(f"  Boundary tests:    {boundary_results['passed']}/{n_boundary} passed")
    print(f"  Periodicity tests: {periodicity_results['passed']}/{total_periodicity} passed")
    print(f"  Random tests:      {random_results['passed']}/{n_random} passed")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL:             {total_passed}/{total_tests} passed ({100*total_passed/total_tests:.1f}%)")
    
    overall_max_rel = max(boundary_results['max_rel_diff'], random_results['max_rel_diff'])
    overall_max_abs = max(boundary_results['max_abs_diff'], random_results['max_abs_diff'])
    print(f"\n  Overall max relative diff: {overall_max_rel:.2e}")
    print(f"  Overall max absolute diff: {overall_max_abs:.2e}")
    
    if total_failed == 0:
        print("\n✓ All tests passed! Numba version agrees with original everywhere.")
    else:
        print(f"\n✗ {total_failed} test(s) failed")
    
    return {
        'boundary': boundary_results,
        'periodicity': periodicity_results,
        'random': random_results,
        'total_passed': total_passed,
        'total_tests': total_tests
    }


#%%
if __name__ == "__main__":
    # Load the data
    data = np.load("data/sigma_grid0f4a.npz")
    kx = data["kx"]
    ky = data["ky"]
    sigma_grid = data["sigma_grid"]
    
    # Run comprehensive test
    results = test_numba_vs_original(kx, ky, sigma_grid, square_lattice, n_random=20000)



#%%
"""
Benchmark: sigma_func_period (scipy) vs sigma_func_period_numba (numba)
"""

import time



def benchmark_interpolators(kx_grid, ky_grid, sigma_grid, lattice, n_calls=10000):
    """
    Benchmark the original vs Numba interpolator.
    
    Parameters:
    -----------
    n_calls : int
        Number of function calls for timing
    """
    # Create both interpolators
    sigma_original = create_self_energy_interpolator(kx_grid, ky_grid, sigma_grid, lattice, kx=1, ky=1)
    sigma_numba = create_self_energy_interpolator_numba(kx_grid, ky_grid, sigma_grid, lattice)
    
    # Generate random test points
    np.random.seed(42)
    q = float(lattice.q)
    k_range = q * 1.5
    kx_test = np.random.uniform(-k_range, k_range, n_calls)
    ky_test = np.random.uniform(-k_range, k_range, n_calls)
    
    print(f"Benchmarking interpolators with {n_calls:,} function calls")
    print(f"  k range: [-{k_range:.4f}, {k_range:.4f}]")
    print("=" * 60)
    
    # Warm-up for Numba (JIT compilation)
    print("\nWarming up Numba (JIT compilation)...")
    for i in range(100):
        _ = sigma_numba(kx_test[i], ky_test[i])
    print("  Done.")
    
    # Benchmark original (scipy RectBivariateSpline)
    print("\nBenchmarking original (scipy RectBivariateSpline)...")
    start = time.perf_counter()
    for i in range(n_calls):
        _ = sigma_original(kx_test[i], ky_test[i], lattice)
    time_original = time.perf_counter() - start
    print(f"  Total time: {time_original:.4f} s")
    print(f"  Per call:   {time_original/n_calls*1e6:.2f} µs")
    
    # Benchmark Numba
    print("\nBenchmarking Numba (bilinear interpolation)...")
    start = time.perf_counter()
    for i in range(n_calls):
        _ = sigma_numba(kx_test[i], ky_test[i])
    time_numba = time.perf_counter() - start
    print(f"  Total time: {time_numba:.4f} s")
    print(f"  Per call:   {time_numba/n_calls*1e6:.2f} µs")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    speedup = time_original / time_numba
    print(f"  Original (scipy):  {time_original:.4f} s  ({time_original/n_calls*1e6:.2f} µs/call)")
    print(f"  Numba:             {time_numba:.4f} s  ({time_numba/n_calls*1e6:.2f} µs/call)")
    print(f"  Speedup:           {speedup:.1f}x faster with Numba")
    
    return {
        'time_original': time_original,
        'time_numba': time_numba,
        'speedup': speedup,
        'n_calls': n_calls
    }


#%%
if __name__ == "__main__":
    # Load the data
    data = np.load("data/sigma_grid0f4a.npz")
    kx = data["kx"]
    ky = data["ky"]
    sigma_grid = data["sigma_grid"]
    
    # Run benchmark
    results = benchmark_interpolators(kx, ky, sigma_grid, square_lattice, n_calls=10000)

# %%
