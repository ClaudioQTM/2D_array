"""
Test to verify that the new vectorized D_bounds function
produces the same results as the old scalar-only implementation.
"""

import numpy as np
import sys
from smatrix import (
    square_lattice,
    collective_lamb_shift,
    create_self_energy_interpolator_numba,
)
from input_states import gaussian_in_state
from scattering import _make_integrand_and_bounds


def old_D_bounds_scalar(Dpx, Dpy, COM_K, G, H, E):
    """
    Old D_bounds implementation that only works with scalar inputs.
    This is the original version before vectorization.
    """
    Dp = np.array([Dpx, Dpy])
    q_para = COM_K + Dp / 2 + G
    l_para = COM_K - Dp / 2 + H
    D_min = np.linalg.norm(q_para) - E / 2
    D_max = E / 2 - np.linalg.norm(l_para)
    return [D_min, D_max]


def test_scalar_inputs():
    """Test D_bounds with scalar inputs."""
    print("\n" + "=" * 60)
    print("Test 1: Scalar Inputs")
    print("=" * 60)

    # Setup parameters
    E = 2.0 + collective_lamb_shift
    lattice = square_lattice
    in_state = gaussian_in_state

    # Get the new vectorized D_bounds function
    sigma_func_period = create_self_energy_interpolator_numba(
        np.array([0.0, float(lattice.q / 2)]),
        np.array([0.0, float(lattice.q / 2)]),
        np.zeros((2, 2), dtype=np.complex128),
        lattice=lattice,
    )
    integrand, D_bounds_new = _make_integrand_and_bounds(
        E, lattice, in_state, sigma_func_period
    )

    # Test parameters
    COM_K = np.array([0.1, 0.2])
    G = np.array([0.0, 0.0])
    H = np.array([0.0, 0.0])

    # Test cases with scalar inputs
    test_cases = [
        (0.5, 0.3),
        (0.0, 0.0),
        (0.1, 0.4),
        (-0.2, 0.3),
        (0.8, -0.5),
    ]

    all_passed = True
    for Dpx, Dpy in test_cases:
        # Old implementation
        D_min_old, D_max_old = old_D_bounds_scalar(Dpx, Dpy, COM_K, G, H, E)

        # New vectorized implementation
        D_min_new, D_max_new = D_bounds_new(Dpx, Dpy, COM_K, G, H)

        # Check if results match
        match = np.allclose(D_min_old, D_min_new) and np.allclose(D_max_old, D_max_new)

        status = "✓ PASS" if match else "✗ FAIL"
        print(f"\nDpx={Dpx:6.2f}, Dpy={Dpy:6.2f}")
        print(f"  Old: D_min={D_min_old:8.5f}, D_max={D_max_old:8.5f}")
        print(f"  New: D_min={D_min_new:8.5f}, D_max={D_max_new:8.5f}")
        print(f"  {status}")

        if not match:
            all_passed = False
            print(
                f"  Difference: D_min={abs(D_min_old - D_min_new):.2e}, D_max={abs(D_max_old - D_max_new):.2e}"
            )

    return all_passed


def test_array_inputs():
    """Test D_bounds with array inputs and verify vectorization."""
    print("\n" + "=" * 60)
    print("Test 2: Array Inputs (Vectorization Test)")
    print("=" * 60)

    # Setup parameters
    E = 2.0 + collective_lamb_shift
    lattice = square_lattice
    in_state = gaussian_in_state

    # Get the new vectorized D_bounds function
    sigma_func_period = create_self_energy_interpolator_numba(
        np.array([0.0, float(lattice.q / 2)]),
        np.array([0.0, float(lattice.q / 2)]),
        np.zeros((2, 2), dtype=np.complex128),
        lattice=lattice,
    )
    integrand, D_bounds_new = _make_integrand_and_bounds(
        E, lattice, in_state, sigma_func_period
    )

    # Test parameters
    COM_K = np.array([0.1, 0.2])
    G = np.array([0.0, 0.0])
    H = np.array([0.0, 0.0])

    # Create array inputs
    n_samples = 100
    np.random.seed(42)
    Dpx_array = np.random.uniform(-1.0, 1.0, n_samples)
    Dpy_array = np.random.uniform(-1.0, 1.0, n_samples)

    print(f"\nTesting with {n_samples} samples...")

    # Compute using vectorized function (should be fast)
    D_min_vec, D_max_vec = D_bounds_new(Dpx_array, Dpy_array, COM_K, G, H)

    # Compute using loop over old scalar function (slow reference)
    D_min_loop = np.zeros(n_samples)
    D_max_loop = np.zeros(n_samples)
    for i in range(n_samples):
        D_min_loop[i], D_max_loop[i] = old_D_bounds_scalar(
            Dpx_array[i], Dpy_array[i], COM_K, G, H, E
        )

    # Check if results match
    min_match = np.allclose(D_min_vec, D_min_loop)
    max_match = np.allclose(D_max_vec, D_max_loop)

    print(f"\nD_min arrays match: {min_match}")
    print(f"D_max arrays match: {max_match}")

    if min_match and max_match:
        print("\n✓ PASS: Vectorized function produces identical results to scalar loop")

        # Show some sample values
        print("\nSample values (first 5):")
        for i in range(min(5, n_samples)):
            print(f"  Sample {i}: Dpx={Dpx_array[i]:6.3f}, Dpy={Dpy_array[i]:6.3f}")
            print(f"    D_min: vec={D_min_vec[i]:8.5f}, loop={D_min_loop[i]:8.5f}")
            print(f"    D_max: vec={D_max_vec[i]:8.5f}, loop={D_max_loop[i]:8.5f}")
    else:
        print("\n✗ FAIL: Arrays do not match")
        max_diff_min = np.max(np.abs(D_min_vec - D_min_loop))
        max_diff_max = np.max(np.abs(D_max_vec - D_max_loop))
        print(f"  Max difference in D_min: {max_diff_min:.2e}")
        print(f"  Max difference in D_max: {max_diff_max:.2e}")

    return min_match and max_match


def test_consistency():
    """Test that array input at index 0 matches scalar input with same values."""
    print("\n" + "=" * 60)
    print("Test 3: Consistency Between Scalar and Array Modes")
    print("=" * 60)

    # Setup parameters
    E = 2.0 + collective_lamb_shift
    lattice = square_lattice
    in_state = gaussian_in_state

    # Get the new vectorized D_bounds function
    sigma_func_period = create_self_energy_interpolator_numba(
        np.array([0.0, float(lattice.q / 2)]),
        np.array([0.0, float(lattice.q / 2)]),
        np.zeros((2, 2), dtype=np.complex128),
        lattice=lattice,
    )
    integrand, D_bounds_new = _make_integrand_and_bounds(
        E, lattice, in_state, sigma_func_period
    )

    # Test parameters
    COM_K = np.array([0.1, 0.2])
    G = np.array([0.0, 0.0])
    H = np.array([0.0, 0.0])

    # Test values
    Dpx_scalar = 0.5
    Dpy_scalar = 0.3

    # Call with scalar inputs
    D_min_scalar, D_max_scalar = D_bounds_new(Dpx_scalar, Dpy_scalar, COM_K, G, H)

    # Call with array inputs (single element)
    D_min_array, D_max_array = D_bounds_new(
        np.array([Dpx_scalar]), np.array([Dpy_scalar]), COM_K, G, H
    )

    # Check consistency
    min_match = np.allclose(D_min_scalar, D_min_array[0])
    max_match = np.allclose(D_max_scalar, D_max_array[0])

    print(f"\nScalar mode: D_min={D_min_scalar:8.5f}, D_max={D_max_scalar:8.5f}")
    print(f"Array mode:  D_min={D_min_array[0]:8.5f}, D_max={D_max_array[0]:8.5f}")

    if min_match and max_match:
        print("\n✓ PASS: Scalar and array modes are consistent")
    else:
        print("\n✗ FAIL: Scalar and array modes are inconsistent")

    return min_match and max_match


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "=" * 60)
    print("Test 4: Edge Cases")
    print("=" * 60)

    # Setup parameters
    E = 2.0 + collective_lamb_shift
    lattice = square_lattice
    in_state = gaussian_in_state

    # Get the new vectorized D_bounds function
    sigma_func_period = create_self_energy_interpolator_numba(
        np.array([0.0, float(lattice.q / 2)]),
        np.array([0.0, float(lattice.q / 2)]),
        np.zeros((2, 2), dtype=np.complex128),
        lattice=lattice,
    )
    integrand, D_bounds_new = _make_integrand_and_bounds(
        E, lattice, in_state, sigma_func_period
    )

    COM_K = np.array([0.1, 0.2])
    G = np.array([0.0, 0.0])
    H = np.array([0.0, 0.0])

    all_passed = True

    # Test 1: Zero inputs
    print("\nTest 4a: Zero inputs")
    D_min_old, D_max_old = old_D_bounds_scalar(0.0, 0.0, COM_K, G, H, E)
    D_min_new, D_max_new = D_bounds_new(0.0, 0.0, COM_K, G, H)
    match = np.allclose(D_min_old, D_min_new) and np.allclose(D_max_old, D_max_new)
    print(f"  Old: D_min={D_min_old:8.5f}, D_max={D_max_old:8.5f}")
    print(f"  New: D_min={D_min_new:8.5f}, D_max={D_max_new:8.5f}")
    print(f"  {'✓ PASS' if match else '✗ FAIL'}")
    all_passed = all_passed and match

    # Test 2: Large values
    print("\nTest 4b: Large values")
    D_min_old, D_max_old = old_D_bounds_scalar(10.0, 10.0, COM_K, G, H, E)
    D_min_new, D_max_new = D_bounds_new(10.0, 10.0, COM_K, G, H)
    match = np.allclose(D_min_old, D_min_new) and np.allclose(D_max_old, D_max_new)
    print(f"  Old: D_min={D_min_old:8.5f}, D_max={D_max_old:8.5f}")
    print(f"  New: D_min={D_min_new:8.5f}, D_max={D_max_new:8.5f}")
    print(f"  {'✓ PASS' if match else '✗ FAIL'}")
    all_passed = all_passed and match

    # Test 3: Negative values
    print("\nTest 4c: Negative values")
    D_min_old, D_max_old = old_D_bounds_scalar(-0.5, -0.5, COM_K, G, H, E)
    D_min_new, D_max_new = D_bounds_new(-0.5, -0.5, COM_K, G, H)
    match = np.allclose(D_min_old, D_min_new) and np.allclose(D_max_old, D_max_new)
    print(f"  Old: D_min={D_min_old:8.5f}, D_max={D_max_old:8.5f}")
    print(f"  New: D_min={D_min_new:8.5f}, D_max={D_max_new:8.5f}")
    print(f"  {'✓ PASS' if match else '✗ FAIL'}")
    all_passed = all_passed and match

    # Test 4: Empty array
    print("\nTest 4d: Empty array")
    try:
        D_min_new, D_max_new = D_bounds_new(np.array([]), np.array([]), COM_K, G, H)
        print(f"  Result: D_min shape={D_min_new.shape}, D_max shape={D_max_new.shape}")
        print("  ✓ PASS: Handles empty arrays without crashing")
    except Exception as e:
        print(f"  ✗ FAIL: Exception raised: {e}")
        all_passed = False

    return all_passed


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Testing Vectorized D_bounds Function")
    print("=" * 70)

    # Run all tests
    test_results = []

    test_results.append(("Scalar Inputs", test_scalar_inputs()))
    test_results.append(("Array Inputs", test_array_inputs()))
    test_results.append(("Consistency", test_consistency()))
    test_results.append(("Edge Cases", test_edge_cases()))

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    for test_name, passed in test_results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s}: {status}")

    all_passed = all([result[1] for result in test_results])

    print("=" * 70)
    if all_passed:
        print("  All tests PASSED! ✓")
        print("  The vectorized D_bounds function is working correctly.")
    else:
        print("  Some tests FAILED! ✗")
        print("  Please review the failures above.")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)

# %%
