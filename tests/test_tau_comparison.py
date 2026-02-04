"""Test to compare tau_matrix_element and tau_matrix_element_polar."""

#%%
def test_tau_agreement():
    """Test whether tau_matrix_element_polar agrees with tau_matrix_element
    at multiple points in the first Brillouin zone."""
    
    print("=" * 70)
    print("Testing tau_matrix_element vs tau_matrix_element_polar")
    print("Sampling multiple points in the 1st Brillouin zone")
    print("=" * 70)
    
    # Common parameters
    E = 2 * (square_lattice.omega_e + collective_lamb_shift)
    bound = np.pi / square_lattice.a  # BZ boundary
    
    print(f"\nTest parameters:")
    print(f"  E = {E:.6f}")
    print(f"  omega_e = {square_lattice.omega_e}")
    print(f"  a = {square_lattice.a:.6f}")
    print(f"  BZ boundary = {bound:.6f}")
    print(f"  collective_lamb_shift = {collective_lamb_shift:.6f}")
    
    # Sample points in 1st BZ: high symmetry points and intermediate points
    # High symmetry points: Gamma (center), X (edge), M (corner)
    Q_points = [
        # High symmetry points
        (0.0, 0.0, "Gamma"),              # Center
        (bound, 0.0, "X"),                 # Edge center (kx direction)
        (0.0, bound, "Y"),                 # Edge center (ky direction)
        (bound, bound, "M"),               # Corner
        # Along Gamma-X path
        (bound/4, 0.0, "Gamma-X 1/4"),
        (bound/2, 0.0, "Gamma-X 1/2"),
        (3*bound/4, 0.0, "Gamma-X 3/4"),
        # Along Gamma-M diagonal
        (bound/4, bound/4, "Gamma-M 1/4"),
        (bound/2, bound/2, "Gamma-M 1/2"),
        (3*bound/4, 3*bound/4, "Gamma-M 3/4"),
        # Off-diagonal interior points
        (bound/2, bound/4, "Interior 1"),
        (bound/4, bound/2, "Interior 2"),
    ]
    
    results = []
    tolerance = 0.05
    all_passed = True
    
    print("\n" + "-" * 70)
    print(f"{'Q point':<25} {'Cartesian':<28} {'Polar':<28} {'Rel Diff':<12}")
    print("-" * 70)
    
    for Qx, Qy, label in Q_points:
        Q = np.array([Qx, Qy])
        
        print(f"\nComputing for Q = ({Qx:.4f}, {Qy:.4f}) [{label}]...")
        
        tau_cartesian = tau_matrix_element(E, Q, square_lattice)
        tau_polar = tau_matrix_element_polar(E, Q, square_lattice, n_jobs=4)
        
        abs_diff = abs(tau_cartesian - tau_polar)
        rel_diff = abs_diff / abs(tau_cartesian) if abs(tau_cartesian) > 0 else abs_diff
        passed = rel_diff < tolerance
        
        if not passed:
            all_passed = False
        
        status = "✓" if passed else "✗"
        print(f"  {label:<23} {tau_cartesian:<28} {tau_polar:<28} {rel_diff:.2e} {status}")
        
        results.append({
            'Q': Q,
            'label': label,
            'cartesian': tau_cartesian,
            'polar': tau_polar,
            'rel_diff': rel_diff,
            'passed': passed
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"  {r['label']:<16}: rel_diff = {r['rel_diff']:.2e}  {status}")
    
    max_rel_diff = max(r['rel_diff'] for r in results)
    avg_rel_diff = np.mean([r['rel_diff'] for r in results])
    
    print(f"\n  Max relative difference: {max_rel_diff:.2e}")
    print(f"  Avg relative difference: {avg_rel_diff:.2e}")
    
    if all_passed:
        print(f"\n✓ ALL TESTS PASSED (within {tolerance*100}% tolerance)")
    else:
        n_failed = sum(1 for r in results if not r['passed'])
        print(f"\n✗ {n_failed}/{len(results)} TESTS FAILED")
    
    return results


if __name__ == "__main__":
    test_tau_agreement()

# %%
