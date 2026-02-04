#%%
"""
Test script to compare nquad (quadrature) vs QMC integration methods
for the scattering integral calculation.
"""

import numpy as np
import time

# Import from S_matrix
from S_matrix import (
    scattering_integral,
    scattering_integral_qmc,
    square_lattice,
    collective_lamb_shift,
    gaussian_in_state
)


def test_single_point(k_para, p_para, Ek, Ep, name="Test", run_nquad=True, qmc_m=13):
    """Test a single parameter set with both QMC and optionally nquad.
    
    Parameters
    ----------
    k_para, p_para : array
        Parallel momentum vectors
    Ek, Ep : float
        Energies
    name : str
        Name for this test case
    run_nquad : bool
        Whether to run nquad (slow)
    qmc_m : int
        QMC sample exponent (2^m samples)
    
    Returns
    -------
    dict with 'qmc' and 'nquad' results
    """
    result = {'name': name, 'k_para': k_para, 'p_para': p_para, 'Ek': Ek, 'Ep': Ep}
    
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"  k_para = {k_para}")
    print(f"  p_para = {p_para}")
    print(f"  Ek = {Ek:.6f}, Ep = {Ep:.6f}")
    print(f"  Total E = {Ek + Ep:.6f}")
    
    # QMC integration
    print(f"\n  QMC (m={qmc_m}, {2**qmc_m} samples)...")
    try:
        start = time.time()
        qmc_result = scattering_integral_qmc(
            k_para, Ek, p_para, Ep, square_lattice, gaussian_in_state,
            m=qmc_m, seed=42
        )
        qmc_time = time.time() - start
        result['qmc'] = {'result': qmc_result, 'time': qmc_time, 'success': True}
        print(f"    Result: {qmc_result}")
        print(f"    Magnitude: {abs(qmc_result):.6e}")
        print(f"    Time: {qmc_time:.3f}s")
    except Exception as e:
        result['qmc'] = {'success': False, 'error': str(e)}
        print(f"    ERROR: {e}")
    
    # nquad integration (optional)
    if run_nquad:
        print(f"\n  nquad (parallelized)...")
        try:
            start = time.time()
            nquad_result = scattering_integral(
                k_para, Ek, p_para, Ep, square_lattice, gaussian_in_state
            )
            nquad_time = time.time() - start
            result['nquad'] = {'result': nquad_result, 'time': nquad_time, 'success': True}
            print(f"    Result: {nquad_result}")
            print(f"    Magnitude: {abs(nquad_result):.6e}")
            print(f"    Time: {nquad_time:.3f}s")
            
            # Compare
            if result['qmc'].get('success'):
                rel_err = abs(abs(qmc_result) - abs(nquad_result)) / (abs(nquad_result) + 1e-15)
                print(f"\n  COMPARISON:")
                print(f"    Relative error (magnitude): {rel_err:.6e}")
                print(f"    Agreement: {'GOOD' if rel_err < 0.1 else 'POOR'}")
                result['rel_error'] = rel_err
        except Exception as e:
            result['nquad'] = {'success': False, 'error': str(e)}
            print(f"    ERROR: {e}")
    
    return result


def run_multi_point_comparison(run_nquad=False, qmc_m=13):
    """Run comparison tests at multiple k_para, p_para, Ek, Ep values.
    
    Parameters
    ----------
    run_nquad : bool
        Whether to run nquad for each test (very slow!)
    qmc_m : int
        QMC sample exponent
    """
    bound = np.pi / square_lattice.a
    E_resonance = square_lattice.omega_e + collective_lamb_shift
    
    # Define test cases: (k_para, p_para, Ek, Ep, name)
    test_cases = [
        # Case 1: Origin
        (np.array([0.0, 0.0]), np.array([0.0, 0.0]), 
         E_resonance + 0.01, E_resonance + 0.01, 
         "Origin (k=0, p=0)"),
        
        # Case 2: k shifted in x
        (np.array([0.1, 0.0]), np.array([0.0, 0.0]), 
         E_resonance + 0.01, E_resonance + 0.01, 
         "k shifted in x"),
        
        # Case 3: Both k and p shifted
        (np.array([0.1, 0.0]), np.array([-0.1, 0.0]), 
         E_resonance + 0.01, E_resonance + 0.01, 
         "k and p shifted opposite"),
        
        # Case 4: Diagonal shift
        (np.array([0.1, 0.1]), np.array([0.0, 0.0]), 
         E_resonance + 0.01, E_resonance + 0.01, 
         "k diagonal shift"),
        
        # Case 5: Different energies
        (np.array([0.0, 0.0]), np.array([0.0, 0.0]), 
         E_resonance + 0.02, E_resonance + 0.01, 
         "Different Ek, Ep"),
        
        # Case 6: Near BZ edge
        (np.array([bound * 0.3, 0.0]), np.array([0.0, 0.0]), 
         E_resonance + 0.01, E_resonance + 0.01, 
         "Near BZ edge (30%)"),
    ]
    
    print("=" * 70)
    print("MULTI-POINT COMPARISON TEST: nquad vs QMC")
    print("=" * 70)
    print(f"Lattice: a = {square_lattice.a:.4f}, omega_e = {square_lattice.omega_e}")
    print(f"Collective Lamb shift = {collective_lamb_shift:.6f}")
    print(f"E_resonance = {E_resonance:.6f}")
    print(f"BZ boundary = {bound:.4f}")
    print(f"QMC samples: 2^{qmc_m} = {2**qmc_m}")
    print(f"Running nquad: {run_nquad}")
    
    all_results = []
    
    for k_para, p_para, Ek, Ep, name in test_cases:
        result = test_single_point(k_para, p_para, Ek, Ep, name, run_nquad=run_nquad, qmc_m=qmc_m)
        all_results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test Name':<30} | {'QMC Mag':>12} | {'nquad Mag':>12} | {'Rel Err':>10} | {'Status':>8}")
    print("-" * 80)
    
    for r in all_results:
        name = r['name'][:30]
        qmc_mag = f"{abs(r['qmc']['result']):.4e}" if r['qmc'].get('success') else "ERROR"
        nquad_mag = f"{abs(r['nquad']['result']):.4e}" if r.get('nquad', {}).get('success') else "N/A"
        rel_err = f"{r['rel_error']:.2e}" if 'rel_error' in r else "N/A"
        status = "GOOD" if r.get('rel_error', 1) < 0.1 else ("N/A" if 'rel_error' not in r else "POOR")
        print(f"{name:<30} | {qmc_mag:>12} | {nquad_mag:>12} | {rel_err:>10} | {status:>8}")
    
    return all_results


def run_comparison_test(run_nquad=True):
    """Run comparison tests between nquad and QMC methods (original function).
    
    Parameters
    ----------
    run_nquad : bool
        If True, also run the slow nquad integration (default: True)
    """
    
    # Test parameters - Origin of the first Brillouin zone
    k_para = np.array([0.0, 0.0])  # Origin point
    p_para = np.array([0.0, 0.0])  # Origin point
    E_shift = 0.01  # Small energy shift above resonance
    Ek = square_lattice.omega_e + collective_lamb_shift + E_shift
    Ep = square_lattice.omega_e + collective_lamb_shift + E_shift
    
    results = {
        'nquad': {},
        'qmc': {}
    }
    
    print("=" * 60)
    print("Comparison Test: nquad vs QMC Integration")
    print("=" * 60)
    print(f"\nTest Parameters (Origin of 1st Brillouin Zone):")
    print(f"  k_para = {k_para}  (origin)")
    print(f"  p_para = {p_para}  (origin)")
    print(f"  Ek = Ep = {Ek:.6f}")
    print(f"  Total Energy E = {Ek + Ep:.6f}")
    print(f"  Lattice spacing a = {square_lattice.a:.6f}")
    print(f"  omega_e = {square_lattice.omega_e}")
    print(f"  Collective Lamb shift = {collective_lamb_shift:.6f}")
    print()
    
    # Test 1: QMC integration FIRST (fast)
    qmc_m_values = [10, 12, 13, 14]  # 2^m samples: 1024, 4096, 8192, 16384
    
    for m in qmc_m_values:
        n_samples = 2**m
        print("-" * 60)
        print(f"Running QMC integration (m={m}, n_samples={n_samples})...")
        try:
            start_time = time.time()
            result_qmc = scattering_integral_qmc(
                k_para, Ek, p_para, Ep, square_lattice, gaussian_in_state,
                m=m, seed=42
            )
            qmc_time = time.time() - start_time
            
            results['qmc'][m] = {
                'result': result_qmc,
                'time': qmc_time,
                'n_samples': n_samples,
                'success': True
            }
            print(f"  Result: {result_qmc}")
            print(f"  Real part: {result_qmc.real:.10e}")
            print(f"  Imag part: {result_qmc.imag:.10e}")
            print(f"  Magnitude: {abs(result_qmc):.10e}")
            print(f"  Time: {qmc_time:.3f} seconds")
                
        except Exception as e:
            results['qmc'][m] = {'success': False, 'error': str(e)}
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Test 2: nquad integration (OPTIONAL - can be slow)
    if run_nquad:
        print("-" * 60)
        print("Running nquad integration (parallelized)...")
        print("  This may take a few minutes...")
        try:
            start_time = time.time()
            result_nquad = scattering_integral(
                k_para, Ek, p_para, Ep, square_lattice, gaussian_in_state
            )
            nquad_time = time.time() - start_time
            
            results['nquad'] = {
                'result': result_nquad,
                'time': nquad_time,
                'success': True
            }
            print(f"  Result: {result_nquad}")
            print(f"  Real part: {result_nquad.real:.10e}")
            print(f"  Imag part: {result_nquad.imag:.10e}")
            print(f"  Magnitude: {abs(result_nquad):.10e}")
            print(f"  Time: {nquad_time:.3f} seconds")
            
            # Compare QMC results with nquad
            print("\n  Comparison with QMC:")
            for m in results['qmc']:
                if results['qmc'][m].get('success'):
                    qmc_result = results['qmc'][m]['result']
                    rel_error = abs(abs(qmc_result) - abs(result_nquad)) / (abs(result_nquad) + 1e-15)
                    speedup = nquad_time / results['qmc'][m]['time']
                    print(f"    m={m}: rel_error = {rel_error:.6e}, speedup = {speedup:.1f}x")
                    results['qmc'][m]['rel_error_mag'] = rel_error
                    
        except Exception as e:
            results['nquad'] = {'success': False, 'error': str(e)}
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    else:
        print("-" * 60)
        print("Skipping nquad integration (set run_nquad=True to enable)")
        results['nquad'] = {'success': False, 'error': 'Skipped'}
        print()
    
    return results


def generate_markdown_report(results):
    """Generate a markdown report from the test results."""
    
    report = []
    report.append("# Integration Method Comparison: nquad vs QMC")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append("This report compares two integration methods for computing the scattering integral:")
    report.append("- **nquad**: Scipy's adaptive quadrature integration (parallelized)")
    report.append("- **QMC**: Quasi-Monte Carlo using Sobol sequences")
    report.append("")
    
    report.append("## Test Configuration")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    report.append(f"| k_para | [0.0, 0.0] |")
    report.append(f"| p_para | [0.0, 0.0] |")
    report.append(f"| Lattice spacing (a) | {square_lattice.a:.6f} |")
    report.append(f"| omega_e | {square_lattice.omega_e} |")
    report.append(f"| Collective Lamb shift | {collective_lamb_shift:.6f} |")
    report.append("")
    
    report.append("## Results")
    report.append("")
    
    # nquad results
    report.append("### nquad (Quadrature) Results")
    report.append("")
    if results['nquad'].get('success'):
        r = results['nquad']['result']
        report.append(f"- **Result**: `{r}`")
        report.append(f"- **Real part**: `{r.real:.10e}`")
        report.append(f"- **Imaginary part**: `{r.imag:.10e}`")
        report.append(f"- **Magnitude**: `{abs(r):.10e}`")
        report.append(f"- **Execution time**: `{results['nquad']['time']:.3f}` seconds")
    else:
        report.append(f"- **Error**: {results['nquad'].get('error', 'Unknown error')}")
    report.append("")
    
    # QMC results table
    report.append("### QMC (Quasi-Monte Carlo) Results")
    report.append("")
    report.append("| m | Samples | Real Part | Imag Part | Magnitude | Time (s) | Rel. Error | Speedup |")
    report.append("|---|---------|-----------|-----------|-----------|----------|------------|---------|")
    
    nquad_time = results['nquad'].get('time', 1)
    
    for m in sorted(results['qmc'].keys()):
        qmc_data = results['qmc'][m]
        if qmc_data.get('success'):
            r = qmc_data['result']
            t = qmc_data['time']
            n = qmc_data['n_samples']
            rel_err = qmc_data.get('rel_error_mag', 'N/A')
            speedup = nquad_time / t if results['nquad'].get('success') else 'N/A'
            
            rel_err_str = f"{rel_err:.2e}" if isinstance(rel_err, float) else rel_err
            speedup_str = f"{speedup:.1f}x" if isinstance(speedup, float) else speedup
            
            report.append(f"| {m} | {n} | {r.real:.6e} | {r.imag:.6e} | {abs(r):.6e} | {t:.3f} | {rel_err_str} | {speedup_str} |")
        else:
            report.append(f"| {m} | 2^{m} | ERROR | ERROR | ERROR | - | - | - |")
    
    report.append("")
    
    report.append("## Recommendations")
    report.append("")
    report.append("1. **For high accuracy**: Use `scattering_integral()` (nquad) or `scattering_integral_qmc()` with m≥14")
    report.append("2. **For speed with reasonable accuracy**: Use `scattering_integral_qmc()` with m=12-13")
    report.append("3. **For quick estimates**: Use `scattering_integral_qmc()` with m=10")
    report.append("")
    
    report.append("## Code Usage")
    report.append("")
    report.append("```python")
    report.append("# Quadrature (high accuracy, slower)")
    report.append("result = scattering_integral(k_para, Ek, p_para, Ep, lattice, in_state)")
    report.append("")
    report.append("# QMC (adjustable accuracy/speed tradeoff)")
    report.append("result = scattering_integral_qmc(k_para, Ek, p_para, Ep, lattice, in_state, m=13, seed=42)")
    report.append("```")
    report.append("")
    
    return "\n".join(report)


#%%
if __name__ == "__main__":
    import sys
    
    print("Integration Comparison Test\n")
    print("Usage:")
    print("  python test_nquad_vs_qmc.py          # QMC only, multiple points")
    print("  python test_nquad_vs_qmc.py nquad    # QMC + nquad comparison")
    print("  python test_nquad_vs_qmc.py single   # Original single-point test")
    print()
    
    run_nquad = 'nquad' in sys.argv
    single_test = 'single' in sys.argv
    
    if single_test:
        # Original single-point comparison
        results = run_comparison_test(run_nquad=run_nquad)
        
        # Generate and save the report
        print("=" * 60)
        print("Generating markdown report...")
        report = generate_markdown_report(results)
        
        report_path = "/Users/ywan8652/Desktop/2D_Array/src/integration_comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
    else:
        # Multi-point comparison
        results = run_multi_point_comparison(run_nquad=run_nquad, qmc_m=13)
    
    print("=" * 60)
    print("\nTest completed!")

# %%
