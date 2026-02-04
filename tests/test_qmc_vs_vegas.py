#%%
"""
Test script to compare QMC (Sobol) vs Vegas Monte Carlo integration methods.
"""

import numpy as np
import time

# Import from S_matrix
from S_matrix import (
    scattering_integral_qmc,
    scattering_integral_vegas,
    square_lattice,
    collective_lamb_shift,
    gaussian_in_state
)


def run_qmc_vs_vegas_comparison():
    """Compare QMC and Vegas integration methods."""
    
    # Test parameters - Origin of the first Brillouin zone
    k_para = np.array([0.0, 0.0])
    p_para = np.array([0.0, 0.0])
    E_shift = 0.01
    Ek = square_lattice.omega_e + collective_lamb_shift + E_shift
    Ep = square_lattice.omega_e + collective_lamb_shift + E_shift
    
    results = {
        'qmc': {},
        'vegas': {}
    }
    
    print("=" * 70)
    print("Comparison Test: QMC (Sobol) vs Vegas Monte Carlo")
    print("=" * 70)
    print(f"\nTest Parameters:")
    print(f"  k_para = {k_para}")
    print(f"  p_para = {p_para}")
    print(f"  Ek = Ep = {Ek:.6f}")
    print(f"  Total Energy E = {Ek + Ep:.6f}")
    print()
    
    # QMC tests with different sample sizes
    qmc_configs = [
        (10, "1024 samples"),
        (12, "4096 samples"),
        (13, "8192 samples"),
        (14, "16384 samples"),
    ]
    
    print("-" * 70)
    print("QMC (Sobol Sequence) Integration")
    print("-" * 70)
    
    for m, desc in qmc_configs:
        print(f"\n  m={m} ({desc})...")
        try:
            start = time.time()
            result = scattering_integral_qmc(
                k_para, Ek, p_para, Ep, square_lattice, gaussian_in_state,
                m=m, seed=42
            )
            elapsed = time.time() - start
            
            results['qmc'][m] = {
                'result': result,
                'time': elapsed,
                'samples': 2**m,
                'success': True
            }
            print(f"    Result: {result}")
            print(f"    Magnitude: {abs(result):.6e}")
            print(f"    Time: {elapsed:.3f}s")
        except Exception as e:
            results['qmc'][m] = {'success': False, 'error': str(e)}
            print(f"    ERROR: {e}")
    
    # Vegas tests with different configurations
    vegas_configs = [
        (5, 500, "5 itn × 500 eval = 2500 total"),
        (10, 500, "10 itn × 500 eval = 5000 total"),
        (10, 1000, "10 itn × 1000 eval = 10000 total"),
        (10, 2000, "10 itn × 2000 eval = 20000 total"),
    ]
    
    print("\n" + "-" * 70)
    print("Vegas (Adaptive Monte Carlo) Integration")
    print("-" * 70)
    
    for nitn, neval, desc in vegas_configs:
        key = f"{nitn}x{neval}"
        print(f"\n  {desc}...")
        try:
            start = time.time()
            result = scattering_integral_vegas(
                k_para, Ek, p_para, Ep, square_lattice, gaussian_in_state,
                nitn=nitn, neval=neval
            )
            elapsed = time.time() - start
            
            results['vegas'][key] = {
                'result': result,
                'time': elapsed,
                'nitn': nitn,
                'neval': neval,
                'total_evals': nitn * neval,
                'success': True
            }
            print(f"    Result: {result}")
            print(f"    Magnitude: {abs(result):.6e}")
            print(f"    Time: {elapsed:.3f}s")
        except Exception as e:
            results['vegas'][key] = {'success': False, 'error': str(e)}
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def generate_report(results):
    """Generate markdown report."""
    
    report = []
    report.append("# QMC vs Vegas Monte Carlo Integration Comparison")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append("This report compares two Monte Carlo integration methods:")
    report.append("- **QMC (Sobol)**: Quasi-Monte Carlo using low-discrepancy Sobol sequences")
    report.append("- **Vegas**: Adaptive importance sampling Monte Carlo")
    report.append("")
    
    report.append("## QMC Results")
    report.append("")
    report.append("| m | Samples | Result (Real) | Result (Imag) | Magnitude | Time (s) |")
    report.append("|---|---------|---------------|---------------|-----------|----------|")
    
    for m in sorted(results['qmc'].keys()):
        data = results['qmc'][m]
        if data.get('success'):
            r = data['result']
            report.append(f"| {m} | {data['samples']} | {r.real:.6e} | {r.imag:.6e} | {abs(r):.6e} | {data['time']:.3f} |")
        else:
            report.append(f"| {m} | - | ERROR | - | - | - |")
    
    report.append("")
    report.append("## Vegas Results")
    report.append("")
    report.append("| Config | Total Evals | Result (Real) | Result (Imag) | Magnitude | Time (s) |")
    report.append("|--------|-------------|---------------|---------------|-----------|----------|")
    
    for key in results['vegas']:
        data = results['vegas'][key]
        if data.get('success'):
            r = data['result']
            report.append(f"| {key} | {data['total_evals']} | {r.real:.6e} | {r.imag:.6e} | {abs(r):.6e} | {data['time']:.3f} |")
        else:
            report.append(f"| {key} | - | ERROR | - | - | - |")
    
    report.append("")
    report.append("## Comparison Summary")
    report.append("")
    
    # Find reference (highest QMC)
    qmc_ref = None
    for m in sorted(results['qmc'].keys(), reverse=True):
        if results['qmc'][m].get('success'):
            qmc_ref = results['qmc'][m]['result']
            qmc_ref_m = m
            break
    
    if qmc_ref:
        report.append(f"Using QMC m={qmc_ref_m} as reference (magnitude: {abs(qmc_ref):.6e})")
        report.append("")
        report.append("| Method | Config | Rel. Error (mag) | Time (s) | Efficiency |")
        report.append("|--------|--------|------------------|----------|------------|")
        
        ref_mag = abs(qmc_ref)
        
        for m in sorted(results['qmc'].keys()):
            data = results['qmc'][m]
            if data.get('success'):
                rel_err = abs(abs(data['result']) - ref_mag) / ref_mag
                eff = 1.0 / (rel_err * data['time'] + 1e-10) if rel_err > 0 else float('inf')
                report.append(f"| QMC | m={m} | {rel_err:.2e} | {data['time']:.3f} | {eff:.1f} |")
        
        for key in results['vegas']:
            data = results['vegas'][key]
            if data.get('success'):
                rel_err = abs(abs(data['result']) - ref_mag) / ref_mag
                eff = 1.0 / (rel_err * data['time'] + 1e-10) if rel_err > 0 else float('inf')
                report.append(f"| Vegas | {key} | {rel_err:.2e} | {data['time']:.3f} | {eff:.1f} |")
    
    report.append("")
    report.append("## Conclusions")
    report.append("")
    report.append("- **QMC (Sobol)**: Deterministic, reproducible, optimal for smooth integrands")
    report.append("- **Vegas**: Adaptive, better for integrands with localized features")
    report.append("- For this integral, compare accuracy vs time tradeoffs above")
    report.append("")
    
    return "\n".join(report)


#%%
if __name__ == "__main__":
    print("Starting QMC vs Vegas comparison test...\n")
    
    results = run_qmc_vs_vegas_comparison()
    
    print("\n" + "=" * 70)
    print("Generating report...")
    
    report = generate_report(results)
    report_path = "/Users/ywan8652/Desktop/2D_Array/src/qmc_vs_vegas_comparison.md"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    print("=" * 70)
    print("\nTest completed!")

# %%
