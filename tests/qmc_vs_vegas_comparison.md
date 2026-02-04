# QMC vs Vegas Monte Carlo Integration Comparison

## Summary

This report compares two Monte Carlo integration methods:
- **QMC (Sobol)**: Quasi-Monte Carlo using low-discrepancy Sobol sequences
- **Vegas**: Adaptive importance sampling Monte Carlo

## QMC Results

| m | Samples | Result (Real) | Result (Imag) | Magnitude | Time (s) |
|---|---------|---------------|---------------|-----------|----------|
| 10 | 1024 | -6.798605e+02 | 1.514644e+02 | 6.965283e+02 | 0.145 |
| 12 | 4096 | -6.964226e+02 | 1.644491e+02 | 7.155752e+02 | 0.496 |
| 13 | 8192 | -6.946789e+02 | 1.643828e+02 | 7.138631e+02 | 0.977 |
| 14 | 16384 | -6.946786e+02 | 1.644006e+02 | 7.138669e+02 | 1.952 |

## Vegas Results

| Config | Total Evals | Result (Real) | Result (Imag) | Magnitude | Time (s) |
|--------|-------------|---------------|---------------|-----------|----------|
| 5x500 | 2500 | -6.924440e+02 | 1.661842e+02 | 7.121067e+02 | 0.993 |
| 10x500 | 5000 | -6.963182e+02 | 1.661909e+02 | 7.158760e+02 | 1.071 |
| 10x1000 | 10000 | -6.953076e+02 | 1.617052e+02 | 7.138636e+02 | 2.276 |
| 10x2000 | 20000 | -6.946141e+02 | 1.629042e+02 | 7.134609e+02 | 4.287 |

## Comparison Summary

Using QMC m=14 as reference (magnitude: 7.138669e+02)

| Method | Config | Rel. Error (mag) | Time (s) | Efficiency |
|--------|--------|------------------|----------|------------|
| QMC | m=10 | 2.43e-02 | 0.145 | 284.7 |
| QMC | m=12 | 2.39e-03 | 0.496 | 842.9 |
| QMC | m=13 | 5.31e-06 | 0.977 | 192673.0 |
| QMC | m=14 | 0.00e+00 | 1.952 | inf |
| Vegas | 5x500 | 2.47e-03 | 0.993 | 408.2 |
| Vegas | 10x500 | 2.81e-03 | 1.071 | 331.8 |
| Vegas | 10x1000 | 4.63e-06 | 2.276 | 95000.0 |
| Vegas | 10x2000 | 5.69e-04 | 4.287 | 410.2 |

## Conclusions

- **QMC (Sobol)**: Deterministic, reproducible, optimal for smooth integrands
- **Vegas**: Adaptive, better for integrands with localized features
- For this integral, compare accuracy vs time tradeoffs above
