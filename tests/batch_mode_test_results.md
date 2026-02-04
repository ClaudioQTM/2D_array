## Batch vs scalar mode test results

- Environment: `conda` env `atomicarray`
- Command: `conda run -n atomicarray env PYTHONPATH=src python -m pytest tests/test_batch_mode.py -q`
- Result: `6 passed in 3.90s`
- Notes: tests use 256 random 2D vectors in the first Brillouin zone (full square |kx|, |ky| ≤ q/2, including the boundary)
