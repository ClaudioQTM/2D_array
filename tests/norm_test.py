from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "../src"))

from S_matrix import  square_lattice, collective_lamb_shift
from input_states import gaussian_in_state
import numpy as np
import vegas
import time

q0 = np.array([0, 0, 50 * (square_lattice.omega_e + collective_lamb_shift)])
l0 = np.array([0, 0, 50 * (square_lattice.omega_e + collective_lamb_shift)])
sigma = np.pi / (3 * square_lattice.a)
cut_off = 4 * sigma
q_up = q0 + cut_off
q_low = q0 - cut_off
l_up = q_up
l_low = q_low

# Instantiate the Gaussian input state once and reuse it in the integrands.
_gaussian_in_state = gaussian_in_state(q0=q0, l0=l0, sigma=sigma)


def test_integrand(x):
    """Integrand for Vegas: x = [qx, qy, qz, lx, ly, lz]."""
    qx, qy, qz, lx, ly, lz = x[0], x[1], x[2], x[3], x[4], x[5]
    q = np.array([qx, qy, qz], dtype=float)
    l = np.array([lx, ly, lz], dtype=float)

    # Energies from full 3D momenta (c = 1 in this model).
    Eq = np.linalg.norm(q)
    El = np.linalg.norm(l)

    # Parallel components [kx, ky] used by gaussian_in_state.
    q_para = q[:2]
    l_para = l[:2]

    val = _gaussian_in_state(q_para, Eq, l_para, El)
    return val**2


@vegas.rbatchintegrand
def test_integrand2(x):
    """Vectorised version: x has shape (6,) or (n, 6) with [qx, qy, qz, lx, ly, lz]."""
    x = np.asarray(x)
    is_scalar = x.ndim == 1
    if is_scalar:
        x = x.reshape(1, -1)
    else:
        if x.shape[0] == 6 and x.shape[1] != 6:
            x = x.T
        elif x.shape[1] != 6:
            raise ValueError(f"Expected shape (n, 6) or (6, n); got {x.shape}.")

    qx, qy, qz, lx, ly, lz = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    Eq = np.sqrt(qz**2 + qx**2 + qy**2)
    El = np.sqrt(lz**2 + lx**2 + ly**2)
    q_para = np.stack([qx, qy], axis=1)
    l_para = np.stack([lx, ly], axis=1)

    val = _gaussian_in_state(q_para, Eq, l_para, El)
    val = val**2
    if is_scalar:
        return val[0]
    return val



# 6D integration bounds: [qx, qy, qz, lx, ly, lz]
bounds = [
    [q_low[0], q_up[0]], [q_low[1], q_up[1]], [q_low[2], q_up[2]],
    [l_low[0], l_up[0]], [l_low[1], l_up[1]], [l_low[2], l_up[2]],
]
t0 = time.perf_counter()
norm_integ = vegas.Integrator(bounds)
vegas.Integrator(bounds)(test_integrand2, nitn=5, neval=5e4)
result = vegas.Integrator(bounds)(test_integrand2, nitn=10, neval=5e4)
elapsed = time.perf_counter() - t0
print(f"Time taken: {elapsed:.2f} seconds")
#result2 = vegas.Integrator(bounds)(test_integrand2, nitn=10, neval=25000)
print("test_integrand: ", result.summary())
#print("test_integrand2:", result2.summary())


'''
def test_integrands_agree(n_samples=500, rtol=1e-10, atol=1e-12, seed=42):
    """
    Verify test_integrand and test_integrand2 agree on random points within bounds.
    Restricts to qz > 0, lz > 0 (forward light cone) since coord_convert uses
    sqrt(Eq² - q_para²) and always returns positive qz, while the Cartesian
    branch of gaussian_in_state uses the input qz directly.
    """
    rng = np.random.default_rng(seed)
    eps = 1e-6
    x_samples = np.column_stack([
        rng.uniform(q_low[0], q_up[0], n_samples),
        rng.uniform(q_low[1], q_up[1], n_samples),
        rng.uniform(max(eps, q_low[2]), q_up[2], n_samples),  # qz > 0
        rng.uniform(l_low[0], l_up[0], n_samples),
        rng.uniform(l_low[1], l_up[1], n_samples),
        rng.uniform(max(eps, l_low[2]), l_up[2], n_samples),  # lz > 0
    ])
    max_diff = 0.0
    n_fail = 0
    for i in range(n_samples):
        x = x_samples[i]
        v1 = test_integrand(x)
        v2 = test_integrand2(x)
        diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
        if not np.isclose(v1, v2, rtol=rtol, atol=atol):
            n_fail += 1
            if n_fail <= 3:  # Show first few failures
                print(f"  Mismatch at x={x}: test_integrand={v1:.6e}, test_integrand2={v2:.6e}")
    print(f"test_integrands_agree: {n_fail}/{n_samples} mismatches, max_diff={max_diff:.2e}")
    assert n_fail == 0, f"test_integrand and test_integrand2 disagree at {n_fail} points"
    print("  PASS: test_integrand and test_integrand2 agree everywhere")


test_integrands_agree()


sampler = qmc.Sobol(d=6, scramble=True, seed=42)
samples = sampler.random_base2(13)
u1, u2, u3, u4, u5, u6 = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3], samples[:, 4], samples[:, 5]
qx = q_low[0] + u1 * (q_up[0] - q_low[0])
qy = q_low[1] + u2 * (q_up[1] - q_low[1])
qz = q_low[2] + u3 * (q_up[2] - q_low[2])
lx = l_low[0] + u4 * (l_up[0] - l_low[0])
ly = l_low[1] + u5 * (l_up[1] - l_low[1])
lz = l_low[2] + u6 * (l_up[2] - l_low[2])

q_arr = np.array([qx, qy, qz])
l_arr = np.array([lx, ly, lz])

total = 0.0
jacobian = (q_up[0] - q_low[0]) * (q_up[1] - q_low[1]) * (q_up[2] - q_low[2]) * (l_up[0] - l_low[0]) * (l_up[1] - l_low[1]) * (l_up[2] - l_low[2])
n_samples = q_arr.shape[1]
# Iterate along second axis: each (q, l) is one sample [qx, qy, qz] and [lx, ly, lz]
for q, l in zip(q_arr.T, l_arr.T):
    x = np.concatenate([q, l])  # test_integrand expects [qx, qy, qz, lx, ly, lz]
    val = test_integrand(x)     # already returns |ψ|²
    total += val * jacobian
# QMC estimate: (1/n) * sum(f) * volume
total /= 2**13
print(total)




'''

