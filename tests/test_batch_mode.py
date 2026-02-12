"""
Batch vs scalar mode tests for S_matrix and model.

Primary check: the fast batch result agrees with the slow scalar result at every
point. For each test we:
  1. Compute the batch result (one call with shape (n, 2) — fast).
  2. Compute the scalar result at each point (n separate calls — slow).
  3. Assert that batch[i] matches scalar(point_i) for all i.

Secondary check (where applicable): (2, n) input gives the same batch as (n, 2).

We use a moderate n_points so the scalar-loop comparison runs in reasonable time.
"""
import numpy as np
from model import EMField
from input_states import gaussian_in_state
from smatrix import (
    coord_convert,
    create_self_energy_interpolator_numba,
    square_lattice,
    sw_propagator,
    t,
    legs,
)
from scattering import _make_integrand_and_bounds

# Number of points: enough to stress batch vs scalar agreement without slow tests
N_POINTS = 10000


def _points(n_points=None, seed=42):
    """Random 2D vectors in the first Brillouin zone, including the boundary."""
    if n_points is None:
        n_points = N_POINTS
    rng = np.random.default_rng(seed)
    bound = float(square_lattice.q / 2)  # full BZ: |kx|, |ky| <= q/2
    return rng.uniform(-bound, bound, size=(n_points, 2))


def test_disp_rel_scalar_vs_batch():
    """DispRel: slow scalar at each point agrees with fast batch result."""
    field = EMField()
    points = _points()
    k_z = 0.3
    # Fast: single batch call
    k_z_batch = np.full(points.shape[0], k_z)
    batch = field.DispRel(points, k_z_batch)
    assert batch.shape == (points.shape[0],)
    # Slow: scalar call at each point; must match batch
    scalar_results = np.array([float(field.DispRel(p, k_z)) for p in points])
    assert np.allclose(scalar_results, batch), "scalar loop vs batch mismatch"
    # (2, n) input gives same batch
    batch_t = field.DispRel(points.T, k_z_batch)
    assert np.allclose(batch_t, batch)


def test_green_tensor_scalar_vs_batch():
    """GreenTensor: slow scalar at each point agrees with fast batch result."""
    field = EMField()
    points = _points()
    z = 0.5
    E = max(2.0, float(square_lattice.q / 2) * 1.01)
    z_batch = np.full(points.shape[0], z)
    # Fast: single batch call
    batch = field.GreenTensor(points, z_batch, E)
    assert batch.shape == (points.shape[0], 3, 3)
    # Slow: scalar call at each point; must match batch
    scalar_results = np.array([field.GreenTensor(p, z, E) for p in points])
    assert np.allclose(scalar_results, batch), "scalar loop vs batch mismatch"
    batch_t = field.GreenTensor(points.T, z_batch, E)
    assert np.allclose(batch_t, batch)


def test_coord_convert_scalar_vs_batch():
    """coord_convert: slow scalar at each point agrees with fast batch result."""
    points = _points()
    E = max(2.0, float(square_lattice.q / 2) * 1.01)
    # Fast: single batch call
    batch = coord_convert(points, E)
    assert batch.shape == (points.shape[0], 3)
    # Slow: scalar call at each point; must match batch
    scalar_results = np.array([coord_convert(p, E) for p in points])
    assert np.allclose(scalar_results, batch), "scalar loop vs batch mismatch"
    batch_t = coord_convert(points.T, E)
    assert np.allclose(batch_t, batch)


def test_sw_propagator_scalar_vs_batch():
    """sw_propagator: slow scalar at each point agrees with fast batch result."""
    points = _points()
    E = 2 * square_lattice.omega_e
    sigma_func_period = create_self_energy_interpolator_numba(
        np.array([0.0, float(square_lattice.q / 2)]),
        np.array([0.0, float(square_lattice.q / 2)]),
        np.zeros((2, 2), dtype=np.complex128),
        lattice=square_lattice,
    )
    # Fast: single batch call
    batch = sw_propagator(points, E, square_lattice, sigma_func_period)
    assert batch.shape == (points.shape[0],)
    # Slow: scalar call at each point; must match batch
    scalar_results = np.array([sw_propagator(p, E, square_lattice, sigma_func_period) for p in points])
    assert np.allclose(scalar_results, batch), "scalar loop vs batch mismatch"
    batch_t = sw_propagator(points.T, np.full(points.shape[0], E), square_lattice, sigma_func_period)
    assert np.allclose(batch_t, batch)


def test_t_scalar_vs_batch():
    """t: slow scalar at each point agrees with fast batch result."""
    points = _points()
    E = 2 * square_lattice.omega_e
    sigma_func_period = create_self_energy_interpolator_numba(
        np.array([0.0, float(square_lattice.q / 2)]),
        np.array([0.0, float(square_lattice.q / 2)]),
        np.zeros((2, 2), dtype=np.complex128),
        lattice=square_lattice,
    )
    # Fast: single batch call
    batch = t(points, E, square_lattice, sigma_func_period)
    assert batch.shape == (points.shape[0],)
    # Slow: scalar call at each point; must match batch
    scalar_results = np.array([t(p, E, square_lattice, sigma_func_period) for p in points])
    assert np.allclose(scalar_results, batch), "scalar loop vs batch mismatch"
    batch_t = t(points.T, np.full(points.shape[0], E), square_lattice, sigma_func_period)
    assert np.allclose(batch_t, batch)


def test_legs_scalar_vs_batch():
    """legs: slow scalar at each (q,l) pair agrees with fast batch result (in & out)."""
    q_points = _points()
    l_points = _points(seed=43)
    Eq = 2 * square_lattice.omega_e
    El = 2 * square_lattice.omega_e
    sigma_func_period = create_self_energy_interpolator_numba(
        np.array([0.0, float(square_lattice.q / 2)]),
        np.array([0.0, float(square_lattice.q / 2)]),
        np.zeros((2, 2), dtype=np.complex128),
        lattice=square_lattice,
    )
    # Fast: single batch call
    batch_in = legs(q_points, Eq, l_points, El, square_lattice, sigma_func_period, direction="in")
    batch_out = legs(q_points, Eq, l_points, El, square_lattice, sigma_func_period, direction="out")
    assert batch_in.shape == (q_points.shape[0],)
    assert batch_out.shape == (q_points.shape[0],)
    # Slow: scalar call at each (q, l) pair; must match batch
    scalar_in = np.array([
        legs(q_points[i], Eq, l_points[i], El, square_lattice, sigma_func_period, direction="in")
        for i in range(q_points.shape[0])
    ])
    scalar_out = np.array([
        legs(q_points[i], Eq, l_points[i], El, square_lattice, sigma_func_period, direction="out")
        for i in range(q_points.shape[0])
    ])
    assert np.allclose(scalar_in, batch_in), "scalar loop vs batch (in) mismatch"
    assert np.allclose(scalar_out, batch_out), "scalar loop vs batch (out) mismatch"
    batch_in_t = legs(
        q_points.T,
        np.full(q_points.shape[0], Eq),
        l_points.T,
        np.full(q_points.shape[0], El),
        square_lattice,
        sigma_func_period,
    )
    assert np.allclose(batch_in_t, batch_in)


def test_make_integrand_scalar_vs_batch():
    """_make_integrand_and_bounds integrand: scalar calls match batch result."""
    E = 2 * square_lattice.omega_e
    sigma_func_period = create_self_energy_interpolator_numba(
        np.array([0.0, float(square_lattice.q / 2)]),
        np.array([0.0, float(square_lattice.q / 2)]),
        np.zeros((2, 2), dtype=np.complex128),
        lattice=square_lattice,
    )
    integrand, D_bounds = _make_integrand_and_bounds(E, square_lattice, _test_in_state(E), sigma_func_period)

    points = _bz_points(n_random=N_POINTS, seed=0)
    Dpx = points[:, 0]
    Dpy = points[:, 1]
    COM_K = np.array([0.05, -0.03])
    G = np.array([0.0, 0.0])
    H = np.array([0.0, 0.0])

    D_min, D_max = D_bounds(Dpx, Dpy, COM_K, G, H)
    D = 0.5 * (D_min + D_max)

    batch = integrand(D, Dpx, Dpy, COM_K, G, H)
    scalar = np.array([
        integrand(D[i], Dpx[i], Dpy[i], COM_K, G, H) for i in range(points.shape[0])
    ])

    assert np.allclose(scalar, batch), "integrand scalar loop vs batch mismatch"


def test_make_integrand_bounds_scalar_vs_batch():
    """_make_integrand_and_bounds D_bounds: scalar calls match batch result."""
    E = 2 * square_lattice.omega_e
    sigma_func_period = create_self_energy_interpolator_numba(
        np.array([0.0, float(square_lattice.q / 2)]),
        np.array([0.0, float(square_lattice.q / 2)]),
        np.zeros((2, 2), dtype=np.complex128),
        lattice=square_lattice,
    )
    _, D_bounds = _make_integrand_and_bounds(E, square_lattice, _test_in_state(E), sigma_func_period)

    points = _bz_points(n_random=N_POINTS, seed=1)
    Dpx = points[:, 0]
    Dpy = points[:, 1]
    COM_K = np.array([-0.04, 0.02])
    G = np.array([0.0, 0.0])
    H = np.array([0.0, 0.0])

    D_min_batch, D_max_batch = D_bounds(Dpx, Dpy, COM_K, G, H)

    D_min_scalar = np.array([
        D_bounds(Dpx[i], Dpy[i], COM_K, G, H)[0] for i in range(points.shape[0])
    ])
    D_max_scalar = np.array([
        D_bounds(Dpx[i], Dpy[i], COM_K, G, H)[1] for i in range(points.shape[0])
    ])

    assert np.allclose(D_min_scalar, D_min_batch), "D_min scalar vs batch mismatch"
    assert np.allclose(D_max_scalar, D_max_batch), "D_max scalar vs batch mismatch"


def _test_in_state(E):
    """Helper: simple Gaussian input state for integrand tests."""
    q0 = np.array([0.0, 0.0, E / 2])
    l0 = np.array([0.0, 0.0, E / 2])
    sigma = np.pi / (3 * square_lattice.a)
    return gaussian_in_state(q0=q0, l0=l0, sigma=sigma)


def _bz_points(n_random=256, seed=0):
    """Sample points over the full BZ and include high-symmetry points."""
    rng = np.random.default_rng(seed)
    bound = np.pi / square_lattice.a

    random_points = rng.uniform(-bound, bound, size=(n_random, 2))

    hs = bound
    high_symmetry = np.array(
        [
            [0.0, 0.0],      # Gamma
            [hs, 0.0],       # X
            [0.0, hs],       # Y
            [-hs, 0.0],
            [0.0, -hs],
            [hs, hs],        # M
            [hs, -hs],
            [-hs, hs],
            [-hs, -hs],
        ],
        dtype=np.float64,
    )

    return np.vstack([high_symmetry, random_points])
