from __future__ import annotations

import sys
import types

import numpy as np

import eigenstate_solving.eigen_eq_vegas as eigen_eq_vegas_mod
import eigenstate_solving.eigen_eq_integrand as eigen_eq_integrand_mod
from eigenstate_solving.eigen_eq_integrand import (
    _make_eigen_eq_integrand,
    _make_eigen_eq_integrand_numba,
)
from eigenstate_solving.eigen_eq_vegas import eigen_eq_itr, eigen_eq_itr_batch
from smatrix import square_lattice


def _zero_sigma_interpolator():
    def _sigma_func_period(kx, ky):
        _ = kx, ky
        return 0.0 + 0.0j

    return _sigma_func_period


def test_make_eigen_eq_integrand_numba_matches_legacy():
    sigma_func = _zero_sigma_interpolator()
    E = 2 * float(square_lattice.omega_e)
    Q = np.array([0.05, -0.03], dtype=np.float64)
    G = np.array([0.0, 0.0], dtype=np.float64)
    H = np.array([0.0, 0.0], dtype=np.float64)
    tEQ = np.exp(1j * np.pi / 4)

    integrand_legacy = _make_eigen_eq_integrand(E, Q, G, H, square_lattice, sigma_func, tEQ)
    integrand_numba = _make_eigen_eq_integrand_numba(
        E, Q, G, H, square_lattice, sigma_func, tEQ
    )

    x_samples = np.array(
        [
            [-0.8, -0.8, 0.2],
            [-0.4, 0.3, 0.7],
            [0.0, 0.0, 0.5],
            [0.6, -0.2, 0.1],
            [0.9, 0.9, 0.95],
        ],
        dtype=np.float64,
    )

    legacy_vals = np.array([integrand_legacy(x) for x in x_samples], dtype=np.complex128)
    numba_vals_scalar = np.array([integrand_numba(x) for x in x_samples], dtype=np.complex128)
    numba_vals_batch = np.asarray(integrand_numba(x_samples), dtype=np.complex128)

    assert np.allclose(numba_vals_scalar, legacy_vals, rtol=1e-10, atol=1e-10)
    assert np.allclose(numba_vals_batch, legacy_vals, rtol=1e-10, atol=1e-10)


def test_eigen_eq_itr_batch_matches_legacy_practically(monkeypatch):
    sigma_func = _zero_sigma_interpolator()
    E = 2 * float(square_lattice.omega_e)
    Q = np.array([0.03, -0.02], dtype=np.float64)
    tEQ = np.exp(1j * np.pi / 4)

    monkeypatch.setattr(
        eigen_eq_vegas_mod,
        "GH_filter_vectorized",
        lambda Q, E, lattice: [(np.array([0.0, 0.0]), np.array([0.0, 0.0]))],
    )
    monkeypatch.setattr(
        eigen_eq_vegas_mod,
        "tau_matrix_element",
        lambda E, Q, lattice, sigma_func_period: 1.0 + 0.0j,
    )
    monkeypatch.setattr(
        eigen_eq_integrand_mod, "_build_eigen_eq_geometry_kernel", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        eigen_eq_integrand_mod, "t", lambda k, e, lattice, sigma_func_period=None: 1.0 + 0.25j
    )
    monkeypatch.setattr(
        eigen_eq_integrand_mod,
        "legs",
        lambda rG, E1, sH, E2, lattice, sigma_func_period, direction="in": (
            1.2 + 0.1j if direction == "in" else 0.7 - 0.2j
        ),
    )

    class _FakeIntegrator:
        def __init__(self, bounds):
            self._bounds = bounds

        def __call__(self, func, nitn, neval):
            _ = nitn, neval
            samples = np.array(
                [
                    [-1.0, -1.0, 0.0],
                    [-0.5, 0.5, 0.25],
                    [0.0, 0.0, 0.5],
                    [0.4, -0.2, 0.75],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float64,
            )
            if getattr(func, "_is_lbatch", False):
                values = np.asarray(func(samples), dtype=np.float64)
            else:
                values = np.array([func(sample) for sample in samples], dtype=np.float64)
            volume = np.prod([b[1] - b[0] for b in self._bounds])
            return types.SimpleNamespace(mean=float(volume * np.mean(values)))

    def _fake_lbatchintegrand(func):
        func._is_lbatch = True
        return func

    fake_vegas = types.SimpleNamespace(
        Integrator=_FakeIntegrator, lbatchintegrand=_fake_lbatchintegrand
    )
    monkeypatch.setitem(sys.modules, "vegas", fake_vegas)

    np.random.seed(12345)
    legacy_val = eigen_eq_itr(
        Q, E, square_lattice, sigma_func, tEQ, neval=600, nitn=1
    )
    np.random.seed(12345)
    batch_val = eigen_eq_itr_batch(
        Q, E, square_lattice, sigma_func, tEQ, neval=600, nitn=1
    )

    assert np.allclose(batch_val, legacy_val, rtol=1e-10, atol=1e-10)
