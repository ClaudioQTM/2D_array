from __future__ import annotations

from collections import Counter

import numpy as np

from model.model import EMField, SquareLattice
from scattering.filters import GH_filter_original, GH_filter_vectorized


def _to_multiset_from_original(gh_pairs):
    if len(gh_pairs) == 0:
        return Counter()
    rows = np.array(
        [(g[0], g[1], h[0], h[1]) for g, h in gh_pairs],
        dtype=float,
    )
    rows = np.round(rows, decimals=12)
    return Counter(map(tuple, rows.tolist()))


def test_gh_filter_original_agrees_with_vec():
    lattice = SquareLattice(
        a_lmd_ratio=0.4,
        omega_e=100.0,
        dipole_unit_vector=np.array([1.0, 1.0j, 0.0]) / np.sqrt(2),
        gamma=1.0,
        field=EMField(),
        grid_cutoff=20,
    )
    q = float(lattice.q)
    rng = np.random.default_rng(20260302)

    fixed_cases = [
        (np.array([0.0, 0.0]), 0.20 * q),
        (np.array([0.0, 0.0]), 0.35 * q),
        (np.array([0.0, 0.0]), 0.55 * q),
        (np.array([0.0, 0.0]), 0.75 * q),
        (np.array([0.0, 0.0]), 0.95 * q),
        (np.array([0.15 * q, -0.20 * q]), 0.30 * q),
        (np.array([0.15 * q, -0.20 * q]), 0.55 * q),
        (np.array([-0.30 * q, 0.25 * q]), 0.45 * q),
        (np.array([-0.30 * q, 0.25 * q]), 0.80 * q),
        (np.array([0.49 * q, 0.0]), 0.30 * q),
        (np.array([0.35 * q, -0.35 * q]), 0.95 * q),
        (np.array([0.49 * q, 0.0]), 0.60 * q),
        (np.array([0.0, -0.49 * q]), 0.60 * q),
        (np.array([-0.49 * q, 0.49 * q]), 0.90 * q),
        (np.array([0.25 * q, 0.25 * q]), 0.40 * q),
        (np.array([-0.40 * q, -0.10 * q]), 0.70 * q),
    ]
    random_cases = [
        (
            rng.uniform(-0.49 * q, 0.49 * q, size=2),
            float(rng.uniform(0.15 * q, 0.95 * q)),
        )
        for _ in range(48)
    ]

    for Q, E in fixed_cases + random_cases:
        out_original = GH_filter_original(Q, E, lattice)
        out_vec = GH_filter_vectorized(Q, E, lattice)
        assert _to_multiset_from_original(out_original) == _to_multiset_from_original(
            out_vec
        )
