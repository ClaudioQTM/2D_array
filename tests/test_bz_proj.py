import numpy as np

from scattering import BZ_proj
from smatrix import square_lattice


def test_bz_proj_recovers_original_vector_after_lattice_shifts():
    """Shifting by integer multiples of q should not change BZ projection."""
    rng = np.random.default_rng(1234)
    q = float(square_lattice.q)

    # Keep points strictly inside the first BZ to avoid boundary ambiguity.
    base_vectors = rng.uniform(-0.49 * q, 0.49 * q, size=(100, 2))
    shifts = rng.integers(-8, 9, size=(100, 2))
    shifted_vectors = base_vectors + shifts * q

    projected = np.array([BZ_proj(v, square_lattice) for v in shifted_vectors])

    assert np.allclose(projected, base_vectors, atol=1e-12, rtol=0.0)
