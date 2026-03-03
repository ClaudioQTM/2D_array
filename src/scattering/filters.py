"""Momentum-space filters used by scattering integral backends."""

from __future__ import annotations


import numpy as np

from model.model import c


c = float(c)

"""
def J_filter(k_para, p_para, lattice):
    COM_K_out = (k_para + p_para) / 2

    J_grid_x, J_grid_y = lattice.lattice_grid
    J_x_flat = J_grid_x.ravel()
    J_y_flat = J_grid_y.ravel()

    Jx_shifted = COM_K_out[0] + J_x_flat / 2
    Jy_shifted = COM_K_out[1] + J_y_flat / 2
    Jx_mask = (-float(lattice.q) / 2 < Jx_shifted) & (Jx_shifted < float(lattice.q) / 2)
    Jy_mask = (-float(lattice.q) / 2 < Jy_shifted) & (Jy_shifted < float(lattice.q) / 2)

    J_mask = Jx_mask & Jy_mask
    return J_x_flat[J_mask], J_y_flat[J_mask]


def GH_filter(COM_K, E, lattice):
    G_grid_x, G_grid_y = lattice.lattice_grid
    G_x_flat = G_grid_x.ravel()
    G_y_flat = G_grid_y.ravel()

    max_Delta_norm = np.sqrt(
        (np.pi / lattice.a - abs(COM_K[0])) ** 2
        + (np.pi / lattice.a - abs(COM_K[1])) ** 2
    )

    first_mask = (
        np.linalg.norm(COM_K + np.column_stack([G_x_flat, G_y_flat]), axis=1)
        <= E / c + np.sqrt(2) / 2 * max_Delta_norm
    )
    G_x_filtered = G_x_flat[first_mask]
    G_y_filtered = G_y_flat[first_mask]

    G_x_mesh, H_x_mesh = np.meshgrid(G_x_filtered, G_x_filtered, indexing="ij")
    G_y_mesh, H_y_mesh = np.meshgrid(G_y_filtered, G_y_filtered, indexing="ij")

    G_x_pairs = G_x_mesh.ravel()
    G_y_pairs = G_y_mesh.ravel()
    H_x_pairs = H_x_mesh.ravel()
    H_y_pairs = H_y_mesh.ravel()

    second_mask = (
        np.linalg.norm(
            2 * COM_K + np.column_stack([G_x_pairs + H_x_pairs, G_y_pairs + H_y_pairs]),
            axis=1,
        )
        <= E / c
    )

    return (
        G_x_pairs[second_mask],
        G_y_pairs[second_mask],
        H_x_pairs[second_mask],
        H_y_pairs[second_mask],
    )
"""


def GH_filter_original(Q, E, lattice):

    G_grid_x, G_grid_y = lattice.lattice_grid

    G_x_flat = G_grid_x.ravel()
    G_y_flat = G_grid_y.ravel()
    # first filter: ||G|| <= E/c + sqrt(2) * pi/a
    first_mask = (
        np.linalg.norm(np.column_stack([G_x_flat, G_y_flat]), axis=1)
        <= E / c + np.sqrt(2) * lattice.q / 2
    )

    G_x_filtered = G_x_flat[first_mask]
    G_y_filtered = G_y_flat[first_mask]

    # function d returns the minimum distance on each direction to 1st BZ
    def d(v):
        v = np.absolute(v) - lattice.q / 2
        return np.where(v > 0, v, 0)

    d_G_x = d(G_x_filtered)
    d_G_y = d(G_y_filtered)

    d_G = np.column_stack([d_G_x, d_G_y])
    # the minimum distance to 1st BZ of vector G, namely, min_{r in 1st BZ} ||r + G||
    d_G_norm = np.linalg.norm(d_G, axis=1)
    # second filter: d(G) <= E/c
    second_mask = d_G_norm <= E / c

    G_x_filtered2 = G_x_filtered[second_mask]
    G_y_filtered2 = G_y_filtered[second_mask]
    d_G_norm2 = d_G_norm[second_mask]
    G_filtered_pairs = np.column_stack([G_x_filtered2, G_y_filtered2])

    minimum_shift_vectors = np.array(
        [
            [0, 0],
            [-lattice.q, 0],
            [lattice.q, 0],
            [0, -lattice.q],
            [0, lattice.q],
            [-lattice.q, -lattice.q],
            [lattice.q, -lattice.q],
            [-lattice.q, lattice.q],
            [lattice.q, lattice.q],
        ]
    )
    # calculate the legitamate pair of G and H in GH double summation
    legit_pair_list = []

    for i, g in enumerate(G_filtered_pairs):
        for j, h in enumerate(G_filtered_pairs):
            # 3rd filter: d(G)+d(H) <= E/c
            if d_G_norm2[i] + d_G_norm2[j] > E / c:
                continue

            # 4th filter: min_{v in nine possible shift vectors} ||v + G + H + Q|| <= E/c
            norm_list = np.linalg.norm(minimum_shift_vectors + Q + g + h, axis=1)
            minimum_norm = np.min(norm_list)
            if minimum_norm <= E / c:
                legit_pair_list.append([g, h])

    return legit_pair_list


def GH_filter(Q, E, lattice):
    grid_cutoff = (E / c + np.sqrt(2) * lattice.q / 2) // lattice.q

    G_grid_x, G_grid_y = np.meshgrid(
        np.arange(-grid_cutoff, grid_cutoff + 1),
        np.arange(-grid_cutoff, grid_cutoff + 1),
        indexing="ij",
    )
    G_grid_x = float(lattice.q) * G_grid_x
    G_grid_y = float(lattice.q) * G_grid_y

    G_x_flat = G_grid_x.ravel()
    G_y_flat = G_grid_y.ravel()
    # first filter: ||G|| <= E/c + sqrt(2) * pi/a
    first_mask = (
        np.linalg.norm(np.column_stack([G_x_flat, G_y_flat]), axis=1)
        <= E / c + np.sqrt(2) * lattice.q / 2
    )

    G_x_filtered = G_x_flat[first_mask]
    G_y_filtered = G_y_flat[first_mask]

    # function d returns the minimum distance on each direction to 1st BZ
    def d(v):
        v = np.absolute(v) - lattice.q / 2
        return np.where(v > 0, v, 0)

    d_G_x = d(G_x_filtered)
    d_G_y = d(G_y_filtered)

    d_G = np.column_stack([d_G_x, d_G_y])
    # the minimum distance to 1st BZ of vector G, namely, min_{r in 1st BZ} ||r + G||
    d_G_norm = np.linalg.norm(d_G, axis=1)
    # second filter: d(G) <= E/c
    second_mask = d_G_norm <= E / c

    G_x_filtered2 = G_x_filtered[second_mask]
    G_y_filtered2 = G_y_filtered[second_mask]
    d_G_norm2 = d_G_norm[second_mask]
    G_filtered_pairs = np.column_stack([G_x_filtered2, G_y_filtered2])

    minimum_shift_vectors = np.array(
        [
            [0, 0],
            [-lattice.q, 0],
            [lattice.q, 0],
            [0, -lattice.q],
            [0, lattice.q],
            [-lattice.q, -lattice.q],
            [lattice.q, -lattice.q],
            [-lattice.q, lattice.q],
            [lattice.q, lattice.q],
        ]
    )
    # calculate the legitamate pair of G and H in GH double summation
    legit_pair_list = []

    for i, g in enumerate(G_filtered_pairs):
        for j, h in enumerate(G_filtered_pairs):
            # 3rd filter: d(G)+d(H) <= E/c
            if d_G_norm2[i] + d_G_norm2[j] > E / c:
                continue

            # 4th filter: min_{v in nine possible shift vectors} ||v + G + H + Q|| <= E/c
            norm_list = np.linalg.norm(minimum_shift_vectors + Q + g + h, axis=1)
            minimum_norm = np.min(norm_list)
            if minimum_norm <= E / c:
                legit_pair_list.append([g, h])

    return legit_pair_list


def GH_filter_vectorized(Q, E, lattice):
    """Vectorized GH filter returning (G,H) pairs."""
    q = float(lattice.q)
    e_over_c = float(E) / c
    e_over_c_sq = e_over_c * e_over_c
    # Upper bound used to build a finite reciprocal-lattice window.
    first_radius = e_over_c + np.sqrt(2.0) * q / 2.0
    first_radius_sq = first_radius * first_radius

    # Build candidate G points on a square integer grid scaled by q.
    grid_cutoff = int(np.floor(first_radius / q))
    grid_idx = np.arange(-grid_cutoff, grid_cutoff + 1, dtype=float)
    G_grid_x, G_grid_y = np.meshgrid(q * grid_idx, q * grid_idx, indexing="ij")

    G_x_flat = G_grid_x.ravel()
    G_y_flat = G_grid_y.ravel()

    # 1st filter: keep G vectors inside the radial preselection bound.
    # Use squared norms to avoid an unnecessary sqrt on every point.
    first_mask = (G_x_flat * G_x_flat + G_y_flat * G_y_flat) <= first_radius_sq
    G_x_filtered = G_x_flat[first_mask]
    G_y_filtered = G_y_flat[first_mask]

    # d(v): per-axis distance to the first BZ boundary (0 if already inside).
    d_G_x = np.maximum(np.abs(G_x_filtered) - q / 2.0, 0.0)
    d_G_y = np.maximum(np.abs(G_y_filtered) - q / 2.0, 0.0)
    # Elementwise 2-norm calculation: sqrt(x_1^2 + x_2^2) -> faster than np.linalg.norm
    d_G_norm = np.hypot(d_G_x, d_G_y)

    # 2nd filter: keep G whose minimum distance to the first BZ is <= E/c.
    second_mask = d_G_norm <= e_over_c
    G_x_filtered2 = G_x_filtered[second_mask]
    G_y_filtered2 = G_y_filtered[second_mask]
    d_G_norm2 = d_G_norm[second_mask]

    if G_x_filtered2.size == 0:
        return []

    # Broadcast pairwise sums (n x n) for all (G, H) without Python loops.
    # These are reused by the 3rd and 4th filters.
    sum_x = G_x_filtered2[:, None] + G_x_filtered2[None, :] + float(Q[0]) # x component of  broadcasted Q + q + G + H
    sum_y = G_y_filtered2[:, None] + G_y_filtered2[None, :] + float(Q[1]) # y component of broadcasted Q + q + G + H 

    # 3rd filter: triangle-type pruning in distance space.
    third_mask = (d_G_norm2[:, None] + d_G_norm2[None, :]) <= e_over_c

    # 4th filter: periodic image check over 9 nearest reciprocal shifts.
    # We keep a tiny loop over the 9 shifts (constant-size, cheap) and vectorize
    # over all (G, H) pairs inside each iteration.
    shift_vectors = np.array(
        [
            [0.0, 0.0],
            [-q, 0.0],
            [q, 0.0],
            [0.0, -q],
            [0.0, q],
            [-q, -q],
            [q, -q],
            [-q, q],
            [q, q],
        ]
    )
    # Track min shifted norm^2 for each (G, H) pair.
    min_norm_sq = np.full(sum_x.shape, np.inf, dtype=float)
    for shift_x, shift_y in shift_vectors:
        norm_sq = (sum_x + shift_x) ** 2 + (sum_y + shift_y) ** 2
        min_norm_sq = np.minimum(min_norm_sq, norm_sq)
    fourth_mask = min_norm_sq <= e_over_c_sq

    # Final selection and conversion back to the original format:
    # a list of [g, h] pairs, where each g/h is a length-2 vector.
    pair_mask = third_mask & fourth_mask
    i_idx, j_idx = np.nonzero(pair_mask)
    g_vectors = np.column_stack([G_x_filtered2[i_idx], G_y_filtered2[i_idx]])
    h_vectors = np.column_stack([G_x_filtered2[j_idx], G_y_filtered2[j_idx]])
    return [[g, h] for g, h in zip(g_vectors, h_vectors, strict=True)]


__all__ = ["GH_filter", "GH_filter_vectorized"]
