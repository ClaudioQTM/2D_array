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


def GH_filter(Q, E, lattice):

    G_grid_x, G_grid_y = lattice.lattice_grid
    
    G_x_flat = G_grid_x.ravel()
    G_y_flat = G_grid_y.ravel()
    # first filter: ||G|| <= E/c + sqrt(2) * pi/a
    first_mask = (
        np.linalg.norm(np.column_stack([G_x_flat, G_y_flat]), axis=1)
        <= E / c + np.sqrt(2) * lattice.q/2
  
    )

    G_x_filtered = G_x_flat[first_mask]
    G_y_filtered = G_y_flat[first_mask]



    # function d returns the minimum distance on each direction to 1st BZ
    def d(v):
        v = np.absolute(v) - lattice.q/2
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

    minimum_shift_vectors  = np.array([[0,0],
                [-lattice.q,0],
                [lattice.q,0],
                [0,-lattice.q],
                [0,lattice.q],
                [-lattice.q,-lattice.q],
                [lattice.q,-lattice.q],
                [-lattice.q,lattice.q],
                [lattice.q,lattice.q]])
    # calculate the legitamate pair of G and H in GH double summation
    legit_pair_list = []
    
    for i,g in enumerate(G_filtered_pairs):
        for j,h in enumerate(G_filtered_pairs):
            # 3rd filter: d(G)+d(H) <= E/c
            if d_G_norm2[i] + d_G_norm2[j] > E / c:
                continue

            # 4th filter: min_{v in nine possible shift vectors} ||v + G + H + Q|| <= E/c
            norm_list = np.linalg.norm(minimum_shift_vectors + Q + g + h,axis = 1)
            minimum_norm = np.min(norm_list)
            if minimum_norm <= E / c:
                legit_pair_list.append([g, h])


    

    return legit_pair_list





__all__ = ["GH_filter"]
