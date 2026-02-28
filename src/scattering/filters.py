"""Momentum-space filters used by scattering integral backends."""

from __future__ import annotations
import numpy as np
from model import c

c = float(c)


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


__all__ = ["J_filter", "GH_filter"]
