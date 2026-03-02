"""Compatibility exports for the moved model package."""

from __future__ import annotations

from .model import (
    EMField,
    SquareLattice,
    c,
    epsilon_0,
    hbar,
    k_space_summation,
    mu_0,
    real_space_summation,
    self_energy,
)

__all__ = [
    "EMField",
    "SquareLattice",
    "c",
    "epsilon_0",
    "hbar",
    "k_space_summation",
    "mu_0",
    "real_space_summation",
    "self_energy",
    "gaussian_in_state",
]


def __getattr__(name: str):
    if name == "gaussian_in_state":
        # Lazy import prevents model<->smatrix circular import at module import time.
        from .input_states import gaussian_in_state

        return gaussian_in_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
