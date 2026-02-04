import numpy as np
from model import c
from S_matrix import coord_convert


class gaussian_in_state:

    def __init__(self, q0, l0, sigma):
        self.q0 = q0
        self.l0 = l0
        self.sigma = sigma

    def __call__(self, q_para, Eq, l_para, El):
        """Callable interface: in_state(q_para, Eq, l_para, El) -> value."""
        return self._eval(q_para, Eq, l_para, El)

    def _eval(self, q_para, Eq, l_para, El):
        """
        Normalized product of two 3D Gaussian distributions in k-space.

        This represents a two-photon input state where each photon has a Gaussian
        distribution centered at q0 (for photon q) and l0 (for photon l).
        Supports batch inputs in either parallel coordinates (q_para/l_para) or
        full 3D Cartesian coordinates.

        Parameters:
        -----------
        q_para, l_para : array-like
            2D parallel momentum [kx, ky] for each photon, or full 3D Cartesian
            momentum [kx, ky, kz]. Accepts shape (2,), (n, 2), (2, n) or
            (3,), (n, 3), (3, n).
        Eq, El : float or array-like
            Energy of each photon (required when using 2D parallel momentum).
        Returns:
        --------
        float or ndarray
            Normalized product of two Gaussians for each input pair.
        """
        # --- Input validation and preprocessing ---
        q0 = np.asarray(self.q0, dtype=np.float64)
        l0 = np.asarray(self.l0, dtype=np.float64)
        sigma_arr = np.asarray(self.sigma, dtype=np.float64)

        # sigma: scalar -> isotropic [sigma, sigma, sigma]; else must be length-3
        if sigma_arr.ndim == 0:
            sigma_vec = np.full(3, float(sigma_arr))
        else:
            sigma_arr = sigma_arr.reshape(-1)
            if sigma_arr.size != 3:
                raise ValueError("sigma must be a scalar or a length-3 array.")
            sigma_vec = sigma_arr

        c_val = float(c)


        def _normalize_input(k_input, expected_dim):
            """Normalize momentum input to shape (n, expected_dim). Returns (array, is_single)."""
            k_arr = np.asarray(k_input, dtype=np.float64)
            if k_arr.ndim == 1:
                if k_arr.shape[0] != expected_dim:
                    raise ValueError(f"Expected length-{expected_dim} vector.")
                return k_arr.reshape(1, expected_dim), True
            if k_arr.ndim == 2:
                if k_arr.shape[1] == expected_dim:
                    return k_arr, False
                if k_arr.shape[0] == expected_dim:
                    return k_arr.T, False
            raise ValueError(f"Expected shape (n, {expected_dim}) or ({expected_dim}, n).")

        def _as_cartesian(k_input, E):
            """Convert in-plane momentum [kx, ky] to 3D Cartesian using E."""
            k_arr = np.asarray(k_input, dtype=np.float64)
            # Parallel momentum [kx, ky]: infer kz from dispersion |k| = E/c
            k_para, is_single = _normalize_input(k_arr, 2)
            if E is None:
                raise ValueError("Eq/El must be provided when using parallel momenta.")
            E_arr = np.asarray(E, dtype=np.float64)
            if E_arr.ndim == 0:
                E_arr = np.full(k_para.shape[0], float(E_arr))
            else:
                E_arr = E_arr.reshape(-1)
                if E_arr.shape[0] != k_para.shape[0]:
                    if k_para.shape[0] == 1:
                        k_para = np.broadcast_to(k_para, (E_arr.shape[0], 2))
                        is_single = False
                    elif E_arr.shape[0] == 1:
                        E_arr = np.broadcast_to(E_arr, (k_para.shape[0],))
                    else:
                        raise ValueError("E must be scalar or match k_para batch size.")

            k_cart = coord_convert(k_para, E_arr)
            return k_cart, is_single

        # Convert both momenta to 3D Cartesian
        q_cart, q_single = _as_cartesian(q_para, Eq)
        l_cart, l_single = _as_cartesian(l_para, El)

        # Broadcast if one is single point and the other is a batch
        n_q, n_l = q_cart.shape[0], l_cart.shape[0]
        if n_q != n_l:
            if n_q == 1:
                q_cart = np.broadcast_to(q_cart, (n_l, 3))
            elif n_l == 1:
                l_cart = np.broadcast_to(l_cart, (n_q, 3))
            else:
                raise ValueError("q and l must have same batch size or be broadcastable.")

        # Product of two normalized 3D Gaussians: N * exp(-|k-k0|^2/(4*sigma^2))
        normalization = (2 * np.pi)**(-0.75) * (np.prod(sigma_vec))**(-0.5)
        gaussian_q = np.exp(-0.25 * np.sum((q_cart - q0)**2 / (sigma_vec**2), axis=1))
        gaussian_l = np.exp(-0.25 * np.sum((l_cart - l0)**2 / (sigma_vec**2), axis=1))

        values = (normalization**2) * gaussian_q * gaussian_l
        if q_single and l_single:
            return float(values[0])
        return values


