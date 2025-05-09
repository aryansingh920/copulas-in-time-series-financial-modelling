"""
Multivariate Gumbel Copula implementation.
Supports arbitrary dimension (d ≥ 2) using the Marshall–Olkin algorithm.

Created on 05/05/2025
@author: Aryan

Relative Path: src/copula/GumbelCopula.py
"""

from typing import Dict
import numpy as np
from copula.CopulaDistribution import CopulaDistribution


class GumbelCopula(CopulaDistribution):
    """Multivariate Gumbel Copula (d ≥ 2)."""

    def __init__(self):
        super().__init__(name="Gumbel")

    # ──────────────────────────────────────────────────────────────────────────
    # Analytical functions
    # ──────────────────────────────────────────────────────────────────────────
    def cdf(self, u: np.ndarray, params: Dict) -> np.ndarray:
        """
        Generate samples from a d-dimensional Gumbel copula.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        params : Dict
            *Required* - `theta` (float ≥ 1)
            *Optional* - `dimension` (int ≥ 2, default 2).

        Returns
        -------
        np.ndarray
            (n_samples, dimension) array of uniforms.
        """
        theta = params.get("theta", 1.5)
        d = params.get("dimension", 2)

        if theta < 1:
            raise ValueError("Theta must be ≥ 1 for Gumbel copula.")
        if d < 2:
            raise ValueError("Dimension must be at least 2.")

        # Step 1: Generate latent variable V using stable distribution
        v = np.random.gamma(1.0 / theta, 1.0, size=n_samples)

        # Step 2: Generate independent uniforms
        u = np.random.uniform(0, 1, (n_samples, d))

        # Step 3: Transform using Marshall–Olkin method
        w = (-np.log(u) / v[:, None]) ** (1 / theta)

        return np.exp(-w)
