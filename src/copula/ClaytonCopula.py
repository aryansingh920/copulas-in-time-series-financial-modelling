"""
Multivariate Clayton copula.

Implements the Marshall–Olkin construction so any dimension ≥ 2 is supported.
"""

from typing import Dict
import numpy as np

from copula.CopulaDistribution import CopulaDistribution


class ClaytonCopula(CopulaDistribution):
    """Clayton copula for arbitrary dimension d ≥ 2."""

    def __init__(self) -> None:
        super().__init__(name="Clayton")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def simulate(self, n_samples: int, params: Dict) -> np.ndarray:
        """
        Draw samples from a d‑dimensional Clayton copula.

        Parameters
        ----------
        n_samples : int
            Number of observations to generate.
        params : Dict
            *Required*  – ``theta``  (float > 0)  
            *Optional*  – ``dimension``  (int ≥ 2, default 2).

        Returns
        -------
        np.ndarray
            ``(n_samples, dimension)`` array of uniforms.
        """
        theta = params.get("theta", 2.0)
        d = params.get("dimension", 2)

        if theta <= 0:
            raise ValueError("θ must be > 0 for a Clayton copula.")
        if d < 2:
            raise ValueError("dimension must be at least 2.")

        # Marshall–Olkin algorithm
        # 1. latent variable  W ~ Gamma(1/θ, 1)
        w = np.random.gamma(shape=1.0 / theta, scale=1.0, size=n_samples)

        # 2. independent uniforms
        u = np.random.uniform(size=(n_samples, d))

        # 3. transform
        x = (1.0 - np.log(u) / w[:, None]) ** (-1.0 / theta)

        return x
