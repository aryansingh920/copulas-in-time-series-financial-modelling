"""
Created on 05 / 05 / 2025

@author: Aryan
Updated: 09 / 05 / 2025 – added CDF / PDF implementations
"""

from typing import Dict
import numpy as np

from copula.CopulaDistribution import CopulaDistribution


class ClaytonCopula(CopulaDistribution):
    """Bivariate Clayton copula."""

    def __init__(self) -> None:
        super().__init__(name="Clayton")

    # ──────────────────────────────────────────────────────────────────────────
    # Analytical functions
    # ──────────────────────────────────────────────────────────────────────────
    def cdf(self, u: np.ndarray, params: Dict) -> np.ndarray:
        """
        Clayton copula CDF.

        Args
        ----
        u       : array‑like, shape (..., 2) with values in (0, 1]
        params  : {'theta': float > 0}

        Returns
        -------
        np.ndarray – C(u₁,u₂)
        """
        theta = params.get("theta", 2.0)
        if theta <= 0:
            raise ValueError("θ (theta) must be > 0 for the Clayton copula.")

        u1, u2 = u[..., 0], u[..., 1]
        return np.power(np.power(u1, -theta) + np.power(u2, -theta) - 1.0,
                        -1.0 / theta)

    def pdf(self, u: np.ndarray, params: Dict) -> np.ndarray:
        """
        Clayton copula PDF.

        Args
        ----
        u       : array‑like, shape (..., 2) with values in (0, 1]
        params  : {'theta': float > 0}

        Returns
        -------
        np.ndarray – c(u₁,u₂)
        """
        theta = params.get("theta", 2.0)
        if theta <= 0:
            raise ValueError("θ (theta) must be > 0 for the Clayton copula.")

        u1, u2 = u[..., 0], u[..., 1]
        inner = np.power(u1, -theta) + np.power(u2, -theta) - 1.0
        coef = (1.0 + theta) * np.power(u1 * u2, -(1.0 + theta))
        return coef * np.power(inner, -(2.0 + 1.0 / theta))

    # (simulate() implementation from the original file remains unchanged)
