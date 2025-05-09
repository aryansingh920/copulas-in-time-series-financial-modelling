"""
Created on 05 / 05 / 2025

@author: Aryan
Updated: 09 / 05 / 2025 – added CDF / PDF implementations
"""

from typing import Dict

import numpy as np

from copula.CopulaDistribution import CopulaDistribution


class GumbelCopula(CopulaDistribution):
    """Bivariate Gumbel copula."""

    def __init__(self) -> None:
        super().__init__(name="Gumbel")

    # ──────────────────────────────────────────────────────────────────────────
    # Analytical functions
    # ──────────────────────────────────────────────────────────────────────────
    def cdf(self, u: np.ndarray, params: Dict) -> np.ndarray:
        """
        Gumbel copula CDF.

        θ ≥ 1.

        C(u,v) = exp{ −[(−ln u)^θ + (−ln v)^θ]^{1/θ} }
        """
        theta = params.get("theta", 2.0)
        if theta < 1:
            raise ValueError("θ (theta) must be ≥ 1 for the Gumbel copula.")

        u1, u2 = u[..., 0], u[..., 1]
        a = (-np.log(u1)) ** theta
        b = (-np.log(u2)) ** theta
        return np.exp(-np.power(a + b, 1.0 / theta))

    def pdf(self, u: np.ndarray, params: Dict) -> np.ndarray:
        """
        Gumbel copula PDF.

        c(u,v) = C(u,v) · [ (−ln u·−ln v)^{θ−1} ·
                  {1 + (θ−1)( (−ln u)^θ + (−ln v)^θ )^{1/θ} } ] / (u v)
        """
        theta = params.get("theta", 2.0)
        if theta < 1:
            raise ValueError("θ (theta) must be ≥ 1 for the Gumbel copula.")

        u1, u2 = u[..., 0], u[..., 1]
        ln_u1, ln_u2 = -np.log(u1), -np.log(u2)
        S = (ln_u1 ** theta + ln_u2 ** theta) ** (1.0 / theta)

        C = np.exp(-S)
        part1 = (ln_u1 * ln_u2) ** (theta - 1)
        part2 = 1.0 + (theta - 1.0) * S
        return C * part1 * part2 / (u1 * u2)
