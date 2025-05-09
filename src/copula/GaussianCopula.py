"""
Created on 05 / 05 / 2025

@author: Aryan
Updated: 09 / 05 / 2025 – added CDF / PDF implementations
"""

from typing import Dict

import numpy as np
from scipy.stats import norm, multivariate_normal

from CopulaDistribution import CopulaDistribution


class GaussianCopula(CopulaDistribution):
    """Gaussian copula for arbitrary dimension."""

    def __init__(self) -> None:
        super().__init__(name="Gaussian")

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _corr_matrix(params: Dict, dim: int) -> np.ndarray:
        """Return/validate correlation matrix."""
        R = params.get("corr")
        if R is None:
            R = np.eye(dim)
        R = np.asarray(R, dtype=float)
        if R.shape != (dim, dim):
            raise ValueError(f"Correlation matrix must be {dim}×{dim}.")
        return R

    # ──────────────────────────────────────────────────────────────────────────
    # Analytical functions
    # ──────────────────────────────────────────────────────────────────────────
    def cdf(self, u: np.ndarray, params: Dict) -> np.ndarray:
        dim = u.shape[-1]
        R = self._corr_matrix(params, dim)
        z = norm.ppf(u.clip(1e-12, 1 - 1e-12))           # Φ⁻¹(u)
        mvn = multivariate_normal(mean=np.zeros(dim), cov=R)
        return mvn.cdf(z)

    def pdf(self, u: np.ndarray, params: Dict) -> np.ndarray:
        dim = u.shape[-1]
        R = self._corr_matrix(params, dim)
        z = norm.ppf(u.clip(1e-12, 1 - 1e-12))
        invR = np.linalg.inv(R)
        detR = np.linalg.det(R)

        exponent = -0.5 * np.sum(z @ (invR - np.eye(dim)) * z, axis=-1)
        denom = np.sqrt(detR) * np.prod(norm.pdf(z), axis=-1)
        return np.exp(exponent) / denom
