"""
Created on 05 / 05 / 2025

@author: Aryan
Updated: 09 / 05 / 2025 – added CDF / PDF implementations
"""

from typing import Dict

import numpy as np
from scipy.stats import t
try:
    # SciPy ≥ 1.11
    from scipy.stats import multivariate_t
except ImportError:  # pragma: no cover
    multivariate_t = None

from CopulaDistribution import CopulaDistribution


class StudentTCopula(CopulaDistribution):
    """Student‑t copula (arbitrary dimension)."""

    def __init__(self) -> None:
        super().__init__(name="StudentT")

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _corr_matrix(params: Dict, dim: int) -> np.ndarray:
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
        if multivariate_t is None:  # pragma: no cover
            raise RuntimeError(
                "SciPy ≥ 1.11 is required for the t‑copula CDF.")

        dim = u.shape[-1]
        nu = params.get("df", 4)
        R = self._corr_matrix(params, dim)

        x = t.ppf(u.clip(1e-12, 1 - 1e-12), df=nu)
        return multivariate_t.cdf(x, shape=R, df=nu)

    def pdf(self, u: np.ndarray, params: Dict) -> np.ndarray:
        if multivariate_t is None:  # pragma: no cover
            raise RuntimeError(
                "SciPy ≥ 1.11 is required for the t‑copula PDF.")

        dim = u.shape[-1]
        nu = params.get("df", 4)
        R = self._corr_matrix(params, dim)

        x = t.ppf(u.clip(1e-12, 1 - 1e-12), df=nu)
        joint_pdf = multivariate_t.pdf(x, shape=R, df=nu)
        marginals = np.prod(t.pdf(x, df=nu), axis=-1)
        return joint_pdf / marginals
