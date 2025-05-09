"""
Created on 05 / 05 / 2025
 
@author: Aryan
Updated: 09 / 05 / 2025 – added CDF / PDF implementations
"""

from typing import Dict

import numpy as np
from scipy.stats import t
try:
    # SciPy ≥ 1.11
    from scipy.stats import multivariate_t
except ImportError:  # pragma: no cover
    multivariate_t = None

from copula.CopulaDistribution import CopulaDistribution


class StudentTCopula(CopulaDistribution):
    """Student‑t copula (arbitrary dimension)."""

    def __init__(self) -> None:
        super().__init__(name="StudentT")

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _corr_matrix(params: Dict, dim: int) -> np.ndarray:
        R = params.get("corr_matrix", params.get("corr"))
        if R is None:
            R = np.eye(dim)
        R = np.asarray(R, dtype=float)
        if R.shape != (dim, dim):
            raise ValueError(f"Correlation matrix must be {dim}×{dim}.")
        return R

    # ──────────────────────────────────────────────────────────────────────────
    # Simulation
    # ──────────────────────────────────────────────────────────────────────────
    def simulate(self, n_samples: int, params: Dict) -> np.ndarray:
        """
        Simulate from the Student-t copula.
        
        Args:
            n_samples: Number of samples to generate
            params: Dictionary containing:
                   - df: Degrees of freedom (default: 4)
                   - corr_matrix or corr: Correlation matrix
                   - dimension: Dimension of the copula
                   
        Returns:
            Array of shape (n_samples, dimension) with simulated copula values
        """
        # Extract parameters
        df = params.get("df", 4)
        # Determine dimension from correlation matrix if provided, otherwise use dimension param
        if "corr_matrix" in params or "corr" in params:
            R = params.get("corr_matrix", params.get("corr"))
            dim = len(R)
        else:
            dim = params.get("dimension", 2)
            # Default to identity matrix if no correlation provided
            R = np.eye(dim)

        # Ensure R is properly formatted
        R = np.asarray(R, dtype=float)

        # Generate multivariate t random variables
        try:
            chol = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            # If matrix is not positive definite, add small value to diagonal
            R = R + np.eye(dim) * 1e-6
            chol = np.linalg.cholesky(R)

        z = np.random.standard_normal(size=(n_samples, dim))
        chi2 = np.random.chisquare(df=df, size=n_samples) / df

        # Apply the t-distribution transformation
        mvt_samples = np.dot(z, chol.T) / np.sqrt(chi2)[:, np.newaxis]

        # Transform to uniform margins using the t-CDF
        u = t.cdf(mvt_samples, df=df)

        return u

    # ──────────────────────────────────────────────────────────────────────────
    # Analytical functions
    # ──────────────────────────────────────────────────────────────────────────
    def cdf(self, u: np.ndarray, params: Dict) -> np.ndarray:
        if multivariate_t is None:  # pragma: no cover
            raise RuntimeError(
                "SciPy ≥ 1.11 is required for the t‑copula CDF.")

        dim = u.shape[-1]
        nu = params.get("df", 4)
        R = self._corr_matrix(params, dim)

        x = t.ppf(u.clip(1e-12, 1 - 1e-12), df=nu)
        return multivariate_t.cdf(x, shape=R, df=nu)

    def pdf(self, u: np.ndarray, params: Dict) -> np.ndarray:
        if multivariate_t is None:  # pragma: no cover
            raise RuntimeError(
                "SciPy ≥ 1.11 is required for the t‑copula PDF.")

        dim = u.shape[-1]
        nu = params.get("df", 4)
        R = self._corr_matrix(params, dim)

        x = t.ppf(u.clip(1e-12, 1 - 1e-12), df=nu)
        joint_pdf = multivariate_t.pdf(x, shape=R, df=nu)
        marginals = np.prod(t.pdf(x, df=nu), axis=-1)
        return joint_pdf / marginals
