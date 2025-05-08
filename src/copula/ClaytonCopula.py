"""
Created on 05/05/2025

@author: Aryan

Filename: ClaytonCopula.py

Relative Path: src/copula/ClaytonCopula.py
"""

import numpy as np
from typing import Dict
from copula.CopulaDistribution import CopulaDistribution
from scipy.stats import gamma

class ClaytonCopula(CopulaDistribution):
    """Clayton copula implementation (multivariate)."""

    def __init__(self):
        """Initialize a Clayton copula."""
        super().__init__(name="Clayton")

    def simulate(self, n_samples: int, params: Dict) -> np.ndarray:
        """
        Simulate samples from a multivariate Clayton copula.
        
        Args:
            n_samples: Number of samples to generate
            params: Dictionary containing:
                - theta: Dependency parameter (> 0)
                - dimension: Number of dimensions (from corr_matrix shape)
        
        Returns:
            np.ndarray: Uniform samples from Clayton copula
        """
        theta = params.get('theta', 2.0)
        dim = params.get('dimension', 2)

        if theta <= 0:
            raise ValueError("Theta must be > 0 for Clayton copula.")

        # Generate gamma random variable
        gamma_rv = gamma.rvs(1/theta, size=n_samples)
        gamma_rv = gamma_rv.reshape(-1, 1)

        # Generate independent uniform random variables
        u = np.random.uniform(0, 1, (n_samples, dim))

        # Apply Clayton copula transformation
        uniform_samples = (1 + theta * (-np.log(u)) / gamma_rv) ** (-1/theta)

        return uniform_samples
