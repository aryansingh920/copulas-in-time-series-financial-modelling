"""
Created on 05/05/2025

@author: Aryan

Filename: CopulaModel.py

Relative Path: src/copula/CopulaModel.py
"""

import pandas as pd

from typing import List,  Dict

from src.Marginal import Marginal
from CopulaDistribution import CopulaDistribution

class CopulaModel:
    """Main class for copula-based modeling."""

    def __init__(self, copula: CopulaDistribution, marginals: List[Marginal], dimension: int):
        """
        Initialize a copula model.
        
        Args:
            copula: A copula distribution
            marginals: List of marginal distributions
            dimension: Dimension of the copula (number of variables)
        """
        self.copula = copula
        self.marginals = marginals
        self.dimension = dimension

        # Ensure the number of marginals matches the dimension
        if len(marginals) != dimension:
            raise ValueError(
                f"Number of marginals ({len(marginals)}) must match the dimension ({dimension})")

    def simulate(self, n_samples: int, copula_params: Dict) -> pd.DataFrame:
        """
        Simulate samples from the copula model.
        
        Args:
            n_samples: Number of samples to generate
            copula_params: Parameters for the copula
            
        Returns:
            DataFrame with samples from the joint distribution
        """
        # Generate samples from the copula
        uniform_samples = self.simulate(n_samples, copula_params)

        # Transform using the marginal distributions
        data = {}
        for i, marginal in enumerate(self.marginals):
            data[marginal.name] = marginal.inverse_cdf(uniform_samples[:, i])

        return pd.DataFrame(data)
