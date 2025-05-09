"""
Created on 05/05/2025
Last updated on 09/05/2025 – added compute_risk_measures()

@author: Aryan

Filename: VineCopula.py
Relative Path: src/copula/TimeSeries/VineCopula.py
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


class VineCopula:
    """
    **Simplified** R‑vine copula implementation.  For production work you should
    swap this out for a full library, but this is sufficient for testing.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────
    def __init__(self, n_variables: int, copula_families: List[str] | None = None):
        self.n_variables = n_variables
        self.n_edges = n_variables * (n_variables - 1) // 2

        if copula_families is None:
            self.copula_families = ["gaussian"] * self.n_edges
        else:
            if len(copula_families) != self.n_edges:
                raise ValueError(
                    f"Expected {self.n_edges} copula families, got {len(copula_families)}"
                )
            self.copula_families = copula_families

        self.parameters: dict[tuple[int, int], dict] = {}
        self.tree_structure: list = []
        self.is_fitted = False

    # ──────────────────────────────────────────────────────────────────────
    # Fitting (very lightweight)
    # ──────────────────────────────────────────────────────────────────────
    def fit(self, data: pd.DataFrame, method: str = "spearman") -> "VineCopula":
        # Empirical CDF transform
        u = data.rank(axis=0, pct=True, method="average")

        corr = u.corr(method=method).abs()

        # Build maximum‑spanning first tree via Prim's algorithm
        rem = set(range(self.n_variables))
        tree: list[tuple[int, int, float]] = []

        # initial edge
        i, j = divmod(corr.values.argmax(), self.n_variables)
        tree.append((i, j, corr.iat[i, j]))
        rem.remove(i)
        visited = {i}

        while rem:
            best = (-1.0, None)  # (corr, edge)
            for vi in visited:
                for vj in rem:
                    if corr.iat[vi, vj] > best[0]:
                        best = (corr.iat[vi, vj], (vi, vj, corr.iat[vi, vj]))
            _, edge = best
            tree.append(edge)
            visited.add(edge[1])
            rem.remove(edge[1])

        self.tree_structure.append(tree)

        # Store simplified Gaussian parameters
        for k, (v1, v2, c) in enumerate(tree):
            self.parameters[(v1, v2)] = {
                "type": self.copula_families[k],
                "param": 2 * np.sin(np.pi * c / 6),  # Fisher transform
            }

        self.is_fitted = True
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Simulation (simplified)
    # ──────────────────────────────────────────────────────────────────────
    def simulate(self, n_samples: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .simulate().")

        u = np.random.rand(n_samples, self.n_variables)

        for v1, v2, _ in self.tree_structure[0]:
            rho = self.parameters[(v1, v2)]["param"]
            z1 = norm.ppf(u[:, v1])
            z2 = norm.ppf(u[:, v2])
            z2 = rho * z1 + np.sqrt(1 - rho**2) * z2
            u[:, v2] = norm.cdf(z2)

        cols = [f"U{i+1}" for i in range(self.n_variables)]
        return pd.DataFrame(u, columns=cols)

    # ──────────────────────────────────────────────────────────────────────
    # NEW  – Risk measures
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_risk_measures(sim_ret: np.ndarray, alpha: float = 0.95) -> dict[str, float]:
        """
        Compute equal‑weighted portfolio VaR & CVaR from simulated *simple* returns.
        """
        if sim_ret.ndim != 2:
            raise ValueError("sim_ret must be 2‑D (samples × assets).")

        port_ret = sim_ret.mean(axis=1)  # equal weights
        port_ret.sort()                  # ascending

        idx = int((1 - alpha) * len(port_ret))
        var = -port_ret[idx]
        cvar = -port_ret[:idx].mean()
        return {"VaR": float(var), "CVaR": float(cvar)}
