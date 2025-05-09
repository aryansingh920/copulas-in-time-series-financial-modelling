"""
Created on 05/05/2025
Last updated on 09/05/2025 – added robust error handling & input validation

@author: Aryan

Filename: GARCHVineCopula.py
Relative Path: src/copula/TimeSeries/GARCHVineCopula.py
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

from copula.TimeSeries.FinancialCopulaModel import FinancialCopulaModel
from copula.TimeSeries.GARCHModel import GARCHModel
from copula.TimeSeries.VineCopula import VineCopula


class GARCHVineCopula(FinancialCopulaModel):
    """
    GARCH‑based Vine Copula for capturing volatility dynamics and complex
    inter‑asset dependencies in high‑dimensional settings.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        super().__init__(
            name="GARCH-based Vine Copula",
            description="Volatility + complex dependency",
            use_case="High-dimensional asset modeling",
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.garch_models: List[Tuple[str, GARCHModel]] = []
        self.vine_copula: VineCopula | None = None
        self.asset_names: List[str] | None = None
        self.standardized_residuals: pd.DataFrame | None = None
        self.original_data: pd.DataFrame | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────────────────
    def fit(self, data: pd.DataFrame, **kwargs) -> "GARCHVineCopula":
        """
        Fit the GARCH‑Vine Copula model to return data.

        Parameters
        ----------
        data : DataFrame
            Asset returns (columns = assets).
        **kwargs
            garch_params    – dict of per‑asset parameter overrides
            copula_families – list of pair‑copula families for the vine
        """
        self._validate_input_data(data)
        self.original_data = data.copy()
        self.asset_names = list(data.columns)
        n_assets = len(self.asset_names)

        garch_params: Dict[str, Dict] = kwargs.get("garch_params", {})
        copula_families = kwargs.get("copula_families")

        # ------------------------------------------------------------------
        # 1.  Fit univariate GARCH models
        # ------------------------------------------------------------------
        self.logger.info("Fitting GARCH models to %d assets …", n_assets)
        std_resid = pd.DataFrame(index=data.index)

        for asset in self.asset_names:
            params = garch_params.get(asset, {})
            try:
                model = GARCHModel(
                    omega=params.get("omega", 0.01),
                    alpha=params.get("alpha", 0.10),
                    beta=params.get("beta", 0.80),
                )
                model.fit(data[asset].values)
            except Exception as exc:
                self.logger.exception(
                    "GARCH fit failed for %s: %s", asset, exc)
                raise RuntimeError(
                    f"GARCH fit failed for {asset}: {exc}") from exc

            self.garch_models.append((asset, model))
            std_resid[asset] = model.residuals

        self.standardized_residuals = std_resid

        # ------------------------------------------------------------------
        # 2.  Fit the vine copula
        # ------------------------------------------------------------------
        self.logger.info("Fitting Vine Copula to standardized residuals …")
        try:
            self.vine_copula = VineCopula(n_assets, copula_families)
            self.vine_copula.fit(std_resid)
        except Exception as exc:
            self.logger.exception("Vine copula fit failed: %s", exc)
            raise RuntimeError(f"Vine copula fit failed: {exc}") from exc

        self.is_fitted = True
        self.logger.info("GARCH‑Vine Copula fitting completed.")
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Simulation
    # ──────────────────────────────────────────────────────────────────────
    def simulate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic asset returns from the fitted model.

        Parameters
        ----------
        n_samples          – number of simulated rows
        seed               – RNG seed (optional)
        forecast_horizon   – steps ahead for GARCH volatility forecast
        innovation_dist    – ignored for now (placeholder)
        """
        self._check_fitted()

        seed = kwargs.get("seed")
        if seed is not None:
            np.random.seed(seed)

        horizon = kwargs.get("forecast_horizon", 1)

        # 1. Copula layer → correlated uniforms
        u = self.vine_copula.simulate(n_samples)

        # 2. Inverse‑CDF to standard normals
        z = pd.DataFrame(index=range(n_samples))
        for idx, asset in enumerate(self.asset_names):
            z[asset] = norm.ppf(u[f"U{idx+1}"])

        # 3. Feed through GARCH forecasts
        sim_ret = pd.DataFrame(index=range(n_samples))
        for asset, model in self.garch_models:
            vol_path = model.forecast_volatility(horizon)
            sigma = vol_path[-1]  # use last horizon step
            sim_ret[asset] = model.mean + sigma * z[asset]

        return sim_ret

    # ──────────────────────────────────────────────────────────────────────
    # Risk measures
    # ──────────────────────────────────────────────────────────────────────
    def compute_risk_measures(self, **kwargs) -> Dict:
        """
        Compute VaR and CVaR per asset + equal‑weighted portfolio.

        Parameters
        ----------
        alpha             – tail probability (default 0.05)
        forecast_horizon  – horizon for simulation
        n_simulations     – simulation count
        """
        self._check_fitted()

        alpha = kwargs.get("alpha", 0.05)
        horizon = kwargs.get("forecast_horizon", 1)
        n_sim = kwargs.get("n_simulations", 10_000)

        sims = self.simulate(
            n_samples=n_sim,
            forecast_horizon=horizon,
            seed=kwargs.get("seed"),
        )

        var = {}
        cvar = {}
        for col in self.asset_names:
            cutoff = np.percentile(sims[col], alpha * 100.0)
            var[col] = cutoff
            cvar[col] = sims[col][sims[col] <= cutoff].mean()

        weights = np.full(len(self.asset_names), 1.0 / len(self.asset_names))
        port_ret = sims.dot(weights)
        port_cutoff = np.percentile(port_ret, alpha * 100.0)

        return {
            "VaR": var,
            "CVaR": cvar,
            "Portfolio_VaR": port_cutoff,
            "Portfolio_CVaR": port_ret[port_ret <= port_cutoff].mean(),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics / plotting
    # ──────────────────────────────────────────────────────────────────────
    def plot_volatility_surface(self, **kwargs) -> plt.Figure:
        """
        Visualize historical and forecast volatilities for each asset.
        """
        self._check_fitted()

        horizon = kwargs.get("forecast_horizon", 20)
        n_assets = len(self.garch_models)
        fig, axes = plt.subplots(n_assets, 1, figsize=(10, 3 * n_assets))

        if n_assets == 1:  # keep iterable
            axes = [axes]

        for ax, (asset, model) in zip(axes, self.garch_models):
            hist_vol = model.conditional_volatility
            fcast_vol = model.forecast_volatility(horizon)

            ax.plot(hist_vol, label="Historical")
            ax.plot(
                range(len(hist_vol), len(hist_vol) + horizon),
                fcast_vol,
                "--",
                label="Forecast",
            )
            ax.set_title(f"Volatility – {asset}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Volatility")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.tight_layout()
        return fig

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_input_data(data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame.")
        if data.isna().any().any():
            raise ValueError("Input data contains NaNs – please clean them.")
        if data.shape[1] < 2:
            raise ValueError("Need at least two assets for a copula model.")
