"""
Main entry‑point that loads *your own* simulated market data from the *data/*
directory and performs a GARCH‑Vine‑Copula risk analysis on an arbitrary
number (*n*) of assets.

Author: Aryan (updated 2025‑05‑07)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copula.DataAnalyzer import DataAnalyzer
from copula.TimeSeries.GARCHVineCopula import GARCHVineCopula


from copula.DataLoader import load_returns_from_data_folder


def plot_returns(returns_df: pd.DataFrame, out_path: str = "returns_timeseries.png") -> None:
    """Quick time‑series plot of every asset's returns."""
    n_assets = returns_df.shape[1]
    plt.figure(figsize=(12, 2.5 * n_assets))
    for idx, col in enumerate(returns_df.columns, 1):
        plt.subplot(n_assets, 1, idx)
        plt.plot(returns_df.index, returns_df[col], linewidth=0.7)
        plt.title(col)
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_garch_vine_analysis(base_dir: str = "data", alpha: float = 0.05) -> None:
    """Load data, fit GARCH‑Vine Copula and print risk measures."""
    print(f"🔍 Loading data from: {base_dir}/ …")
    returns_df = load_returns_from_data_folder(base_dir)

    print("\n===== DATA SUMMARY =====")
    print(f"Assets: {', '.join(returns_df.columns)}")
    print(returns_df.describe())

    plot_returns(returns_df)

    # 🔧 Correlation diagnostics
    analyzer = DataAnalyzer(returns_df)
    corrs = analyzer.compute_correlations()
    print("\nPearson correlation matrix:\n", corrs["Pearson"].round(3))

    # 🚀 Fit GARCH‑Vine Copula
    print("\n⚙️  Fitting GARCH‑Vine Copula model …")
    gv_model = GARCHVineCopula().fit(returns_df)

    # 🛡️  Risk measures
    print(f"\n===== RISK MEASURES (alpha={alpha}) =====")
    risk = gv_model.compute_risk_measures(alpha=alpha)
    for asset, var in risk["VaR"].items():
        print(f"VaR[{asset}] = {var:.5f},  CVaR = {risk['CVaR'][asset]:.5f}")
    print(f"Portfolio VaR = {risk['Portfolio_VaR']:.5f}")
    print(f"Portfolio CVaR = {risk['Portfolio_CVaR']:.5f}")


if __name__ == "__main__":
    # The user can override the folder with an environment variable or CLI arg.
    DATA_DIR = os.getenv("DATA_DIR", "data")
    run_garch_vine_analysis(DATA_DIR)
