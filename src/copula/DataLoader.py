"""
Utility to load simulated market datasets stored in the *data/* folder.

Each sub‑directory inside *data/* represents **one asset** and **must** contain:

- **simulation_steps.csv**  — with at least the columns *timestamp* and *price*
- **order_book_snapshots.csv** (ignored here)
- **config_used.json**       (ignored here)

The loader builds a single *pandas.DataFrame* of **log‑returns**, one column
per asset and the *timestamp* as the index, perfectly aligned across all assets.

Example directory layout
------------------------
data/
├── Market_A/
│   ├── simulation_steps.csv
│   ├── order_book_snapshots.csv
│   └── config_used.json
├── Market_B/
│   └── …
└── NVIDIA_HypeCycle_Fixed/
    └── …

If some assets have missing timestamps the loader performs an *inner‑join*
and drops any rows with at least one missing value (to avoid spurious NaNs
in the copula/GARCH estimation).

Author: Aryan (2025‑05‑07)
"""

from __future__ import annotations

import os
from typing import Literal, List

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def _compute_returns(price: pd.Series, return_type: Literal["log", "pct"] = "log") -> pd.Series:
    """Compute log‑ or percentage returns from a price series."""
    if return_type == "log":
        returns = np.log(price).diff()
    elif return_type == "pct":
        # returns = price.pct_change()
        returns = price.pct_change().dropna()
    else:
        raise ValueError("return_type must be 'log' or 'pct'")
    return returns.dropna()


def load_returns_from_data_folder(
    base_dir: str = "data",
    return_type: Literal["log", "pct"] = "log",
    min_obs: int | None = 50,
) -> pd.DataFrame:
    """Scan *base_dir* and build an aligned DataFrame of returns.

    Parameters
    ----------
    base_dir
        Path to the directory that contains one sub‑directory per asset.
    return_type
        'log' (default) ⇒ :math:`\log(P_t) - \log(P_{t-1})`  
        'pct' ⇒ :math:`(P_t/P_{t-1}) - 1`
    min_obs
        If given, assets with fewer than *min_obs* observations are skipped.

    Returns
    -------
    pd.DataFrame
        Index is the *timestamp*, columns are asset names (folder names).
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"{base_dir!r} not found")

    asset_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not asset_dirs:
        raise RuntimeError(f"No asset directories found in {base_dir!r}")

    all_returns: list[pd.DataFrame] = []

    for asset in sorted(asset_dirs):
        sim_path = os.path.join(base_dir, asset, "simulation_steps.csv")
        if not os.path.isfile(sim_path):
            print(f"⚠️  Skipping {asset}: simulation_steps.csv not found")
            continue

        df = pd.read_csv(sim_path)

        if {"timestamp", "price"}.issubset(df.columns) is False:
            print(f"⚠️  Skipping {asset}: required columns missing")
            continue

        # Ensure sorted by timestamp
        df = df.sort_values("timestamp")

        # Compute returns
        returns = _compute_returns(df["price"], return_type)
        if min_obs is not None and len(returns) < min_obs:
            print(f"⚠️  Skipping {asset}: only {len(returns)} observations (<{min_obs})")
            continue

        returns.name = asset
        all_returns.append(returns)

    if not all_returns:
        raise RuntimeError("No valid assets were loaded")

    # Align on the timestamp index using inner join to keep only common dates
    returns_df = pd.concat(all_returns, axis=1, join="inner")

    # Final sanity check
    returns_df = returns_df.dropna(how="any")
    if returns_df.empty:
        raise RuntimeError("Resulting returns DataFrame is empty after alignment – check timestamps")

    returns_df = returns_df.dropna(how="any")
    if returns_df.empty:
        raise RuntimeError(
            "Resulting returns DataFrame is empty after alignment – check timestamps")

    # 🚨 Sanity check: Print correlations and standard deviations
    print("\n===== Sanity Check =====")
    print("Correlation matrix:")
    print(returns_df.corr())
    print("\nStandard deviations:")
    print(returns_df.std())

    plt.hist(returns_df.values.flatten(), bins=50, alpha=0.7)

    plt.title("Distribution of Log-Returns")
    plt.show()
    return returns_df
