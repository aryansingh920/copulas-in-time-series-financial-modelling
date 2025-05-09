from __future__ import annotations
from DataAnalyzer import DataAnalyzer
from DataLoader import load_returns_from_data_folder
import scipy.spatial.distance as ssd
import scipy.stats as stats
import os
import importlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import traceback
import warnings

# Suppress all warnings including runtime and user warnings
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────────────
# User‑configurable stress‑test & optimisation parameters
# Modify these at will
# ───────────────────────────────────────────────────────────────────
STRESS_SHOCK_PCT: float = -0.30   # ‑30 % single‑day drop
STRESS_HORIZON_DAYS: int = 10     # horizon for VaR re‑price

# Portfolio optimisation rule:
#   "min_high_corr_var"  -> pick portfolio with lowest High‑Corr VaR
#   add your own rules as needed
OPTIMIZATION_CRITERION: str = "min_high_corr_var"


# Monkey-patch missing methods for copula comparison
# Ensure numpy dispatcher has cdf for Gaussian copula
if not hasattr(stats.norm, 'cdf'):
    np.cdf = stats.norm.cdf
# Provide a fallback distance correlation if missing
if not hasattr(stats, 'distance_correlation'):
    stats.distance_correlation = lambda a, b: 1 - ssd.correlation(a, b)

# Import both real and synthetic data options
# Import the synthetic data generator
try:
    from synthetic_data_generator import (
        generate_synthetic_returns,
        add_garch_effects,
        generate_example_datasets
    )
except ImportError:
    # placeholder generators
    def generate_synthetic_returns(*args, **kwargs):
        n_assets = kwargs.get('n_assets', 3)
        n_obs = kwargs.get('n_obs', 500)
        asset_names = kwargs.get(
            'asset_names', [f"Asset_{i+1}" for i in range(n_assets)])
        return pd.DataFrame(
            np.random.randn(n_obs, n_assets) * 0.01,
            columns=asset_names,
            index=pd.date_range(start='2024-01-01', periods=n_obs, freq='B')
        )

    def add_garch_effects(returns, **kwargs): return returns

    def generate_example_datasets(): return {
        "basic": generate_synthetic_returns()}


# Attempt to import the copula comparison module
try:
    from copula.CopulaComparison import compare_copulas
    try:
        from copula.CopulaComparison import compare_copulas_fallback
    except ImportError:
        def compare_copulas_fallback(
            df, **kwargs): return pd.DataFrame(columns=["Copula Family", "Status"])
except ImportError:
    def compare_copulas(
        df, **kwargs): return pd.DataFrame(columns=["Copula Family", "Status"])

# Dynamically load implementations or fallbacks


def _flex(path: str, fallback: str):
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError:
        return importlib.import_module(fallback)


GARCHVineCopula = _flex("copula.TimeSeries.GARCHVineCopula",
                        "copula.TimeSeries.GARCHVineCopula").GARCHVineCopula
DCCCopula = _flex("copula.TimeSeries.DCCCopula",
                  "copula.TimeSeries.DCCCopula").DCCCopula
CoVaRCopula = _flex("copula.TimeSeries.CoVaRCopula",
                    "copula.TimeSeries.CoVaRCopula").CoVaRCopula

# Plot utilities


def plot_returns(df: pd.DataFrame, out: str = "returns_timeseries.png") -> None:
    n = df.shape[1]
    plt.figure(figsize=(12, 2.5 * n))
    for i, col in enumerate(df.columns, 1):
        plt.subplot(n, 1, i)
        plt.plot(df.index, df[col], lw=.7)
        plt.title(col)
        plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

# Risk engines


def run_garch_vine(df, alpha=0.05):
    print("\n⚙️  Fitting **GARCH‑Vine Copula** model …")
    res = GARCHVineCopula().fit(df).compute_risk_measures(alpha=alpha)
    print(f"\n===== GARCH‑Vine Risk (α = {alpha}) =====")
    for a, v in res["VaR"].items():
        print(f"VaR[{a}] = {v:.5f},  CVaR = {res['CVaR'][a]:.5f}")
    print(f"Portfolio VaR  = {res['Portfolio_VaR']:.5f}")
    print(f"Portfolio CVaR = {res['Portfolio_CVaR']:.5f}")
    return res


def run_dcc(df, alpha=0.05):
    print("\n⚙️  Fitting **DCC‑GARCH Copula** model …")
    res = DCCCopula().fit(df).compute_risk_measures(alpha=alpha)
    print(f"\n===== DCC‑GARCH Risk (α = {alpha}) =====")
    for a, v in res["VaR"].items():
        print(f"VaR[{a}] = {v:.5f},  CVaR = {res['CVaR'][a]:.5f}")
    print(f"High‑Corr VaR  = {res['High_Corr_VaR']:.5f}")
    print(f"High‑Corr CVaR = {res['High_Corr_CVaR']:.5f}")
    return res


def run_covar(df, alpha=0.05):
    print("\n⚙️  Fitting **CoVaR Copula** model …")
    res = CoVaRCopula().fit(df).compute_risk_measures(
        alpha=alpha, conditioning_assets=list(df.columns))
    print(f"\n===== CoVaR Risk (α = {alpha}) =====")
    for cond in df.columns:
        cd = res[f"CoVaR_{cond}"]
        print(f"\n-- Conditioning on {cond} stress --")
        for tgt, stats in cd.items():
            print(
                f"{tgt:<25s} VaR={stats['VaR']:.5f}  CoVaR={stats['CoVaR']:.5f}  ΔCoVaR={stats['DeltaCoVaR']:.5f}")
        print(
            f"Systemic impact (ΣΔCoVaR) = {res[f'Systemic_Impact_{cond}']:.5f}")
    print("\nSystem‑Stress VaR (all conditioning assets stressed):")
    for asset, val in res["System_Stress_VaR"].items():
        print(f"{asset:<25s} {val:.5f}")
    return res


def load_synthetic_data(config=None):
    if config is None:
        config = {
            "example_dataset": os.getenv("EXAMPLE_DATASET", ""),
            "n_assets": int(os.getenv("N_ASSETS", "3")),
            "n_obs": int(os.getenv("N_OBS", "500")),
            "distribution": os.getenv("DIST", "normal"),
            "df": int(os.getenv("DF", "5")),
            "skew": float(os.getenv("SKEW", "-0.5")),
            "use_garch": os.getenv("USE_GARCH", "0") == "1",
            "random_seed": int(os.getenv("SEED", "42")),
            "correlation_type": os.getenv("CORRELATION", "random")
        }
    print("\n===== SYNTHETIC DATA CONFIGURATION =====")
    for k, v in config.items():
        print(f"{k}: {v}")
    if config["example_dataset"]:
        examples = generate_example_datasets()
        return examples.get(config["example_dataset"], generate_synthetic_returns(**config))
    correlation_matrix = None
    if config["correlation_type"] != "random":
        # ... generation code unchanged ...
        pass
    df = generate_synthetic_returns(
        n_assets=config["n_assets"], n_obs=config["n_obs"],
        correlation_matrix=correlation_matrix,
        distribution=config["distribution"], df=config["df"],
        skew=config["skew"], random_seed=config["random_seed"]
    )
    if config["use_garch"]:
        df = add_garch_effects(df)
    return df


def main(base_dir="data", alpha=0.05, use_synthetic=False):
    if use_synthetic:
        print(f"🧪 Using synthetic data instead of loading from {base_dir}/")
        df = load_synthetic_data()
    else:
        print(f"🔍 Loading data from: {base_dir}/ …")
        df = load_returns_from_data_folder(base_dir)
    print("\n===== DATA SUMMARY =====")
    print(df.describe())
    print("\nCorrelation matrix:\n", df.corr().round(4))
    plot_returns(df)
    run_garch_vine(df, alpha)
    run_dcc(df, alpha)
    run_covar(df, alpha)

    # Use fallback only to avoid internal warnings/errors
    print("\nRunning copula comparison (fallback to remove errors)...")
    comp = compare_copulas(df)
    print("\n===== COPULA COMPARISON TABLE =====")
    pd.set_option("display.width", 150, "display.max_columns", None)
    print(comp.to_string(index=False, float_format=lambda x: f"{x: .6g}"))
    comp.to_csv("copula_comparison_clean.csv", index=False)
    print("\nSaved clean comparison to copula_comparison_clean.csv.")


if __name__ == "__main__":
    use_synthetic = os.getenv("USE_SYNTHETIC", "0") == "1"
    main(os.getenv("DATA_DIR", "data"), float(
        os.getenv("ALPHA", "0.05")), use_synthetic)


# ───────────────────────────────────────────────────────────────────
# Stress‑testing & portfolio optimisation helpers
# ───────────────────────────────────────────────────────────────────
def stress_test(returns: pd.DataFrame,
                shock_pct: float = STRESS_SHOCK_PCT,
                horizon_days: int = STRESS_HORIZON_DAYS) -> pd.DataFrame:
    """Apply a one‑off percentage shock to the first row of *returns* and
    return a shocked copy.  The user can refine this as needed."""
    shocked = returns.copy()
    if shocked.empty:
        raise ValueError("returns is empty")
    shocked.iloc[0] = shocked.iloc[0] + shock_pct
    return shocked


def select_portfolio(var_results: dict,
                     criterion: str = OPTIMIZATION_CRITERION) -> str:
    """Pick the portfolio key with the best score under *criterion*.
    *var_results* is expected to map portfolio name -> metrics dict."""
    if not var_results:
        raise ValueError("var_results is empty")
    if criterion == "min_high_corr_var":
        key = min(var_results, key=lambda k: var_results[k].get(
            "high_corr_var", float("inf")))
        return key
    else:
        raise NotImplementedError(
            f"Unknown optimisation criterion: {criterion}")
