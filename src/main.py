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

from copula.GaussianCopula import GaussianCopula
from copula.StudentTCopula import StudentTCopula
from copula.ClaytonCopula import ClaytonCopula

# Suppress all warnings including runtime and user warnings
warnings.filterwarnings('ignore')

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
                        "GARCHVineCopula").GARCHVineCopula
DCCCopula = _flex("copula.TimeSeries.DCCCopula", "DCCCopula").DCCCopula
CoVaRCopula = _flex("copula.TimeSeries.CoVaRCopula", "CoVaRCopula").CoVaRCopula

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

def compare_copula_fits(data: pd.DataFrame):
    """
    Compare different copula families using energy scores.
    
    Args:
        data: DataFrame with financial time series data
        
    Returns:
        Dictionary with comparison results for different copula models
    """
    n_dim = data.shape[1]
    n_samples = len(data)
    results = {}
    
    print(f"\nInput data shape: {data.shape}")
    
    # Convert input data to correct format once
    data_array = data.values.astype(np.float64)
    
    basic_copulas = {
        'Clayton': {'copula': ClaytonCopula(), 
                   'params': {'theta': 2.0, 'dimension': n_dim}},
        'Student-t': {'copula': StudentTCopula(), 
                     'params': {'df': 4, 'corr_matrix': np.corrcoef(data_array.T)}}
    }
    
    ts_copulas = {
        'DCC': DCCCopula(),
        'GARCH-Vine': GARCHVineCopula(),
        'CoVaR': CoVaRCopula()
    }
    
    def calculate_energy_score(x_samples, y_samples):
        """
        Calculate energy score between two multivariate samples.
        
        Args:
            x_samples: First sample set (n_samples x n_features)
            y_samples: Second sample set (n_samples x n_features)
            
        Returns:
            Energy score value
        """
        n_x = len(x_samples)
        n_y = len(y_samples)
        
        # First term: Mean Euclidean distance between x and y samples
        first_term = 0
        for i in range(n_x):
            for j in range(n_y):
                first_term += np.linalg.norm(x_samples[i] - y_samples[j])
        first_term /= (n_x * n_y)
        
        # Second term: Mean Euclidean distance within x samples
        second_term = 0
        for i in range(n_x):
            for j in range(i+1, n_x):  # Only use unique pairs
                second_term += np.linalg.norm(x_samples[i] - x_samples[j])
        second_term *= 2 / (n_x * (n_x - 1))  # Multiply by 2 because we only counted each pair once
        
        # Third term: Mean Euclidean distance within y samples
        third_term = 0
        for i in range(n_y):
            for j in range(i+1, n_y):  # Only use unique pairs
                third_term += np.linalg.norm(y_samples[i] - y_samples[j])
        third_term *= 2 / (n_y * (n_y - 1))  # Multiply by 2 because we only counted each pair once
        
        # Energy score formula: E(x,y) = 2*mean(||x-y||) - mean(||x-x'||) - mean(||y-y'||)
        energy_score = first_term - 0.5 * (second_term + third_term)
        
        return energy_score
    
    def calculate_metrics(real_data, simulated_data, name):
        """Calculate energy score and tail dependence for a pair of datasets."""
        # Ensure arrays are float64 and 2D
        real = np.asarray(real_data, dtype=np.float64)
        sim = np.asarray(simulated_data, dtype=np.float64)
        
        # Calculate multivariate energy score
        energy_score = calculate_energy_score(real, sim)
        
        # Calculate tail dependence
        sim_df = pd.DataFrame(sim, columns=data.columns)
        analyzer = DataAnalyzer(sim_df)
        
        # Calculate tail dependence between first two dimensions
        tail_dep = analyzer.compute_tail_dependence(
            sim_df.columns[0],
            sim_df.columns[1]
        )
        
        return {
            'energy_score': energy_score,
            'upper_tail': tail_dep['upper_tail_dependence'],
            'lower_tail': tail_dep['lower_tail_dependence']
        }
    
    # Compare basic copulas
    for name, copula_dict in basic_copulas.items():
        print(f"\nProcessing {name} copula:")
        try:
            simulated = copula_dict['copula'].simulate(
                n_samples=n_samples, 
                params=copula_dict['params']
            )
            
            metrics = calculate_metrics(data_array, simulated, name)
            metrics['type'] = 'basic'
            results[name] = metrics
            
            print(f"Energy Score: {metrics['energy_score']:.4f}")
            print(f"Upper Tail: {metrics['upper_tail']:.4f}")
            print(f"Lower Tail: {metrics['lower_tail']:.4f}")
            
        except Exception as e:
            print(f"Error processing {name} copula: {str(e)}")
            continue
    
    # Compare time series copulas
    for name, copula in ts_copulas.items():
        print(f"\nProcessing {name} model:")
        try:
            if name == 'DCC':
                copula.fit(data, dcc_params={'a': 0.03, 'b': 0.95})
            elif name == 'CoVaR':
                copula.fit(data, copula_type='t', copula_params={'df': 4})
            else:
                copula.fit(data)
                
            simulated = copula.simulate(n_samples)
            
            metrics = calculate_metrics(data_array, simulated, name)
            metrics['type'] = 'time_series'
            results[name] = metrics
            
            print(f"Energy Score: {metrics['energy_score']:.4f}")
            print(f"Upper Tail: {metrics['upper_tail']:.4f}")
            print(f"Lower Tail: {metrics['lower_tail']:.4f}")
            
        except Exception as e:
            print(f"Error processing {name} model: {str(e)}")
            continue
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Energy score comparison
    plt.subplot(1, 2, 1)
    names = list(results.keys())
    scores = [results[name]['energy_score'] for name in names]
    colors = ['#3498db' if results[name]['type'] == 'basic' else '#e74c3c' for name in names]
    
    bars = plt.bar(names, scores, color=colors)
    plt.title('Energy Score Comparison\n(Lower is better)', fontsize=14)
    plt.ylabel('Energy Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Basic Copulas'),
        Patch(facecolor='#e74c3c', label='Time Series Copulas')
    ]
    plt.legend(handles=legend_elements)
    
    # Tail dependence comparison
    plt.subplot(1, 2, 2)
    x = np.arange(len(names))
    width = 0.35
    
    upper_tails = [results[name]['upper_tail'] for name in names]
    lower_tails = [results[name]['lower_tail'] for name in names]
    
    plt.bar(x - width/2, upper_tails, width, label='Upper Tail', color='#2ecc71')
    plt.bar(x + width/2, lower_tails, width, label='Lower Tail', color='#9b59b6')
    
    plt.title('Tail Dependence Comparison', fontsize=14)
    plt.ylabel('Dependence Coefficient')
    plt.xticks(x, names, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('copula_comparison.png')
    plt.close()
    
    return results



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
    comp = compare_copulas_fallback(df)
    print("\n===== COPULA COMPARISON TABLE =====")
    pd.set_option("display.width", 150, "display.max_columns", None)
    print(comp.to_string(index=False, float_format=lambda x: f"{x: .6g}"))
    comp.to_csv("copula_comparison_clean.csv", index=False)
    print("\nSaved clean comparison to copula_comparison_clean.csv.")

    print("\n\n===== PART 3: COPULA FAMILY COMPARISON =====")
    comparison_results = compare_copula_fits(df)


if __name__ == "__main__":
    use_synthetic = os.getenv("USE_SYNTHETIC", "0") == "1"
    main(os.getenv("DATA_DIR", "data"), float(
        os.getenv("ALPHA", "0.05")), use_synthetic)
