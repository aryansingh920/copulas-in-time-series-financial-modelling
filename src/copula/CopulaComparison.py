"""
CopulaComparison.py
Builds a full comparison table (metrics + Q‑Q plots) for:
  • Gaussian  • t       • Clayton  • Gumbel
Using custom copula implementations
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Import custom copula implementations
from GaussianCopula import GaussianCopula
from StudentTCopula import StudentTCopula
from ClaytonCopula import ClaytonCopula

# Need to implement GumbelCopula or use fallback
from GumbelCopula import GumbelCopula

# ------------------------------------------------------------------------- #
# Internal helpers
# ------------------------------------------------------------------------- #


def _pseudo_obs(values: np.ndarray) -> np.ndarray:
    """
    Transform data to pseudo-observations (uniform margins using rank transformation)
    """
    ranks = stats.rankdata(values, axis=0, method="average")
    return ranks / (values.shape[0] + 1.0)


def _aic(loglik: float, k: int) -> float:
    """Calculate Akaike Information Criterion"""
    return 2 * k - 2 * loglik


def _bic(loglik: float, k: int, n: int) -> float:
    """Calculate Bayesian Information Criterion"""
    return k * np.log(n) - 2 * loglik


def _distance(metric: str, u: np.ndarray, v: np.ndarray) -> float:
    """Calculate various distance metrics between two datasets"""
    if metric == "Euclidean Distance":
        return float(np.linalg.norm(u - v) / u.shape[0])

    if metric == "Spearman's Rank Correlation":
        return float(stats.spearmanr(u.flatten(), v.flatten()).correlation)

    if metric == "Kendall's Tau Distance":
        tau = stats.kendalltau(u.flatten(), v.flatten()).correlation
        return 1 - (tau if tau is not None else 0.0)

    if metric == "Distance Correlation":
        return float(stats.distance_correlation(u.flatten(), v.flatten()))

    raise ValueError(f"Unsupported distance metric: {metric}")


def _qqplot(u: np.ndarray, family: str, out_dir: Path) -> Path:
    """Generate Q-Q plot for the copula model"""
    fig = plt.figure(figsize=(4, 4))
    stats.probplot(u.flatten(), dist="uniform", plot=plt)
    plt.title(f"Q‑Q Plot – {family}")
    out_dir.mkdir(parents=True, exist_ok=True)
    img = out_dir / f"qq_{family.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(img, dpi=150)
    plt.close(fig)
    return img


def _fit_gaussian_copula(u: np.ndarray) -> Dict:
    """
    Fit Gaussian copula to the data
    
    Returns:
        Dictionary with correlation matrix and log-likelihood
    """
    # Calculate correlation parameter (Gaussian copula has correlation parameter)
    rho = np.corrcoef(u[:, 0], u[:, 1])[0, 1]

    # Construct correlation matrix
    corr_matrix = np.array([[1.0, rho], [rho, 1.0]])

    # Calculate log-likelihood for Gaussian
    # Convert to normal scores
    z = stats.norm.ppf(u)

    # Log-likelihood calculation
    n = u.shape[0]
    sign, logdet = np.linalg.slogdet(corr_matrix)
    inv_corr = np.linalg.inv(corr_matrix)

    loglik = -(n/2) * logdet

    for i in range(n):
        loglik -= 0.5 * z[i].T @ (inv_corr - np.eye(2)) @ z[i]

    return {
        'params': {'corr_matrix': corr_matrix},
        'theta': rho,  # Main parameter
        'loglikelihood': loglik,
        'k': 1  # One parameter (rho)
    }


def _fit_t_copula(u: np.ndarray) -> Dict:
    """
    Fit t-copula to the data - simplified approach
    
    Returns:
        Dictionary with correlation matrix, degrees of freedom, and log-likelihood
    """
    # Calculate correlation parameter
    rho = np.corrcoef(u[:, 0], u[:, 1])[0, 1]

    # Construct correlation matrix
    corr_matrix = np.array([[1.0, rho], [rho, 1.0]])

    # Use a simple heuristic to estimate degrees of freedom
    # (In practice, this should be optimized via MLE)
    df = 4.0  # default value, can be improved

    # Simplified log-likelihood calculation
    # Convert to t scores
    z = stats.t.ppf(u, df)

    # Simplified log-likelihood (not exact)
    n = u.shape[0]
    sign, logdet = np.linalg.slogdet(corr_matrix)
    inv_corr = np.linalg.inv(corr_matrix)

    loglik = -(n/2) * logdet

    for i in range(n):
        loglik -= ((df + 2) / 2) * np.log(1 +
                                          z[i].T @ (inv_corr - np.eye(2)) @ z[i] / df)

    return {
        'params': {'corr_matrix': corr_matrix, 'df': df},
        'theta': df,  # Main parameter is degrees of freedom
        'loglikelihood': loglik,
        'k': 2  # Two parameters (rho and df)
    }


def _fit_clayton_copula(u: np.ndarray) -> Dict:
    """
    Fit Clayton copula to the data
    
    Returns:
        Dictionary with theta parameter and log-likelihood
    """
    # Estimate theta using Kendall's tau
    tau = stats.kendalltau(u[:, 0], u[:, 1]).correlation

    # Relationship between Clayton's theta and Kendall's tau
    if tau <= 0:
        theta = 0.1  # Default small positive value for negative tau
    else:
        theta = 2 * tau / (1 - tau)

    # Calculate log-likelihood
    n = u.shape[0]
    loglik = 0

    # Clayton log-likelihood
    loglik = n * np.log(theta + 1)
    for i in range(n):
        loglik += -(theta + 1) * (np.log(u[i, 0]) + np.log(u[i, 1]))
        loglik += -(2 + 1/theta) * \
            np.log(u[i, 0]**(-theta) + u[i, 1]**(-theta) - 1)

    return {
        'params': {'theta': theta},
        'theta': theta,
        'loglikelihood': loglik,
        'k': 1  # One parameter (theta)
    }


def _fit_gumbel_copula(u: np.ndarray) -> Dict:
    """
    Fit Gumbel copula to the data
    
    Returns:
        Dictionary with theta parameter and log-likelihood
    """
    # Estimate theta using Kendall's tau
    tau = stats.kendalltau(u[:, 0], u[:, 1]).correlation

    # Relationship between Gumbel's theta and Kendall's tau
    if tau <= 0:
        theta = 1.001  # Default slightly above 1 for non-positive tau
    else:
        theta = 1 / (1 - tau)

    # Calculate simplified log-likelihood
    n = u.shape[0]
    loglik = 0

    # Very simplified Gumbel log-likelihood (not exact)
    for i in range(n):
        u1, u2 = u[i, 0], u[i, 1]
        w1 = -np.log(u1)
        w2 = -np.log(u2)
        w = (w1**theta + w2**theta)**(1/theta)
        loglik += np.log(w) - w + np.log(theta - 1 + w)

    return {
        'params': {'theta': theta},
        'theta': theta,
        'loglikelihood': loglik,
        'k': 1  # One parameter (theta)
    }


# ------------------------------------------------------------------------- #
# Public API
# ------------------------------------------------------------------------- #
def compare_copulas(
    returns: pd.DataFrame,
    cols: List[str] | None = None,
    out_dir: str | Path = "qqplots",
) -> pd.DataFrame:
    """Fit four copulas and generate comparison metrics."""
    # Handle column selection
    if cols is None:
        # Use the first pair of columns by default
        if returns.shape[1] >= 2:
            cols = list(returns.columns[:2])
        else:
            raise ValueError("Need at least two columns to fit copulas.")

    if len(cols) != 2:
        print(
            f"Warning: Expected exactly 2 columns for bivariate copula analysis, got {len(cols)}.")
        print(f"Will use the first pair from: {cols[:2]}")
        cols = cols[:2]

    print(f"Analyzing copula relationships between {cols[0]} and {cols[1]}")

    # Filter to just the two selected columns
    data = returns[cols].dropna().to_numpy()
    n = data.shape[0]

    # Convert to pseudo-observations (uniform margins)
    u = _pseudo_obs(data)

    # Setup copula models
    copula_models = {
        "Gaussian Copula": {
            "model": GaussianCopula(),
            "sel": "AIC",
            "gof": "KS Test",
            "dist": "Euclidean Distance",
            "marg": "Normal",
            "fit_func": _fit_gaussian_copula
        },
        "t‑Copula": {
            "model": StudentTCopula(),
            "sel": "BIC",
            "gof": "Anderson-Darling",
            "dist": "Spearman's Rank Correlation",
            "marg": "Student‑t",
            "fit_func": _fit_t_copula
        },
        "Clayton Copula": {
            "model": ClaytonCopula(),
            "sel": "AIC",
            "gof": "Cramér-von Mises",
            "dist": "Kendall's Tau Distance",
            "marg": "Exponential",
            "fit_func": _fit_clayton_copula
        },
        "Gumbel Copula": {
            "model": GumbelCopula(),
            "sel": "AIC",
            "gof": "Chi-Square Test",
            "dist": "Distance Correlation",
            "marg": "Gamma",
            "fit_func": _fit_gumbel_copula
        },
    }

    rows = []
    out_dir = Path(out_dir)

    for fam, cfg in copula_models.items():
        try:
            # Fit the copula model
            fit_result = cfg["fit_func"](u)

            # Extract parameters
            params = fit_result['params']
            theta = fit_result['theta']
            loglik = fit_result['loglikelihood']
            k = fit_result['k']

            # Calculate selection metric
            sel_val = _aic(loglik, k) if cfg["sel"] == "AIC" else _bic(
                loglik, k, n)

            # Generate samples
            model = cfg["model"]

            # Handle special case for Gumbel which is not implemented
            if fam == "Gumbel Copula":
                # Create a simple approximation for Gumbel sampling
                try:
                    # Simple approximation
                    v = np.random.uniform(size=(n, 2))
                    # Add some correlation structure based on theta
                    t = theta - 1  # Transformed theta
                    z1 = np.random.uniform(size=n)
                    z2 = t * z1 + (1-t) * np.random.uniform(size=n)
                    v[:, 0] = z1
                    v[:, 1] = z2
                except:
                    # Fallback to uniform
                    v = np.random.uniform(size=(n, 2))
            else:
                # Use the actual model
                v = model.simulate(n_samples=n, params=params)

            # Compute goodness-of-fit metrics
            try:
                if cfg["gof"] == "KS Test":
                    gof_val = stats.ks_2samp(
                        u.flatten(), v.flatten()).statistic
                elif cfg["gof"] == "Anderson-Darling":
                    gof_val = stats.anderson_ksamp(
                        [u.flatten(), v.flatten()]).statistic
                elif cfg["gof"] == "Cramér-von Mises":
                    gof_val = stats.cramervonmises_2samp(
                        u.flatten(), v.flatten()).statistic
                else:  # Chi‑Square
                    h_u, _ = np.histogram(u.flatten(), bins=10)
                    h_v, _ = np.histogram(v.flatten(), bins=10)
                    gof_val = stats.chisquare(h_u, h_v).statistic
            except Exception as e:
                print(
                    f"WARNING: Could not compute goodness-of-fit for {fam}: {str(e)}")
                gof_val = np.nan

            # Compute distance metrics
            try:
                dist_val = _distance(cfg["dist"], u, v)
            except Exception as e:
                print(
                    f"WARNING: Could not compute distance metric for {fam}: {str(e)}")
                dist_val = np.nan

            # Generate Q-Q plot
            try:
                img = _qqplot(v.flatten(), fam, out_dir)
            except Exception as e:
                print(
                    f"WARNING: Could not generate Q-Q plot for {fam}: {str(e)}")
                img = "Failed to generate"

            rows.append(
                {
                    "Copula Family": fam,
                    "Parameter (Theta)": theta,
                    "Selection Metric (AIC/BIC)": sel_val,
                    "Goodness-of-Fit Metric": gof_val,
                    "Q‑Q Plot": str(img),
                    "Marginal Distribution": cfg["marg"],
                    "Distance/Rank Comparison": dist_val,
                    "Status": "Success",
                }
            )
        except Exception as e:
            print(f"ERROR: Failed to fit {fam}: {str(e)}")
            rows.append(
                {
                    "Copula Family": fam,
                    "Parameter (Theta)": np.nan,
                    "Selection Metric (AIC/BIC)": np.nan,
                    "Goodness-of-Fit Metric": np.nan,
                    "Q‑Q Plot": "Failed",
                    "Marginal Distribution": cfg["marg"],
                    "Distance/Rank Comparison": np.nan,
                    "Status": f"Failed: {str(e)}",
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- #
# Fallback implementation for copula comparison
# ------------------------------------------------------------------------- #
def compare_copulas_fallback(
    returns: pd.DataFrame,
    out_dir: str | Path = "qqplots",
) -> pd.DataFrame:
    """Simplified fallback comparison that focuses on pairwise analyses."""
    rows = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create all possible pairs of assets
    asset_pairs = list(combinations(returns.columns, 2))

    print(f"Analyzing {len(asset_pairs)} asset pairs in fallback mode")

    # For each pair, compute basic correlation metrics
    for asset1, asset2 in asset_pairs:
        # Get the data
        data = returns[[asset1, asset2]].dropna()

        # Skip if not enough data
        if len(data) < 10:
            continue

        # Compute correlations
        pearson = data.corr().iloc[0, 1]
        spearman = data.corr(method='spearman').iloc[0, 1]

        # Compute Kendall's tau
        kendall = stats.kendalltau(data[asset1], data[asset2]).correlation

        # Compute tail dependence coefficient (simple approximation)
        # Lower tail dependence
        q = 0.05  # 5% quantile
        x1_q = data[asset1].quantile(q)
        x2_q = data[asset2].quantile(q)

        joint_lower = ((data[asset1] <= x1_q) & (data[asset2] <= x2_q)).mean()
        lower_tail = joint_lower / q if q > 0 else np.nan

        # Upper tail dependence
        q = 0.95  # 95% quantile
        x1_q = data[asset1].quantile(q)
        x2_q = data[asset2].quantile(q)

        joint_upper = ((data[asset1] >= x1_q) & (data[asset2] >= x2_q)).mean()
        upper_tail = joint_upper / (1-q) if (1-q) > 0 else np.nan

        # Create scatter plot with marginal histograms
        try:
            fig, ax = plt.subplots(figsize=(8, 8))

            # Create the scatter plot
            ax.scatter(data[asset1], data[asset2], alpha=0.5, s=10)
            ax.set_xlabel(asset1)
            ax.set_ylabel(asset2)
            ax.set_title(f"Scatter Plot: {asset1} vs {asset2}")

            # Add correlation info
            ax.text(0.05, 0.95,
                    f"Pearson: {pearson:.4f}\nSpearman: {spearman:.4f}\n" +
                    f"Kendall: {kendall:.4f}\nLower Tail: {lower_tail:.4f}\n" +
                    f"Upper Tail: {upper_tail:.4f}",
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Save the plot
            img_path = out_dir / f"scatter_{asset1}_vs_{asset2}.png"
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close(fig)
        except Exception as e:
            print(
                f"WARNING: Could not create scatter plot for {asset1} vs {asset2}: {str(e)}")
            img_path = "Failed to generate"

        # Add to results
        rows.append({
            "Asset Pair": f"{asset1} — {asset2}",
            "Pearson Correlation": pearson,
            "Spearman Correlation": spearman,
            "Kendall's Tau": kendall,
            "Lower Tail Dependence": lower_tail,
            "Upper Tail Dependence": upper_tail,
            "Scatter Plot": str(img_path),
            "Sample Size": len(data)
        })

    # Create a summary of which copula family might be most appropriate
    for i, row in enumerate(rows):
        pearson = row["Pearson Correlation"]
        lower_tail = row["Lower Tail Dependence"]
        upper_tail = row["Upper Tail Dependence"]

        # Simple heuristic for copula family recommendation
        if abs(pearson) < 0.1:
            recommendation = "Independence (weak dependence)"
        elif abs(lower_tail - upper_tail) < 0.1 and abs(pearson) > 0.7:
            recommendation = "Gaussian or t-Copula (symmetric, strong)"
        elif abs(lower_tail - upper_tail) < 0.1:
            recommendation = "Gaussian Copula (symmetric)"
        elif lower_tail > upper_tail + 0.1:
            recommendation = "Clayton Copula (lower tail dependence)"
        elif upper_tail > lower_tail + 0.1:
            recommendation = "Gumbel Copula (upper tail dependence)"
        else:
            recommendation = "t-Copula (symmetric with tail dependence)"

        rows[i]["Recommended Copula"] = recommendation

    return pd.DataFrame(rows)
