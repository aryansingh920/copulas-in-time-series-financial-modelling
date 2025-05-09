import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.linalg import cholesky
import os

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_assets = 3  # Number of assets/time series - fixed to 3 to match the matrices defined in functions
n_days = 1258  # Number of days
start_date = datetime(2020, 1, 1)  # Start date
start_prices = [100, 70, 50]  # Starting prices for assets

# Create directory for output files
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to generate dates


def generate_dates(start_date, n_days):
    dates = []
    current_date = start_date
    while len(dates) < n_days:
        # Skip weekends
        if current_date.weekday() < 5:  # Monday to Friday
            dates.append(current_date.strftime('%m/%d/%Y %H:%M:%S'))
        current_date = current_date + timedelta(days=1)
    return dates


# Generate dates
dates = generate_dates(start_date, n_days)

# 1. GARCH(1,1) Model Simulation


def simulate_garch(n_days, omega=0.0001, alpha=0.1, beta=0.8, start_price=100):
    # Initial variance
    h = omega / (1 - alpha - beta)
    returns = np.zeros(n_days)
    prices = np.zeros(n_days)
    volatilities = np.zeros(n_days)
    prices[0] = start_price
    volatilities[0] = np.sqrt(h)

    for t in range(1, n_days):
        # Update variance
        if t > 1:
            h = omega + alpha * returns[t-1]**2 + beta * h

        # Save volatility
        volatilities[t] = np.sqrt(h)

        # Generate return
        z = np.random.normal(0, 1)
        returns[t] = np.sqrt(h) * z

        # Calculate price
        prices[t] = prices[t-1] * np.exp(returns[t])

    return prices, returns, volatilities

# 2. DCC (Dynamic Conditional Correlation) Model Simulation


def simulate_dcc(n_days, n_assets):
    # GARCH parameters for each asset
    omega_values = [0.00005, 0.00008, 0.00006]
    alpha_values = [0.09, 0.11, 0.10]
    beta_values = [0.88, 0.86, 0.87]

    # DCC parameters
    a = 0.05
    b = 0.93

    # Initialize arrays
    returns = np.zeros((n_days, n_assets))
    h = np.zeros((n_days, n_assets))
    prices = np.zeros((n_days, n_assets))
    # Store correlation matrices
    correlations = np.zeros((n_days, n_assets, n_assets))
    volatilities = np.zeros((n_days, n_assets))  # Store volatilities

    # Set initial prices
    for i in range(n_assets):
        prices[0, i] = start_prices[i]

    # Initial conditional variances
    for i in range(n_assets):
        h[0, i] = omega_values[i] / (1 - alpha_values[i] - beta_values[i])
        volatilities[0, i] = np.sqrt(h[0, i])

    # Unconditional correlation matrix
    Q_bar = np.array([[1.0, 0.3, 0.5],
                      [0.3, 1.0, 0.2],
                      [0.5, 0.2, 1.0]])

    # Initialize dynamic correlation matrix
    Q = Q_bar.copy()
    # Set initial correlation matrix
    correlations[0] = Q_bar.copy()

    # Generate multivariate time series with DCC
    for t in range(1, n_days):
        # Update GARCH variances
        for i in range(n_assets):
            if t > 1:
                h[t, i] = omega_values[i] + alpha_values[i] * \
                    returns[t-1, i]**2 + beta_values[i] * h[t-1, i]
            else:
                h[t, i] = h[0, i]

            # Store volatility
            volatilities[t, i] = np.sqrt(h[t, i])

        # Standard deviations
        std_devs = np.sqrt(h[t])
        D = np.diag(std_devs)
        D_inv = np.diag(1/std_devs)

        # Update Q matrix
        if t > 1:
            epsilon = returns[t-1] / np.sqrt(h[t-1])
            epsilon = epsilon.reshape(-1, 1)
            Q = (1 - a - b) * Q_bar + a * np.dot(epsilon, epsilon.T) + b * Q

        # Normalize Q to get correlation matrix
        Q_diag = np.diag(1/np.sqrt(np.diag(Q)))
        R = np.dot(np.dot(Q_diag, Q), Q_diag)

        # Store correlation matrix
        correlations[t] = R.copy()

        # Compute covariance matrix
        Sigma = np.dot(np.dot(D, R), D)

        # Generate correlated random variables
        L = cholesky(Sigma, lower=True)
        z = np.random.normal(0, 1, n_assets)
        returns[t] = np.dot(L, z)

        # Update prices
        prices[t] = prices[t-1] * np.exp(returns[t])

    return prices, returns, volatilities, correlations

# 3. Vine Copula Simulation (C-Vine structure)


def simulate_vine(n_days, n_assets):
    # GARCH parameters for each asset (margins)
    omega_values = [0.00004, 0.00007, 0.00005]
    alpha_values = [0.08, 0.10, 0.09]
    beta_values = [0.89, 0.87, 0.88]

    # Initialize arrays
    returns = np.zeros((n_days, n_assets))
    h = np.zeros((n_days, n_assets))
    prices = np.zeros((n_days, n_assets))
    volatilities = np.zeros((n_days, n_assets))  # Store volatilities
    copula_params = np.zeros((n_days, 3))  # Store the 3 copula parameters

    # Set initial prices
    for i in range(n_assets):
        prices[0, i] = start_prices[i]

    # Initial conditional variances
    for i in range(n_assets):
        h[0, i] = omega_values[i] / (1 - alpha_values[i] - beta_values[i])
        volatilities[0, i] = np.sqrt(h[0, i])

    # Vine copula parameters (Gaussian copula with fixed correlations in a C-vine structure)
    # Correlations between first variable and others
    rho_1_2 = 0.6
    rho_1_3 = 0.4
    # Conditional correlation between 2 and 3 given 1
    rho_2_3_1 = 0.3

    # Initial copula parameters
    copula_params[0] = [rho_1_2, rho_1_3, rho_2_3_1]

    # Generate multivariate time series with Vine copula
    for t in range(1, n_days):
        # Update GARCH variances
        for i in range(n_assets):
            if t > 1:
                h[t, i] = omega_values[i] + alpha_values[i] * \
                    returns[t-1, i]**2 + beta_values[i] * h[t-1, i]
            else:
                h[t, i] = h[0, i]

            # Store volatility
            volatilities[t, i] = np.sqrt(h[t, i])

        # Simple time-varying copula parameters (optional)
        # Making copula parameters slightly time-varying for demonstration
        if t > 1:
            # Add small random variation to copula parameters
            rho_1_2 = max(0.1, min(0.9, rho_1_2 + 0.01 * np.random.normal()))
            rho_1_3 = max(0.1, min(0.9, rho_1_3 + 0.01 * np.random.normal()))
            rho_2_3_1 = max(
                0.1, min(0.9, rho_2_3_1 + 0.01 * np.random.normal()))

        # Store copula parameters
        copula_params[t] = [rho_1_2, rho_1_3, rho_2_3_1]

        # C-Vine copula simulation
        # Generate uniform variables
        u = np.random.uniform(0, 1, n_assets)

        # First variable - direct from uniform
        z1 = stats.norm.ppf(u[0])
        returns[t, 0] = z1 * np.sqrt(h[t, 0])

        # Second variable - correlated with first
        z2 = rho_1_2 * z1 + np.sqrt(1 - rho_1_2**2) * stats.norm.ppf(u[1])
        returns[t, 1] = z2 * np.sqrt(h[t, 1])

        # Third variable - C-vine structure
        # First condition on z1
        cond_mean_3_given_1 = rho_1_3 * z1
        cond_var_3_given_1 = 1 - rho_1_3**2

        # Then condition on z2 given z1
        z3 = cond_mean_3_given_1 + np.sqrt(cond_var_3_given_1) * (
            rho_2_3_1 * (z2 - rho_1_2 * z1) / np.sqrt(1 - rho_1_2**2) +
            np.sqrt(1 - rho_2_3_1**2) * stats.norm.ppf(u[2])
        )
        returns[t, 2] = z3 * np.sqrt(h[t, 2])

        # Update prices
        prices[t] = prices[t-1] * np.exp(returns[t])

    return prices, returns, volatilities, copula_params

# 4. CoVaR Model Simulation


def simulate_covar(n_days, n_assets):
    # Initialize arrays
    returns = np.zeros((n_days, n_assets))
    prices = np.zeros((n_days, n_assets))
    volatilities = np.zeros((n_days, n_assets))  # Store asset volatilities
    system_vol = np.zeros(n_days)  # Store system volatility
    # Store correlation matrices
    correlations = np.zeros((n_days, n_assets, n_assets))

    # Set initial prices
    for i in range(n_assets):
        prices[0, i] = start_prices[i]

    # Parameters for CoVaR model
    # Base volatilities
    sigma_base = np.array([0.01, 0.015, 0.012])

    # Initial volatilities
    for i in range(n_assets):
        volatilities[0, i] = sigma_base[i]

    # Contagion matrix (how asset j affects asset i's risk)
    contagion = np.array([
        [1.0, 0.3, 0.4],  # Asset 1's volatility affected by others
        [0.5, 1.0, 0.2],  # Asset 2's volatility affected by others
        [0.6, 0.3, 1.0]   # Asset 3's volatility affected by others
    ])

    # Time-varying dependency parameter
    dependency_factor = 0.7

    # System-wide volatility factor (common to all assets)
    system_vol[0] = 0.01

    # Base correlation matrix
    corr_base = np.array([
        [1.0, 0.4, 0.3],
        [0.4, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])

    # Store initial correlation matrix
    correlations[0] = corr_base.copy()

    # AR(1) process for system volatility
    for t in range(1, n_days):
        system_vol[t] = 0.002 + 0.85 * system_vol[t-1] + \
            0.05 * np.random.normal(0, 1)
        if system_vol[t] < 0.001:  # ensure positive volatility
            system_vol[t] = 0.001

        # Asset-specific volatilities affected by past returns
        vol_adjustment = np.ones(n_assets)
        if t > 1:
            # Increase volatility if there was a large negative return
            for i in range(n_assets):
                if returns[t-1, i] < -0.01:  # threshold for negative return
                    # Increase volatility for all assets based on contagion matrix
                    vol_adjustment += 0.3 * \
                        contagion[:, i] * abs(returns[t-1, i])

        # Current volatilities (adjusted by system vol and contagion)
        current_vols = sigma_base * vol_adjustment * (1 + system_vol[t])

        # Store asset volatilities
        for i in range(n_assets):
            volatilities[t, i] = current_vols[i]

        # Correlation increases in high volatility periods
        vol_factor = min(3, 1 + system_vol[t] * 5)  # Limit the increase
        current_corr = corr_base.copy()

        # Increase correlations in high volatility
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Move correlation toward 1 in high volatility
                current_corr[i, j] = current_corr[j, i] = min(
                    0.95,
                    corr_base[i, j] + (1 - corr_base[i, j]) *
                    (vol_factor - 1) / 2
                )

        # Store correlation matrix
        correlations[t] = current_corr.copy()

        # Create covariance matrix
        cov_matrix = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(n_assets):
                cov_matrix[i, j] = current_vols[i] * \
                    current_vols[j] * current_corr[i, j]

        # Generate correlated returns
        L = cholesky(cov_matrix, lower=True)
        z = np.random.normal(0, 1, n_assets)
        returns[t] = np.dot(L, z)

        # Update prices
        prices[t] = prices[t-1] * np.exp(returns[t])

    return prices, returns, volatilities, system_vol, correlations


# Create necessary directories for each model type and asset
model_types = ["garch", "dcc", "vine", "covar"]
for model in model_types:
    for i in range(1, n_assets + 1):
        # Fix: Use consistent directory naming
        if model == "dcc" or model == "vine" or model == "covar":
            dir_path = f"{output_dir}/simulated_{model}_asset_{i}"
        else:
            dir_path = f"{output_dir}/{model}_asset_{i}"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# Generate time series for each model
print("Generating GARCH time series...")
for i in range(n_assets):
    garch_prices, garch_returns, garch_volatilities = simulate_garch(
        n_days, start_price=start_prices[i])

    # Create DataFrame with prices
    df_price = pd.DataFrame({
        'timestamp': dates,
        'price': garch_prices
    })

    # Create DataFrame with simulation steps (returns, volatilities)
    df_steps = pd.DataFrame({
        'timestamp': dates,
        'price': garch_prices,
        'return': garch_returns,
        'volatility': garch_volatilities
    })

    # Save to CSV
    asset_dir = f"{output_dir}/garch_asset_{i+1}"
    df_price.to_csv(f"{asset_dir}/prices.csv", index=False)
    df_steps.to_csv(f"{asset_dir}/simulation_steps.csv", index=False)

print("Generating DCC time series...")
dcc_prices, dcc_returns, dcc_volatilities, dcc_correlations = simulate_dcc(
    n_days, n_assets)
for i in range(n_assets):
    # Create price DataFrame
    df_price = pd.DataFrame({
        'timestamp': dates,
        'price': dcc_prices[:, i]
    })

    # Create simulation steps DataFrame
    df_steps = pd.DataFrame({
        'timestamp': dates,
        'price': dcc_prices[:, i],
        'return': dcc_returns[:, i],
        'volatility': dcc_volatilities[:, i]
    })

    # For each asset, save correlation with other assets
    for j in range(n_assets):
        df_steps[f'correlation_with_asset_{j+1}'] = [corr[i, j]
                                                     for corr in dcc_correlations]

    # Save to CSV - fix the directory path
    asset_dir = f"{output_dir}/simulated_dcc_asset_{i+1}"
    df_price.to_csv(f"{asset_dir}/prices.csv", index=False)
    df_steps.to_csv(f"{asset_dir}/simulation_steps.csv", index=False)

print("Generating Vine Copula time series...")
vine_prices, vine_returns, vine_volatilities, vine_params = simulate_vine(
    n_days, n_assets)
for i in range(n_assets):
    # Create price DataFrame
    df_price = pd.DataFrame({
        'timestamp': dates,
        'price': vine_prices[:, i]
    })

    # Create simulation steps DataFrame
    df_steps = pd.DataFrame({
        'timestamp': dates,
        'price': vine_prices[:, i],
        'return': vine_returns[:, i],
        'volatility': vine_volatilities[:, i]
    })

    # Add copula parameters
    df_steps['rho_1_2'] = vine_params[:, 0]
    df_steps['rho_1_3'] = vine_params[:, 1]
    df_steps['rho_2_3_1'] = vine_params[:, 2]

    # Save to CSV - fix the directory path
    asset_dir = f"{output_dir}/simulated_vine_asset_{i+1}"
    df_price.to_csv(f"{asset_dir}/prices.csv", index=False)
    df_steps.to_csv(f"{asset_dir}/simulation_steps.csv", index=False)

print("Generating CoVaR time series...")
covar_prices, covar_returns, covar_volatilities, covar_system_vol, covar_correlations = simulate_covar(
    n_days, n_assets)
for i in range(n_assets):
    # Create price DataFrame
    df_price = pd.DataFrame({
        'timestamp': dates,
        'price': covar_prices[:, i]
    })

    # Create simulation steps DataFrame
    df_steps = pd.DataFrame({
        'timestamp': dates,
        'price': covar_prices[:, i],
        'return': covar_returns[:, i],
        'volatility': covar_volatilities[:, i],
        'system_volatility': covar_system_vol
    })

    # For each asset, save correlation with other assets
    for j in range(n_assets):
        df_steps[f'correlation_with_asset_{j+1}'] = [corr[i, j]
                                                     for corr in covar_correlations]

    # Save to CSV - fix the directory path
    asset_dir = f"{output_dir}/simulated_covar_asset_{i+1}"
    df_price.to_csv(f"{asset_dir}/prices.csv", index=False)
    df_steps.to_csv(f"{asset_dir}/simulation_steps.csv", index=False)

print(
    f"All simulations completed. CSV files saved in '{output_dir}' directory.")
