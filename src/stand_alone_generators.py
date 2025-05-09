import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.linalg import cholesky
import os

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_days = 1258  # Number of days
start_date = datetime(2020, 1, 1)  # Start date
start_price = 100  # Starting price for assets

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

# 2. DCC (Dynamic Conditional Correlation) Model Simulation - simplified to one asset


def simulate_dcc(n_days, start_price=100):
    # GARCH parameters
    omega = 0.00005
    alpha = 0.09
    beta = 0.88

    # Initialize arrays
    returns = np.zeros(n_days)
    h = np.zeros(n_days)
    prices = np.zeros(n_days)
    volatilities = np.zeros(n_days)

    # Set initial price
    prices[0] = start_price

    # Initial conditional variance
    h[0] = omega / (1 - alpha - beta)
    volatilities[0] = np.sqrt(h[0])

    # Generate time series with DCC-like characteristics
    for t in range(1, n_days):
        # Update GARCH variance
        if t > 1:
            h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
        else:
            h[t] = h[0]

        # Store volatility
        volatilities[t] = np.sqrt(h[t])

        # Generate return with DCC-like characteristics
        # Adding slight autocorrelation to simulate correlation effects
        if t > 1:
            previous_impact = 0.2 * returns[t-1]
        else:
            previous_impact = 0

        # Generate return with time-varying variance
        z = np.random.normal(0, 1)
        returns[t] = previous_impact + np.sqrt(h[t]) * z

        # Update price
        prices[t] = prices[t-1] * np.exp(returns[t])

    return prices, returns, volatilities

# 3. Vine Copula Simulation - simplified to one asset


def simulate_vine(n_days, start_price=100):
    # GARCH parameters
    omega = 0.00004
    alpha = 0.08
    beta = 0.89

    # Initialize arrays
    returns = np.zeros(n_days)
    h = np.zeros(n_days)
    prices = np.zeros(n_days)
    volatilities = np.zeros(n_days)

    # Set initial price
    prices[0] = start_price

    # Initial conditional variance
    h[0] = omega / (1 - alpha - beta)
    volatilities[0] = np.sqrt(h[0])

    # Parameter for skewness (to simulate vine copula characteristics)
    skew_param = 3

    # Generate time series with vine copula-like characteristics
    for t in range(1, n_days):
        # Update GARCH variance
        if t > 1:
            h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
        else:
            h[t] = h[0]

        # Store volatility
        volatilities[t] = np.sqrt(h[t])

        # Generate return with skewed distribution to simulate vine copula
        z = stats.t.rvs(df=skew_param)  # t-distribution for fat tails
        returns[t] = np.sqrt(h[t]) * z

        # Update price
        prices[t] = prices[t-1] * np.exp(returns[t])

    return prices, returns, volatilities

# 4. CoVaR Model Simulation - simplified to one asset


def simulate_covar(n_days, start_price=100):
    # Initialize arrays
    returns = np.zeros(n_days)
    prices = np.zeros(n_days)
    volatilities = np.zeros(n_days)
    system_vol = np.zeros(n_days)  # System volatility

    # Set initial price
    prices[0] = start_price

    # Base volatility
    sigma_base = 0.01
    volatilities[0] = sigma_base

    # System-wide volatility factor
    system_vol[0] = 0.01

    # AR(1) process for system volatility
    for t in range(1, n_days):
        # Update system volatility with AR(1) process
        system_vol[t] = 0.002 + 0.85 * system_vol[t-1] + \
            0.05 * np.random.normal(0, 1)
        if system_vol[t] < 0.001:  # ensure positive volatility
            system_vol[t] = 0.001

        # Asset volatility affected by system volatility and previous returns
        vol_adjustment = 1.0
        if t > 1:
            # Increase volatility if there was a large negative return (asymmetric effect)
            if returns[t-1] < -0.01:
                vol_adjustment += 0.3 * abs(returns[t-1])

        # Current volatility
        current_vol = sigma_base * vol_adjustment * (1 + system_vol[t])
        volatilities[t] = current_vol

        # Generate return
        z = np.random.normal(0, 1)
        returns[t] = current_vol * z

        # Update price
        prices[t] = prices[t-1] * np.exp(returns[t])

    return prices, returns, volatilities, system_vol


# Create necessary directories for each model type
model_types = ["garch", "dcc", "vine", "covar"]
for model in model_types:
    dir_path = f"{output_dir}/{model}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Generate time series for each model
print("Generating GARCH time series...")
garch_prices, garch_returns, garch_volatilities = simulate_garch(
    n_days, start_price=start_price)

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
garch_dir = f"{output_dir}/garch"
df_price.to_csv(f"{garch_dir}/prices.csv", index=False)
df_steps.to_csv(f"{garch_dir}/simulation_steps.csv", index=False)

print("Generating DCC time series...")
dcc_prices, dcc_returns, dcc_volatilities = simulate_dcc(
    n_days, start_price=start_price)

# Create price DataFrame
df_price = pd.DataFrame({
    'timestamp': dates,
    'price': dcc_prices
})

# Create simulation steps DataFrame
df_steps = pd.DataFrame({
    'timestamp': dates,
    'price': dcc_prices,
    'return': dcc_returns,
    'volatility': dcc_volatilities
})

# Save to CSV
dcc_dir = f"{output_dir}/dcc"
df_price.to_csv(f"{dcc_dir}/prices.csv", index=False)
df_steps.to_csv(f"{dcc_dir}/simulation_steps.csv", index=False)

print("Generating Vine Copula time series...")
vine_prices, vine_returns, vine_volatilities = simulate_vine(
    n_days, start_price=start_price)

# Create price DataFrame
df_price = pd.DataFrame({
    'timestamp': dates,
    'price': vine_prices
})

# Create simulation steps DataFrame
df_steps = pd.DataFrame({
    'timestamp': dates,
    'price': vine_prices,
    'return': vine_returns,
    'volatility': vine_volatilities
})

# Save to CSV
vine_dir = f"{output_dir}/vine"
df_price.to_csv(f"{vine_dir}/prices.csv", index=False)
df_steps.to_csv(f"{vine_dir}/simulation_steps.csv", index=False)

print("Generating CoVaR time series...")
covar_prices, covar_returns, covar_volatilities, covar_system_vol = simulate_covar(
    n_days, start_price=start_price)

# Create price DataFrame
df_price = pd.DataFrame({
    'timestamp': dates,
    'price': covar_prices
})

# Create simulation steps DataFrame
df_steps = pd.DataFrame({
    'timestamp': dates,
    'price': covar_prices,
    'return': covar_returns,
    'volatility': covar_volatilities,
    'system_volatility': covar_system_vol
})

# Save to CSV
covar_dir = f"{output_dir}/covar"
df_price.to_csv(f"{covar_dir}/prices.csv", index=False)
df_steps.to_csv(f"{covar_dir}/simulation_steps.csv", index=False)

print(
    f"All simulations completed. CSV files saved in '{output_dir}' directory.")
