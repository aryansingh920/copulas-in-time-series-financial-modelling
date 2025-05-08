# main.py  – updated 2025‑05‑08
"""Run GARCH‑Vine, DCC‑GARCH and CoVaR risk engines on your dataset"""

from __future__ import annotations
import os, importlib, matplotlib.pyplot as plt, pandas as pd
from copula.DataLoader import load_returns_from_data_folder
from copula.DataAnalyzer import DataAnalyzer

def _flex(path, fallback):
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError:
        return importlib.import_module(fallback)

GARCHVineCopula = _flex('copula.TimeSeries.GARCHVineCopula', 'GARCHVineCopula').GARCHVineCopula
DCCCopula      = _flex('copula.TimeSeries.DCCCopula',      'DCCCopula').DCCCopula
CoVaRCopula    = _flex('copula.TimeSeries.CoVaRCopula',    'CoVaRCopula').CoVaRCopula


def plot_returns(df: pd.DataFrame, out: str = 'returns_timeseries.png') -> None:
    n = df.shape[1]
    plt.figure(figsize=(12, 2.5*n))
    for i, col in enumerate(df.columns, 1):
        plt.subplot(n, 1, i)
        plt.plot(df.index, df[col], lw=.7)
        plt.title(col)
        plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out)
    # plt.show()
    plt.close()


def run_garch_vine(df, alpha=0.05):
    print("\n⚙️  Fitting **GARCH‑Vine Copula** model …")
    res = GARCHVineCopula().fit(df).compute_risk_measures(alpha=alpha)
    print(f"\n===== GARCH‑Vine Risk (α = {alpha}) =====")
    for a,v in res['VaR'].items():
        print(f"VaR[{a}] = {v:.5f},  CVaR = {res['CVaR'][a]:.5f}")
    print(f"Portfolio VaR = {res['Portfolio_VaR']:.5f}")
    print(f"Portfolio CVaR = {res['Portfolio_CVaR']:.5f}")
    return res


def run_dcc(df, alpha=0.05):
    print("\n⚙️  Fitting **DCC‑GARCH Copula** model …")
    res = DCCCopula().fit(df).compute_risk_measures(alpha=alpha)
    print(f"\n===== DCC‑GARCH Risk (α = {alpha}) =====")
    for a,v in res['VaR'].items():
        print(f"VaR[{a}] = {v:.5f},  CVaR = {res['CVaR'][a]:.5f}")
    print(f"High‑Corr VaR = {res['High_Corr_VaR']:.5f}")
    print(f"High‑Corr CVaR = {res['High_Corr_CVaR']:.5f}")
    return res


def run_covar(df, alpha=0.05):
    print("\n⚙️  Fitting **CoVaR Copula** model …")
    res = CoVaRCopula().fit(df).compute_risk_measures(alpha=alpha, conditioning_assets=list(df.columns))
    print(f"\n===== CoVaR Risk (α = {alpha}) =====")
    # Per‑conditioning‑asset detail
    for cond in df.columns:
        covar_dict = res[f'CoVaR_{cond}']
        print(f"\n-- Conditioning on {cond} stress --")
        for tgt, stats in covar_dict.items():
            print(f"{tgt:<25s} VaR={stats['VaR']:.5f}  CoVaR={stats['CoVaR']:.5f}  ΔCoVaR={stats['DeltaCoVaR']:.5f}")
        print(f"Systemic impact (ΣΔCoVaR) = {res[f'Systemic_Impact_{cond}']:.5f}")
    print("\nSystem‑Stress VaR (all conditioning assets stressed):")
    for asset,val in res['System_Stress_VaR'].items():
        print(f"{asset:<25s} {val:.5f}")
    return res


def main(base_dir='data', alpha=0.05):
    print(f"🔍 Loading data from: {base_dir}/ …")
    df = load_returns_from_data_folder(base_dir)
    print("\n===== Sanity Check =====")
    print("Correlation matrix:\n", df.corr().round(6))
    print("\nStandard deviations:\n", df.std())
    print("\n===== DATA SUMMARY =====")
    print(f"Assets: {', '.join(df.columns)}")
    print(df.describe())
    plot_returns(df)

    # quick extra correlation diagnostics
    print("\nPearson correlation matrix:\n", DataAnalyzer(df).compute_correlations()['Pearson'].round(3))

    run_garch_vine(df, alpha)
    run_dcc(df, alpha)
    run_covar(df, alpha)


if __name__ == '__main__':
    main(os.getenv('DATA_DIR', 'data'), float(os.getenv('ALPHA', '0.05')))
