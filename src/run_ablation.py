import os
import sys
import tempfile

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd

from asymmetric_nco import (
    run_nco,
    fit_hmm,
    detect_bull_id,
    apply_confidence_threshold,
    LOOKBACK,
    N_CLUSTERS,
    KAPPA,
    DELTA,
    LAMBDA_TURN_PROPOSED,
    LAMBDA_TURN_BASELINE,
)

BASE_DIR = r"d:\.TESIS\eksperimen\eksprimen 3\paper_sp500_nco"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Load data
df_returns = pd.read_csv(os.path.join(DATA_DIR, "sp500_sectors_monthly_returns.csv"), index_col=0, parse_dates=True)
df_macro = pd.read_csv(os.path.join(DATA_DIR, "sp500_macro_monthly.csv"), index_col=0, parse_dates=True)
df_macro_levels = pd.read_csv(os.path.join(DATA_DIR, "sp500_macro_levels_monthly.csv"), index_col=0, parse_dates=True)

SECTORS = list(df_returns.columns)
N = len(SECTORS)

rf_monthly = (df_macro_levels["^IRX"] / 100.0) / 12.0
rf_monthly = rf_monthly.reindex(df_returns.index).ffill().fillna(0.015 / 12.0)

spy_returns = df_macro["SPY"].reindex(df_returns.index).fillna(0.0)
vix_level = df_macro_levels["^VIX"].reindex(df_returns.index).ffill()

vix_expanding_mean = vix_level.expanding(min_periods=12).mean()
vix_expanding_std = vix_level.expanding(min_periods=12).std().replace(0, 1e-6)
vix_z = ((vix_level - vix_expanding_mean) / vix_expanding_std).fillna(0.0)


def evaluate_strategy(series, rf_series):
    T = len(series)
    cum_wealth = (1.0 + series).cumprod()
    total_ret = cum_wealth.iloc[-1] - 1.0
    cagr = (cum_wealth.iloc[-1]) ** (12.0 / T) - 1.0
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max
    mdd = drawdown.min()
    calmar = cagr / abs(mdd) if abs(mdd) > 0 else 0.0
    return cagr * 100.0, mdd * 100.0, calmar, total_ret * 100.0


def calc_turnover(df_w):
    return df_w.diff().abs().sum(axis=1).fillna(0.0).mean() * 100.0


def run_configuration(config):
    out_of_sample_dates = df_returns.index[LOOKBACK:]
    weights = []
    is_bull_list = []
    last_is_bull = 1
    prev_w = np.ones(N) / N

    for t_idx in range(len(out_of_sample_dates)):
        window_returns = df_returns.iloc[t_idx : t_idx + LOOKBACK]
        curr_rf = rf_monthly.iloc[t_idx + LOOKBACK]

        if config == 'plain':
            is_bull = None
            w = run_nco(window_returns, use_downside=False, objective='min_variance',
                        rf=curr_rf, w_prev=prev_w, lambda_turn=LAMBDA_TURN_BASELINE,
                        n_clusters=N_CLUSTERS)
        elif config == 'baseline_regime':
            X = spy_returns.iloc[: t_idx + LOOKBACK].values.reshape(-1, 1)
            try:
                hmm = fit_hmm(X, sticky_kappa=0.0, covariance_type="full",
                              n_iter=50, tol=1e-2)
                bull_id = detect_bull_id(hmm)
                is_bull = int(hmm.predict(X)[-1] == bull_id)
            except Exception:
                is_bull = 1
            obj = 'max_sharpe' if is_bull else 'min_variance'
            w = run_nco(window_returns, use_downside=False, objective=obj,
                        rf=curr_rf, w_prev=prev_w, lambda_turn=LAMBDA_TURN_BASELINE,
                        n_clusters=N_CLUSTERS)
        elif config == 'asymmetric_static':
            is_bull = None
            w = run_nco(window_returns, use_downside=True, objective='min_variance',
                        rf=curr_rf, w_prev=prev_w, lambda_turn=LAMBDA_TURN_BASELINE,
                        n_clusters=N_CLUSTERS)
        elif config == 'proposed':
            X = np.column_stack([
                spy_returns.iloc[: t_idx + LOOKBACK].values,
                vix_z.iloc[: t_idx + LOOKBACK].values,
            ])
            try:
                hmm = fit_hmm(X, sticky_kappa=KAPPA, covariance_type="diag",
                              n_iter=60, tol=1e-2)
                bull_id = detect_bull_id(hmm)
                p_bull = hmm.predict_proba(X)[-1][bull_id]
                is_bull = apply_confidence_threshold(last_is_bull, p_bull, delta=DELTA)
            except Exception:
                is_bull = last_is_bull
            obj = 'max_sortino' if is_bull else 'min_variance'
            w = run_nco(window_returns, use_downside=True, objective=obj,
                        rf=curr_rf, w_prev=prev_w, lambda_turn=LAMBDA_TURN_PROPOSED,
                        n_clusters=N_CLUSTERS)
        else:
            raise ValueError(f"Unknown configuration: {config}")

        weights.append(w)
        if is_bull is not None:
            is_bull_list.append(is_bull)
            last_is_bull = is_bull
        prev_w = w

    df_w = pd.DataFrame(weights, index=out_of_sample_dates, columns=SECTORS)
    returns = (df_w * df_returns.loc[out_of_sample_dates]).sum(axis=1)
    oos_rf = rf_monthly.loc[out_of_sample_dates]
    cagr, mdd, calmar, total_ret = evaluate_strategy(returns, oos_rf)
    turnover = calc_turnover(df_w)
    switches = int(np.sum(np.diff(is_bull_list) != 0)) if is_bull_list else 0

    return {
        "CAGR (%)": cagr,
        "Max Drawdown (%)": mdd,
        "Calmar Ratio": calmar,
        "Regime Switches": switches,
        "Monthly Turnover (%)": turnover,
    }, df_w, returns


# Component Ablation Table
ablation_rows = []
weights_store = {}
returns_store = {}

for config, name, risk_metric, regime_dynamic in [
    ('plain', 'Plain NCO (Symmetric, No Regime)', 'Symmetric Covariance', 'Static (None)'),
    ('baseline_regime', 'Baseline Regime-NCO (Symmetric, Naive 1D-HMM)', 'Symmetric Covariance', 'Naive 1D-HMM (Unregularized)'),
    ('asymmetric_static', 'Asymmetric NCO (Downside, No Regime)', 'Downside Semi-Covariance', 'Static (None)'),
    ('proposed', 'Proposed Asymmetric Sticky-NCO (Downside, Sticky 2D-HMM)', 'Downside Semi-Covariance', 'Sticky 2D-HMM (Prior + Confidence Thresh)'),
]:
    metrics, df_w, returns = run_configuration(config)
    weights_store[config] = df_w
    returns_store[config] = returns
    ablation_rows.append({
        "Configuration": name,
        "Risk Metric": risk_metric,
        "Regime Dynamic": regime_dynamic,
        **metrics,
    })
    print(f"[ABLATION] {name}: "
          f"CAGR={metrics['CAGR (%)']:.2f}% MaxDD={metrics['Max Drawdown (%)']:.2f}% "
          f"Calmar={metrics['Calmar Ratio']:.2f} Switches={metrics['Regime Switches']} "
          f"Turnover={metrics['Monthly Turnover (%)']:.2f}%/mo")

df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv(os.path.join(RESULTS_DIR, "ablation_components.csv"), index=False)

# Transaction cost sensitivity (net CAGR under slippage)
df_all_returns = pd.read_csv(os.path.join(RESULTS_DIR, "strategy_monthly_returns.csv"), index_col=0, parse_dates=True)
df_w_prop = pd.read_csv(os.path.join(RESULTS_DIR, "weights_proposed_asym_nco.csv"), index_col=0, parse_dates=True)
df_w_base = pd.read_csv(os.path.join(RESULTS_DIR, "weights_baseline_regime_nco.csv"), index_col=0, parse_dates=True)
df_w_plain = pd.read_csv(os.path.join(RESULTS_DIR, "weights_plain_nco.csv"), index_col=0, parse_dates=True)

cost_bps = [0, 5, 10, 15, 20]
cost_results = {}

for bps in cost_bps:
    cost_rate = bps / 10000.0
    turnover_prop = df_w_prop.diff().abs().sum(axis=1).fillna(0.0)
    net_prop = df_all_returns["Proposed Asymmetric Sticky-NCO"] - cost_rate * turnover_prop
    cagr_prop = ((1.0 + net_prop).prod()) ** (12.0 / len(net_prop)) - 1.0

    turnover_base = df_w_base.diff().abs().sum(axis=1).fillna(0.0)
    net_base = df_all_returns["Baseline Regime-NCO"] - cost_rate * turnover_base
    cagr_base = ((1.0 + net_base).prod()) ** (12.0 / len(net_base)) - 1.0

    turnover_plain = df_w_plain.diff().abs().sum(axis=1).fillna(0.0)
    net_plain = df_all_returns["Plain NCO"] - cost_rate * turnover_plain
    cagr_plain = ((1.0 + net_plain).prod()) ** (12.0 / len(net_plain)) - 1.0

    cost_results[f"{bps} bps"] = {
        "Plain NCO Net CAGR (%)": cagr_plain * 100.0,
        "Baseline Regime-NCO Net CAGR (%)": cagr_base * 100.0,
        "Proposed Asym-Sticky Net CAGR (%)": cagr_prop * 100.0,
        "Alpha Spread over Plain (bps)": (cagr_prop - cagr_plain) * 10000.0,
        "Alpha Spread over Base (bps)": (cagr_prop - cagr_base) * 10000.0,
    }

df_costs = pd.DataFrame(cost_results).T
print("\n" + "=" * 80)
print("     TRANSACTION COST SENSITIVITY AUDIT (NET CAGR UNDER SLIPPAGE)")
print("=" * 80)
print(df_costs.to_string(float_format="%.2f"))
print("=" * 80)

df_costs.to_csv(os.path.join(RESULTS_DIR, "transaction_cost_sensitivity.csv"))

print("\n[SUCCESS] Ablation and transaction cost analysis completed.")
