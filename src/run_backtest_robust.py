import os
import sys
import tempfile

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from asymmetric_nco import (
    compute_downside_semicov,
    optimize_portfolio,
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
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Load Data
df_returns = pd.read_csv(os.path.join(DATA_DIR, "sp500_sectors_monthly_returns.csv"), index_col=0, parse_dates=True)
df_macro = pd.read_csv(os.path.join(DATA_DIR, "sp500_macro_monthly.csv"), index_col=0, parse_dates=True)
df_macro_levels = pd.read_csv(os.path.join(DATA_DIR, "sp500_macro_levels_monthly.csv"), index_col=0, parse_dates=True)

SECTORS = list(df_returns.columns)
N = len(SECTORS)

# Risk-free rate (monthly)
rf_monthly = (df_macro_levels["^IRX"] / 100.0) / 12.0
rf_monthly = rf_monthly.reindex(df_returns.index).ffill().fillna(0.015 / 12.0)

spy_returns = df_macro["SPY"].reindex(df_returns.index).fillna(0.0)
vix_level = df_macro_levels["^VIX"].reindex(df_returns.index).ffill()

# Normalize VIX using expanding z-score to avoid lookahead bias
vix_expanding_mean = vix_level.expanding(min_periods=12).mean()
vix_expanding_std = vix_level.expanding(min_periods=12).std().replace(0, 1e-6)
vix_z = ((vix_level - vix_expanding_mean) / vix_expanding_std).fillna(0.0)


# --- Downside Semi-Covariance Matrix ---
# See src/asymmetric_nco.py for all core implementations.

# --- Walk-Forward Backtesting Engine ---
out_of_sample_dates = df_returns.index[LOOKBACK:]
num_periods = len(out_of_sample_dates)

weights_eq = []
weights_mv = []
weights_plain_nco = []
weights_base_regime_nco = []
weights_prop_asym_nco = []

base_is_bull_list = []
prop_is_bull_list = []
prop_bull_probs = []

prev_w_mv = np.ones(N) / N
prev_w_plain = np.ones(N) / N
prev_w_base = np.ones(N) / N
prev_w_prop = np.ones(N) / N

last_prop_is_bull = 1

for t_idx, current_date in enumerate(out_of_sample_dates):
    # Rolling 24-month window for asset covariance
    window_returns = df_returns.iloc[t_idx : t_idx + LOOKBACK]
    curr_rf = rf_monthly.iloc[t_idx + LOOKBACK]
    
    # Expanding window up to t_idx + LOOKBACK for HMM estimation (Zero lookahead)
    hist_spy = spy_returns.iloc[: t_idx + LOOKBACK]
    hist_vix = vix_z.iloc[: t_idx + LOOKBACK]
    
    # 1. Equal Weight
    weights_eq.append(np.ones(N) / N)
    
    # 2. Min-Variance (MVO)
    cov_sym = np.cov(window_returns.values, rowvar=False)
    mu_sym = window_returns.mean().values
    w_mv = optimize_portfolio(mu_sym, cov_sym, obj_type='min_variance', 
                              w_prev=None, lambda_turn=0.0)
    weights_mv.append(w_mv)
    prev_w_mv = w_mv
    
    # 3. Plain NCO (Symmetric)
    w_plain = run_nco(window_returns, use_downside=False, objective='min_variance',
                      rf=curr_rf, w_prev=prev_w_plain, lambda_turn=LAMBDA_TURN_BASELINE)
    weights_plain_nco.append(w_plain)
    prev_w_plain = w_plain
    
    # 4. Baseline Regime-NCO (Paper 1 formulation: 1D Gaussian HMM without sticky prior)
    X_base = hist_spy.values.reshape(-1, 1)
    try:
        hmm_base = fit_hmm(X_base, sticky_kappa=0.0, covariance_type="full",
                           n_iter=50, tol=1e-2)
        bull_id_base = detect_bull_id(hmm_base)
        states_base = hmm_base.predict(X_base)
        base_is_bull = int(states_base[-1] == bull_id_base)
    except Exception:
        base_is_bull = 1
        
    base_is_bull_list.append(base_is_bull)
    obj_base = 'max_sharpe' if base_is_bull else 'min_variance'
    w_base = run_nco(window_returns, use_downside=False, objective=obj_base,
                     rf=curr_rf, w_prev=prev_w_base, lambda_turn=LAMBDA_TURN_BASELINE)
    weights_base_regime_nco.append(w_base)
    prev_w_base = w_base
    
    # 5. Proposed: Asymmetric Sticky-NCO (2D multi-factor HMM with sticky prior and confidence threshold)
    X_prop = np.column_stack([hist_spy.values, hist_vix.values])
    try:
        hmm_prop = fit_hmm(X_prop, sticky_kappa=KAPPA, covariance_type="diag",
                           n_iter=60, tol=1e-2)
        bull_id_prop = detect_bull_id(hmm_prop)
        
        post_prop = hmm_prop.predict_proba(X_prop)[-1]
        p_bull = post_prop[bull_id_prop]
        
        # Confidence-adaptive threshold: delta = 0.65
        prop_is_bull = apply_confidence_threshold(last_prop_is_bull, p_bull, delta=DELTA)
    except Exception:
        prop_is_bull = last_prop_is_bull
        p_bull = 0.5
        
    prop_is_bull_list.append(prop_is_bull)
    prop_bull_probs.append(p_bull)
    last_prop_is_bull = prop_is_bull
    
    # Proposed Asymmetric allocation:
    # Bull -> Max Sortino on Downside Semi-Covariance
    # Bear -> Min Downside Variance on Downside Semi-Covariance
    obj_prop = 'max_sortino' if prop_is_bull else 'min_variance'
    w_prop = run_nco(window_returns, use_downside=True, objective=obj_prop,
                     rf=curr_rf, w_prev=prev_w_prop, lambda_turn=LAMBDA_TURN_PROPOSED)
    weights_prop_asym_nco.append(w_prop)
    prev_w_prop = w_prop

# Build DataFrames
df_w_eq = pd.DataFrame(weights_eq, index=out_of_sample_dates, columns=SECTORS)
df_w_mv = pd.DataFrame(weights_mv, index=out_of_sample_dates, columns=SECTORS)
df_w_plain = pd.DataFrame(weights_plain_nco, index=out_of_sample_dates, columns=SECTORS)
df_w_base = pd.DataFrame(weights_base_regime_nco, index=out_of_sample_dates, columns=SECTORS)
df_w_prop = pd.DataFrame(weights_prop_asym_nco, index=out_of_sample_dates, columns=SECTORS)

oos_sector_returns = df_returns.loc[out_of_sample_dates]
oos_rf = rf_monthly.loc[out_of_sample_dates]
oos_spy = spy_returns.loc[out_of_sample_dates]

ret_spy = oos_spy
ret_eq = (df_w_eq * oos_sector_returns).sum(axis=1)
ret_mv = (df_w_mv * oos_sector_returns).sum(axis=1)
ret_plain = (df_w_plain * oos_sector_returns).sum(axis=1)
ret_base = (df_w_base * oos_sector_returns).sum(axis=1)
ret_prop = (df_w_prop * oos_sector_returns).sum(axis=1)

df_all_returns = pd.DataFrame({
    "S&P 500 (SPY)": ret_spy,
    "Equal-Weight (1/11)": ret_eq,
    "Min-Variance (MVO)": ret_mv,
    "Plain NCO": ret_plain,
    "Baseline Regime-NCO": ret_base,
    "Proposed Asymmetric Sticky-NCO": ret_prop
})

df_all_returns.to_csv(os.path.join(RESULTS_DIR, "strategy_monthly_returns.csv"))

def evaluate_strategy(series, rf_series):
    T = len(series)
    cum_wealth = (1.0 + series).cumprod()
    total_ret = cum_wealth.iloc[-1] - 1.0
    cagr = (cum_wealth.iloc[-1]) ** (12.0 / T) - 1.0
    ann_vol = series.std() * np.sqrt(12.0)
    
    neg_dev = np.minimum(series, 0.0)
    downside_vol = np.sqrt(np.mean(neg_dev**2)) * np.sqrt(12.0)
    
    excess_ret = series - rf_series
    ann_excess = excess_ret.mean() * 12.0
    sharpe = ann_excess / (ann_vol + 1e-8)
    sortino = ann_excess / (downside_vol + 1e-8)
    
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max
    mdd = drawdown.min()
    calmar = cagr / abs(mdd) if abs(mdd) > 0 else 0.0
    
    return {
        "CAGR (%)": cagr * 100.0,
        "Annual Vol (%)": ann_vol * 100.0,
        "Downside Vol (%)": downside_vol * 100.0,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": mdd * 100.0,
        "Calmar Ratio": calmar,
        "Total Return (%)": total_ret * 100.0
    }

metrics = {}
for col in df_all_returns.columns:
    metrics[col] = evaluate_strategy(df_all_returns[col], oos_rf)

df_metrics = pd.DataFrame(metrics).T

def calc_turnover(df_w):
    return df_w.diff().abs().sum(axis=1).mean() * 100.0

df_metrics["Monthly Turnover (%)"] = [
    0.0,
    calc_turnover(df_w_eq),
    calc_turnover(df_w_mv),
    calc_turnover(df_w_plain),
    calc_turnover(df_w_base),
    calc_turnover(df_w_prop)
]

base_switches = int(np.sum(np.diff(base_is_bull_list) != 0))
prop_switches = int(np.sum(np.diff(prop_is_bull_list) != 0))

print("\n" + "="*80)
print("              REFINED WALK-FORWARD RESULTS (S&P 500 SECTORS)")
print(f"       Period: {out_of_sample_dates[0].strftime('%Y-%m')} to {out_of_sample_dates[-1].strftime('%Y-%m')} ({num_periods} Months)")
print("="*80)
print(df_metrics.to_string(float_format="%.2f"))
print("="*80)
print(f"[REGIME STABILITY AUDIT]")
print(f"  Baseline 1D-HMM Switches: {base_switches} switches out of {num_periods} months ({base_switches/num_periods*100:.1f}%)")
print(f"  Sticky 2D-HMM Switches  : {prop_switches} switches out of {num_periods} months ({prop_switches/num_periods*100:.1f}%)")
print(f"  Switching Reduction     : {((base_switches - prop_switches)/max(base_switches,1))*100:.1f}% reduction in whipsaw churning!")
print("="*80)

df_metrics.to_csv(os.path.join(RESULTS_DIR, "performance_metrics.csv"))

# Save weight allocations
df_w_prop.to_csv(os.path.join(RESULTS_DIR, "weights_proposed_asym_nco.csv"))
df_w_base.to_csv(os.path.join(RESULTS_DIR, "weights_baseline_regime_nco.csv"))
df_w_plain.to_csv(os.path.join(RESULTS_DIR, "weights_plain_nco.csv"))

# Save regime state timelines
pd.DataFrame({
    "Baseline_Is_Bull": base_is_bull_list,
    "Sticky_Is_Bull": prop_is_bull_list,
    "Sticky_Bull_Prob": prop_bull_probs
}, index=out_of_sample_dates).to_csv(os.path.join(RESULTS_DIR, "regime_timeline.csv"))

print("\n[SUCCESS] Refined empirical backtest completed successfully.")
