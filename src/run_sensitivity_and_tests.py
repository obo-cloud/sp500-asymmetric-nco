import os
import sys
import tempfile

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb

from asymmetric_nco import (
    run_nco,
    fit_hmm,
    detect_bull_id,
    apply_confidence_threshold,
    LOOKBACK,
    N_CLUSTERS,
    LAMBDA_TURN_PROPOSED,
    LAMBDA_TURN_BASELINE,
)

BASE_DIR = r"d:\.TESIS\eksperimen\eksprimen 3\paper_sp500_nco"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

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

out_of_sample_dates = df_returns.index[LOOKBACK:]
oos_rf = rf_monthly.loc[out_of_sample_dates]


def evaluate_strategy(series):
    T = len(series)
    cum_wealth = (1.0 + series).cumprod()
    cagr = (cum_wealth.iloc[-1]) ** (12.0 / T) - 1.0
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max
    mdd = drawdown.min()
    calmar = cagr / abs(mdd) if abs(mdd) > 0 else 0.0
    ann_vol = series.std() * np.sqrt(12.0)
    excess = series - oos_rf
    sharpe = excess.mean() * 12.0 / (ann_vol + 1e-8)
    return cagr * 100.0, mdd * 100.0, calmar, sharpe


def calc_turnover(df_w):
    return df_w.diff().abs().sum(axis=1).fillna(0.0).mean() * 100.0


def run_proposed(kappa, delta):
    weights = []
    is_bull_list = []
    last_is_bull = 1
    prev_w = np.ones(N) / N

    for t_idx in range(len(out_of_sample_dates)):
        window_returns = df_returns.iloc[t_idx : t_idx + LOOKBACK]
        curr_rf = rf_monthly.iloc[t_idx + LOOKBACK]
        X = np.column_stack([
            spy_returns.iloc[: t_idx + LOOKBACK].values,
            vix_z.iloc[: t_idx + LOOKBACK].values,
        ])
        try:
            hmm = fit_hmm(X, sticky_kappa=kappa, covariance_type="diag",
                          n_iter=60, tol=1e-2)
            bull_id = detect_bull_id(hmm)
            p_bull = hmm.predict_proba(X)[-1][bull_id]
            is_bull = apply_confidence_threshold(last_is_bull, p_bull, delta=delta)
        except Exception:
            is_bull = last_is_bull
        obj = 'max_sortino' if is_bull else 'min_variance'
        w = run_nco(window_returns, use_downside=True, objective=obj,
                    rf=curr_rf, w_prev=prev_w, lambda_turn=LAMBDA_TURN_PROPOSED,
                    n_clusters=N_CLUSTERS)
        weights.append(w)
        is_bull_list.append(is_bull)
        last_is_bull = is_bull
        prev_w = w

    df_w = pd.DataFrame(weights, index=out_of_sample_dates, columns=SECTORS)
    returns = (df_w * df_returns.loc[out_of_sample_dates]).sum(axis=1)
    cagr, mdd, calmar, sharpe = evaluate_strategy(returns)
    return {
        "kappa": kappa,
        "delta": delta,
        "CAGR (%)": cagr,
        "Max DD (%)": mdd,
        "Calmar": calmar,
        "Sharpe": sharpe,
        "Switches": int(np.sum(np.diff(is_bull_list) != 0)),
        "Turnover (%/mo)": calc_turnover(df_w),
    }, returns


# ----------------------------------------------------------------------
# 1. Hyperparameter sensitivity grid (kappa x delta)
# ----------------------------------------------------------------------
KAPPA_GRID = [0.2, 0.3, 0.4]
DELTA_GRID = [0.6, 0.65, 0.7]

grid_rows = []
grid_returns = {}
for kappa, delta in itertools.product(KAPPA_GRID, DELTA_GRID):
    metrics, returns = run_proposed(kappa, delta)
    grid_rows.append(metrics)
    grid_returns[(kappa, delta)] = returns
    print(f"[GRID] kappa={kappa} delta={delta}: CAGR={metrics['CAGR (%)']:.2f}% "
          f"MaxDD={metrics['Max DD (%)']:.2f}% Calmar={metrics['Calmar']:.2f} "
          f"Sharpe={metrics['Sharpe']:.2f} Switches={metrics['Switches']} "
          f"Turnover={metrics['Turnover (%/mo)']:.2f}%/mo")

df_grid = pd.DataFrame(grid_rows)
df_grid.to_csv(os.path.join(RESULTS_DIR, "sensitivity_grid.csv"), index=False)

# ----------------------------------------------------------------------
# 2. Statistical inference: PSR, DSR (K trials), PBO (CSCV)
# ----------------------------------------------------------------------
def psr(returns_excess, sr_star=0.0):
    n = len(returns_excess)
    sr_hat = returns_excess.mean() / (returns_excess.std() + 1e-12) * np.sqrt(12.0)
    skew = stats.skew(returns_excess)
    kurt = stats.kurtosis(returns_excess, fisher=False)
    denom = np.sqrt(1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2)
    z = (sr_hat - sr_star) * np.sqrt(n - 1.0) / denom
    return stats.norm.cdf(z), sr_hat, skew, kurt


def deflated_sr_star(returns_excess, n_trials):
    n = len(returns_excess)
    sr_hat = returns_excess.mean() / (returns_excess.std() + 1e-12) * np.sqrt(12.0)
    skew = stats.skew(returns_excess)
    kurt = stats.kurtosis(returns_excess, fisher=False)
    var_sr = (1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2) / (n - 1.0)
    euler = np.euler_gamma
    sr0 = np.sqrt(var_sr) * (
        (1.0 - euler) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + euler * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    )
    return sr0


df_all_returns = pd.read_csv(os.path.join(RESULTS_DIR, "strategy_monthly_returns.csv"),
                             index_col=0, parse_dates=True)

N_TRIALS = 15

stat_rows = []
for col in df_all_returns.columns:
    excess = (df_all_returns[col] - oos_rf).values
    psr_val, sr_hat, skew, kurt = psr(excess, sr_star=0.0)
    sr0 = deflated_sr_star(excess, N_TRIALS)
    dsr_val, _, _, _ = psr(excess, sr_star=sr0)
    stat_rows.append({
        "Strategy": col,
        "Sharpe": sr_hat,
        "Skew": skew,
        "Kurt": kurt,
        "PSR(SR*=0)": psr_val,
        "DSR(K=15)": dsr_val,
    })
    print(f"[STAT] {col}: SR={sr_hat:.3f} PSR={psr_val:.4f} DSR={dsr_val:.4f}")

df_stats = pd.DataFrame(stat_rows)

# PBO via Combinatorial Symmetric Cross-Validation over the full trial set
# Trials: 9 sensitivity configs + Plain NCO + Baseline Regime-NCO + Asymmetric NCO
trial_cols = [f"kappa={k},delta={d}" for k, d in itertools.product(KAPPA_GRID, DELTA_GRID)]
trial_returns = pd.DataFrame(index=out_of_sample_dates)
for (kappa, delta), name in zip(grid_returns.keys(), trial_cols):
    trial_returns[name] = grid_returns[(kappa, delta)].values

for name, csv in [
    ("Plain NCO", "weights_plain_nco.csv"),
    ("Baseline Regime-NCO", "weights_baseline_regime_nco.csv"),
]:
    df_w = pd.read_csv(os.path.join(RESULTS_DIR, csv), index_col=0, parse_dates=True)
    trial_returns[name] = (df_w * df_returns.loc[out_of_sample_dates]).sum(axis=1).values

M = trial_returns.values.astype(float)
T, N_trials = M.shape

S = 16
usable = (T // S) * S
M_usable = M[T - usable:]
groups = np.arange(S).repeat(usable // S)

split_indices = np.arange(S)
n_combos = int(comb(S, S // 2))
combos = np.array(list(itertools.combinations(split_indices, S // 2)))

is_mask = np.zeros((n_combos, usable), dtype=bool)
for c in range(n_combos):
    sel = combos[c]
    in_group = np.isin(groups, sel)
    is_mask[c] = in_group

MIS = np.zeros((n_combos, N_trials))
MOOS = np.zeros((n_combos, N_trials))
for c in range(n_combos):
    MIS[c] = M_usable[is_mask[c]].mean(axis=0)
    MOOS[c] = M_usable[~is_mask[c]].mean(axis=0)

n_star = np.argmax(MIS, axis=1)
o_ranks = np.argsort(np.argsort(MOOS, axis=1), axis=1) + 1
omega = np.array([o_ranks[c, n_star[c]] / N_trials for c in range(n_combos)])
omega = np.clip(omega, 1e-6, 1.0 - 1e-6)
lam = np.log(omega / (1.0 - omega))
pbo = float(stats.norm.cdf(-lam.mean() / (lam.std() + 1e-12)))
print(f"[STAT] CSCV: S={S}, trials={N_trials}, combos={n_combos}, PBO={pbo:.4f}")
print(f"[STAT] relative rank mean={omega.mean():.3f}, median={np.median(omega):.3f}")

df_stats.to_csv(os.path.join(RESULTS_DIR, "statistical_tests.csv"), index=False)
pd.DataFrame({
    "Metric": ["PBO", "Trial count", "S", "Combos", "Mean rel. rank", "Median rel. rank"],
    "Value": [pbo, N_trials, S, n_combos, omega.mean(), np.median(omega)],
}).to_csv(os.path.join(RESULTS_DIR, "pbo_cscv.csv"), index=False)

print("\n[SUCCESS] Sensitivity grid and statistical inference completed.")
