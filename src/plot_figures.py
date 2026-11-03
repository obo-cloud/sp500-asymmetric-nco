import os
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

BASE_DIR = r"d:\.TESIS\eksperimen\eksprimen 3\paper_sp500_nco"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'lines.linewidth': 1.6,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

COLORS = {
    "S&P 500 (SPY)": "#4a4a4a",                 # Dark Charcoal
    "Equal-Weight (1/11)": "#8c564b",           # Brown
    "Min-Variance (MVO)": "#1f77b4",            # Blue
    "Plain NCO": "#ff7f0e",                     # Orange
    "Baseline Regime-NCO": "#d62728",           # Crimson
    "Proposed Asymmetric Sticky-NCO": "#1b9e77" # Vibrant Forest Green
}

def plot_fig1_cumulative_wealth(df_returns):
    cum_wealth = (1.0 + df_returns).cumprod()
    
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=300)
    for col in cum_wealth.columns:
        is_prop = ("Proposed" in col)
        is_spy = ("SPY" in col)
        ax.plot(cum_wealth.index, cum_wealth[col], label=col, 
                color=COLORS.get(col, "#333333"),
                linewidth=2.4 if is_prop else (1.8 if is_spy else 1.2),
                linestyle='-' if (is_prop or is_spy) else '--')
        
    ax.set_ylabel("Portfolio Value (Base = 1.0)")
    ax.set_xlabel("Year")
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_cumulative_wealth.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_cumulative_wealth.png"), bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Saved Fig 1: Cumulative Wealth")

def plot_fig2_drawdown(df_returns):
    cum_wealth = (1.0 + df_returns).cumprod()
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max * 100.0
    
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=300)
    for col in ["S&P 500 (SPY)", "Plain NCO", "Baseline Regime-NCO", "Proposed Asymmetric Sticky-NCO"]:
        if col in drawdown.columns:
            is_prop = ("Proposed" in col)
            ax.plot(drawdown.index, drawdown[col], label=col,
                    color=COLORS.get(col, "#333333"),
                    linewidth=2.2 if is_prop else 1.3,
                    linestyle='-' if is_prop else '--')
                    
    ax.fill_between(drawdown.index, drawdown["Proposed Asymmetric Sticky-NCO"], 0, 
                    color='#1b9e77', alpha=0.18, label="Proposed Drawdown Envelope")
    
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Year")
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92)
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_drawdown.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_drawdown.png"), bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Saved Fig 2: Drawdown")

def plot_fig3_regime_timeline(df_regimes):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 4.2), dpi=300, sharex=True)
    
    # Baseline HMM
    ax1.step(df_regimes.index, df_regimes["Baseline_Is_Bull"], where='post', color='#d62728', linewidth=1.4)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Bear", "Bull"])
    ax1.set_ylabel("Baseline State")
    ax1.set_title("(a) Baseline 1D-HMM (Frequent Whipsaw Regime Switches)")
    ax1.grid(True)
    
    # Sticky HMM
    ax2.step(df_regimes.index, df_regimes["Sticky_Is_Bull"], where='post', color='#1b9e77', linewidth=1.8, label="Filtered State")
    if "Sticky_Bull_Prob" in df_regimes.columns:
        ax2.plot(df_regimes.index, df_regimes["Sticky_Bull_Prob"], color='#7570b3', linestyle=':', alpha=0.75, label="Bull Posterior P(S=Bull)")
        ax2.axhline(0.65, color='grey', linestyle='--', alpha=0.5, label="Threshold (0.65)")
        ax2.axhline(0.35, color='grey', linestyle='--', alpha=0.5)
        
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(["Bear", "0.5", "Bull"])
    ax2.set_ylabel("Sticky State / Prob")
    ax2.set_xlabel("Year")
    ax2.set_title("(b) Sticky 2D-HMM with Confidence-Adaptive Threshold (Persistent Regimes)")
    ax2.grid(True)
    ax2.legend(loc="upper left", frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_regime_comparison.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_regime_comparison.png"), bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Saved Fig 3: Regime Timeline")

def plot_fig4_dendrograms(sector_returns):
    R = sector_returns.values
    T, N = R.shape
    
    # 1. Symmetric
    corr_sym = np.corrcoef(R, rowvar=False)
    dist_sym = np.sqrt(0.5 * np.maximum(1.0 - corr_sym, 0.0))
    np.fill_diagonal(dist_sym, 0.0)
    Z_sym = linkage(squareform(dist_sym, checks=False), method='ward')
    
    # 2. Downside
    D = np.minimum(R, 0.0)
    semicov = np.dot(D.T, D) / T
    diag = np.sqrt(np.diag(semicov))
    outer = np.outer(diag, diag)
    outer = np.where(outer == 0, 1e-12, outer)
    corr_down = np.clip(semicov / outer, -1.0, 1.0)
    dist_down = np.sqrt(0.5 * np.maximum(1.0 - corr_down, 0.0))
    np.fill_diagonal(dist_down, 0.0)
    Z_down = linkage(squareform(dist_down, checks=False), method='ward')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), dpi=300)
    
    dendrogram(Z_sym, labels=list(sector_returns.columns), ax=ax1, leaf_rotation=45)
    ax1.set_title("(a) Symmetric Covariance Distance Tree")
    ax1.set_ylabel("Ward Linkage Distance")
    ax1.grid(True, axis='y')
    
    dendrogram(Z_down, labels=list(sector_returns.columns), ax=ax2, leaf_rotation=45)
    ax2.set_title("(b) Downside Semi-Covariance Distance Tree")
    ax2.set_ylabel("Ward Linkage Distance")
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_dendrogram_comparison.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_dendrogram_comparison.png"), bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Saved Fig 4: Dendrograms")

def plot_fig5_weights_allocation(df_weights):
    fig, ax = plt.subplots(figsize=(9, 4.3), dpi=300)
    
    sorted_cols = df_weights.mean().sort_values(ascending=False).index
    df_plot = df_weights[sorted_cols] * 100.0
    
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(sorted_cols)-1, 1)) for i in range(len(sorted_cols))]
    ax.stackplot(df_plot.index, [df_plot[c] for c in sorted_cols], 
                 labels=sorted_cols, alpha=0.88, colors=colors)
                 
    ax.set_ylabel("Sector Allocation Weight (%)")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y')
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8)
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_sector_allocation.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_sector_allocation.png"), bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Saved Fig 5: Sector Allocation")

def plot_fig6_metrics_bars(df_metrics):
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    
    x = np.arange(len(df_metrics))
    width = 0.26
    
    cagr = df_metrics["CAGR (%)"].values
    mdd = np.abs(df_metrics["Max Drawdown (%)"].values)
    calmar = df_metrics["Calmar Ratio"].values * 10.0 # Scale by 10 for visual balance
    
    ax.bar(x - width, cagr, width, label="CAGR (%)", color="#1f77b4", alpha=0.85)
    ax.bar(x, mdd, width, label="|Max Drawdown| (%)", color="#d62728", alpha=0.85)
    ax.bar(x + width, calmar, width, label="Calmar Ratio (x10)", color="#1b9e77", alpha=0.85)
    
    ax.set_xticks(x)
    # Wrap strategy labels
    labels = [c.replace(" (SPY)", "\n(SPY)").replace(" (1/11)", "\n(1/11)").replace(" (MVO)", "\n(MVO)").replace("Proposed ", "Proposed\n") for c in df_metrics.index]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Metric Value")
    ax.grid(True, axis='y')
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_performance_comparison.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_performance_comparison.png"), bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Saved Fig 6: Performance Metrics Bar Chart")


if __name__ == "__main__":
    returns_csv = os.path.join(RESULTS_DIR, "strategy_monthly_returns.csv")
    weights_csv = os.path.join(RESULTS_DIR, "weights_proposed_asym_nco.csv")
    sector_csv = os.path.join(DATA_DIR, "sp500_sectors_monthly_returns.csv")
    regime_csv = os.path.join(RESULTS_DIR, "regime_timeline.csv")
    metrics_csv = os.path.join(RESULTS_DIR, "performance_metrics.csv")
    
    if os.path.exists(returns_csv):
        df_ret = pd.read_csv(returns_csv, index_col=0, parse_dates=True)
        plot_fig1_cumulative_wealth(df_ret)
        plot_fig2_drawdown(df_ret)
        
    if os.path.exists(regime_csv):
        df_reg = pd.read_csv(regime_csv, index_col=0, parse_dates=True)
        plot_fig3_regime_timeline(df_reg)
        
    if os.path.exists(sector_csv):
        df_sec = pd.read_csv(sector_csv, index_col=0, parse_dates=True)
        plot_fig4_dendrograms(df_sec)
        
    if os.path.exists(weights_csv):
        df_w = pd.read_csv(weights_csv, index_col=0, parse_dates=True)
        plot_fig5_weights_allocation(df_w)
        
    if os.path.exists(metrics_csv):
        df_met = pd.read_csv(metrics_csv, index_col=0)
        plot_fig6_metrics_bars(df_met)
