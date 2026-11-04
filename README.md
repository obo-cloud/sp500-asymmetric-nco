# Asymmetric Downside-Risk Nested Clustered Optimization (Asym-Sticky-NCO)

[![arXiv](https://img.shields.io/badge/arXiv-2611.05310-b31b1b.svg)](https://arxiv.org/abs/2611.05310)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Code and data repository for the paper (arXiv:2611.05310):

**"Mitigating Return Drag and Regime Whipsaws in Sector Rotation: An Asymmetric Downside-Aware Nested Clustered Optimization Framework"**

Author: Supriadi (COCONUT Computer Club, Makassar, Indonesia — supriadi@coconut.or.id)

---

## Abstract

Hierarchical clustering and Nested Clustered Optimization (NCO) provide an effective paradigm for mitigating Markowitz covariance matrix ill-conditioning. However, two structural vulnerabilities persist when deployed in modern equity markets:

1. **Return Drag:** Symmetric variance penalizes upside and downside return dispersion identically, heavily penalizing high-growth momentum sectors (such as Information Technology).
2. **Whipsaw Instability:** Unregularized Hidden Markov Models (HMM) suffer from hyper-active regime switching (whipsawing), incurring high turnover and transaction decay.

This repository implements **Asymmetric Downside-Risk Nested Clustered Optimization with Sticky Regime Dynamics (Asym-Sticky-NCO)** for tactical asset allocation across all 11 S&P 500 GICS sectors:

- **Bawa-Lindenberg Downside Semi-Covariance Matrix** embedded directly into the tree distance metric and the two-stage allocation.
- **Sticky Multi-Factor Gaussian HMM** with diagonal persistence prior (κ = 0.30) and confidence-adaptive thresholds (δ = 0.65) on S&P 500 returns and CBOE VIX levels.
- **Adaptive Dual-Regime Sortino / Minimum-Downside-Variance Allocation** with turnover regularization.

---

## Empirical Results (2020–2025 Out-of-Sample)

| Strategy | CAGR (%) | Ann. Vol (%) | Downside Vol (%) | Sharpe Ratio | Sortino Ratio | Max DD (%) | Calmar Ratio | Total Ret (%) | Turnover (%/mo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| S&P 500 (SPY) | 16.36 | 15.58 | 9.14 | 0.86 | 1.47 | -23.93 | 0.68 | 127.26 | 0.00 |
| Equal-Weight (1/11) | 14.28 | 14.78 | 8.53 | 0.78 | 1.35 | -18.32 | 0.78 | 106.09 | 0.00 |
| Min-Variance (MVO) | 10.56 | 13.09 | 7.75 | 0.61 | 1.02 | -13.52 | 0.78 | 72.25 | 17.13 |
| Plain NCO (Symmetric) | 10.65 | 14.03 | 8.40 | 0.58 | 0.97 | -14.48 | 0.74 | 72.99 | 26.99 |
| Baseline Regime-NCO | 11.05 | 17.70 | 11.58 | 0.52 | 0.79 | -21.81 | 0.51 | 76.46 | 60.78 |
| **Proposed Asym-Sticky-NCO** | **13.41** | **17.71** | **10.51** | **0.63** | **1.07** | **-14.22** | **0.94** | **97.74** | **30.48** |

Key results:

- **+276 bps/year alpha over Plain NCO** (eliminating return drag).
- **-14.22% maximum drawdown** versus the S&P 500's -23.93%.
- **0.94 Calmar ratio** (+38.2% versus S&P 500, +84.3% versus Baseline Regime-NCO).
- **75.0% reduction in regime switches** (from 16 to 4 switches across 65 months).
- **50% reduction in portfolio turnover** (from 60.78% to 30.48% monthly).

---

## Quickstart and Reproduction

### 1. Installation

```bash
git clone https://github.com/obo-cloud/sp500-asymmetric-nco.git
cd sp500-asymmetric-nco
pip install -r requirements.txt
```

### 2. Run Walk-Forward Backtesting

```bash
python src/run_backtest_robust.py
```

### 3. Run Ablation and Transaction Cost Sensitivity

```bash
python src/run_ablation.py
```

### 4. Run Hyperparameter Sensitivity and Statistical Inference (PSR / DSR / PBO)

```bash
python src/run_sensitivity_and_tests.py
```

### 5. Generate Figures

```bash
python src/plot_figures.py
```

Vector PDFs and PNGs are written to `figures/`.

---

## Repository Structure

```
sp500-asymmetric-nco/
├── data/
│   ├── sp500_daily_adjclose.csv            # Daily adjusted close prices (2018-2025)
│   ├── sp500_sectors_monthly_returns.csv   # 11 sector ETF monthly returns (2018-2025)
│   ├── sp500_macro_monthly.csv             # SPY, ^VIX, ^IRX monthly percentage returns
│   └── sp500_macro_levels_monthly.csv      # Historical end-of-month price/yield levels
├── figures/
│   ├── fig1_cumulative_wealth.pdf          # Out-of-sample equity curves
│   ├── fig2_drawdown.pdf                   # Drawdown underwater plot
│   ├── fig3_regime_comparison.pdf          # Baseline vs Sticky HMM timeline
│   ├── fig4_dendrogram_comparison.pdf      # Symmetric vs downside correlation trees
│   ├── fig5_sector_allocation.pdf          # Dynamic sector allocation weights
│   └── fig6_performance_comparison.pdf     # Risk-return trade-off bar chart
├── results/
│   ├── performance_metrics.csv            # Summary metrics table
│   ├── strategy_monthly_returns.csv       # Realized monthly returns series
│   ├── weights_proposed_asym_nco.csv      # Out-of-sample portfolio weights
│   ├── regime_timeline.csv                # Detected states and bull posteriors
│   ├── transaction_cost_sensitivity.csv   # Net CAGR under 0-20 bps slippage
│   ├── ablation_components.csv            # Step-by-step component decomposition
│   ├── sensitivity_grid.csv               # Kappa x delta hyperparameter grid
│   ├── statistical_tests.csv              # PSR / DSR per strategy
│   └── pbo_cscv.csv                       # CSCV probability of backtest overfitting
├── src/
│   ├── asymmetric_nco.py                  # Core downside NCO and sticky HMM module
│   ├── run_backtest_robust.py             # Walk-forward backtesting pipeline
│   ├── run_ablation.py                    # Component ablation and cost sensitivity
│   ├── run_sensitivity_and_tests.py       # Kappa/delta grid and PSR/DSR/PBO inference
│   ├── plot_figures.py                    # Figure generation script
│   └── download_data.py                   # Yahoo Finance data fetcher
├── LICENSE                                # MIT License
└── requirements.txt                       # Python dependencies
```

---

## Citation

If you use this work in your research, please cite the accompanying paper:

```bibtex
@misc{supriadi2026mitigating,
  author = {Supriadi},
  title = {Mitigating Return Drag and Regime Whipsaws in Sector Rotation: An Asymmetric Downside-Aware Nested Clustered Optimization Framework},
  year = {2026},
  eprint = {2611.05310},
  archivePrefix = {arXiv},
  primaryClass = {q-fin.PM}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
