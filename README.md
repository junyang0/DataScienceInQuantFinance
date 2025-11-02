# DSA5205: Neural Network-Based Dollar-Neutral Trading Strategy

**Institution**: National University of Singapore
**Course**: DSA5205 - Data Science in Quantitative Finance
**Submission**: November 14, 2025

## Overview

This project implements machine learning-based systematic trading strategies using neural networks to generate dollar-neutral portfolio allocations across 22 US large-cap stocks. Models are trained on 2008-2015 data and evaluated out-of-sample from 2015-2025 with realistic transaction costs.

### Performance Summary (Out-of-Sample 2015-2025)

| Strategy | Sharpe (Net) | Annual Return | Max Drawdown | Turnover |
|----------|-------------|---------------|--------------|----------|
| CNN-LSTM | 0.970 | 15.36% | -21.02% | 33.45% |
| MLP | 0.064 | -2.70% | -82.64% | 60.07% |
| LSTM | -0.248 | -1.68% | -24.52% | 14.51% |
| SPY | 0.874 | 14.47% | -31.83% | 0% |

**Key Finding**: CNN-LSTM achieves 10.9% higher Sharpe than SPY (0.970 vs 0.874) with 34% lower drawdown, but the difference is not statistically significant (p=0.4176, bootstrap N=10,000). Performance is comparable to passive indexing with enhanced risk management.

---

## Repository Structure

```
DataScienceInQuantFinance/
├── code.ipynb                      # Main implementation (5 cells)
├── FINAL_REPORT.md                 # Academic report (publication-quality)
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── hyperparameter_optimization*.py # Bayesian optimization (optional)
├── *_oos_weights.csv              # Out-of-sample portfolio weights (3 files)
├── performance_*.csv/png          # Results and visualizations (4 files)
└── optimization_results/          # Hyperparameter tuning outputs (optional)
```

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- 8GB RAM minimum (16GB recommended)
- Internet connection for data download

### Installation

```bash
# Navigate to project directory
cd DataScienceInQuantFinance

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook code.ipynb
```

---

## Running the Code

### Quick Start

Run all cells in `code.ipynb` via: `Cell` → `Run All`

**Expected Runtime**: 15-30 minutes

### Notebook Structure

**Cell 0-1**: Setup and data loading (30 seconds)
- Imports libraries, sets random seeds
- Downloads 22 US stocks + 7 macro indicators from Yahoo Finance (2008-2025)

**Cell 2-3**: Model training (10-20 minutes)
- Defines three architectures: MLP (baseline), LSTM (temporal), CNN-LSTM (hybrid)
- Implements rolling window training with yearly retraining
- Trains 5-model ensemble per architecture
- Generates out-of-sample weights

**Cell 4**: Backtesting (10 seconds)
- Applies 10 bps transaction costs
- Computes comprehensive performance metrics
- Compares against SPY benchmark

**Cell 5-6**: Visualization and statistics (30 seconds)
- Generates publication-quality plots
- Bootstrap confidence intervals (N=10,000)
- Regime-specific analysis (Pre-COVID, COVID, Recovery periods)

### Expected Output

Generated files:
- `MLP_oos_weights.csv`, `LSTM_oos_weights.csv`, `CNNLSTM_oos_weights.csv`
- `performance_comparison.csv`
- `performance_plots_1.png`, `performance_plots_2.png`, `oos_r2_analysis.png`
- `statistical_test_results.csv`, `regime_analysis_results.csv`, `yearly_performance.csv`

---

## Methodology

### Data Specification
- **Universe**: 22 US large-cap stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, BAC, GS, C, JNJ, PFE, MRK, UNH, XOM, CVX, COP, SLB, BA, CAT, MMM, DIS, NKE)
- **Macro Indicators**: IEF, TLT, SHY (bonds), GLD, USO, ^VIX, DX-Y.NYB
- **Frequency**: Weekly (Friday close)
- **Period**: January 2008 - October 2025
- **Features**: 29 total (22 stocks + 7 macros), 25-week lookback window
- **Label**: Weekly return = (Friday close / Monday open - 1), forward-shifted

### Model Architectures

**MLP (Baseline)**: Flatten(25×29) → FC(128) → ReLU → Dropout(0.3) → FC(64) → FC(22)
**LSTM**: LSTM(input=29, hidden=64) → Dropout(0.3) → FC(64) → FC(22)
**CNN-LSTM (Best)**: Conv2D(1→16, kernel 1×5) → BatchNorm → GELU → LSTM(64) → FC(22)

All models enforce dollar-neutral constraint: `weights = weights - mean(weights)`

**Economic Rationale**: CNN-LSTM captures both cross-sectional patterns (sector rotations via Conv2D) and temporal dependencies (momentum/mean-reversion via LSTM).

### Training Details
- **Train window**: 252 weeks (5 years)
- **Validation window**: 50 weeks (1 year)
- **Test period**: October 2015 - September 2025 (519 weeks)
- **Retraining**: Yearly (January of each year)
- **Optimizer**: Adam (lr=1e-4, batch size=64)
- **Loss function**: -Sharpe + λ × transaction_cost_penalty (λ=0.1)
- **Ensemble**: 5 models with different seeds per architecture

### Overfitting Controls
1. Dollar-neutral constraint (eliminates trivial market exposure)
2. Dropout regularization (10-30%)
3. Early stopping (patience=10 epochs)
4. Ensemble averaging (5 models)
5. Rolling window validation (strict temporal ordering)
6. Transaction cost penalty in loss function

### Transaction Costs
- **Rate**: 10 basis points (0.001) per trade
- **Turnover**: `sum(|weights_t - weights_{t-1}|)`
- **Net returns**: `portfolio_return - turnover × 0.001`

---

## Results

### Key Findings

1. **Performance**: CNN-LSTM Sharpe 0.970 vs SPY 0.874 (10.9% improvement)
2. **Risk Management**: Maximum drawdown -21.02% vs SPY -31.83% (34% reduction)
3. **Transaction Costs**: 10.1% Sharpe degradation (1.077 gross → 0.970 net) due to 33.45% weekly turnover
4. **Statistical Significance**: NOT significant (p=0.4176). Wide confidence intervals [0.34, 1.61] due to limited 10-year sample
5. **Regime Dependence**:
   - Pre-COVID (2015-2020): Sharpe 1.49 vs SPY 0.83 (strong outperformance)
   - COVID Crisis (2020): Sharpe 0.34 vs SPY 1.06 (underperformance)
   - Recovery (2021-2025): Sharpe 0.74 vs SPY 0.91 (slight underperformance)
6. **Year-over-Year**: Outperformed SPY in 5 of 10 years (50% win rate)

### Interpretation

While CNN-LSTM shows point estimate outperformance, bootstrap analysis reveals the difference is not statistically significant. Performance is better characterized as comparable to passive indexing with enhanced risk management rather than consistent alpha generation.

---

## Hyperparameter Optimization (Optional)

### Quick Start

```bash
# Install additional dependencies
pip install optuna plotly kaleido

# Run Bayesian optimization for CNN-LSTM
python hyperparameter_optimization.py --model CNNLSTM --trials 100 --splits 3
```

**Runtime**: 8-12 hours (CPU) or 3-5 hours (GPU)

### Validation Study Results (Section 8 of FINAL_REPORT.md)

| Approach | Val Sharpe | Test Sharpe | Interpretation |
|----------|-----------|------------|----------------|
| Domain-Expert (Static) | ~1.0 | 0.969 | Near-optimal without search |
| Standard Optimization | 3.09 | 0.453 | Severe overfitting (85% degradation) |
| Robust Optimization | 1.92 | 0.485 | Reduced overfitting, still underperformed |

**Conclusion**: Domain-expert hyperparameter selection outperformed systematic optimization, validating architectural reasoning over automated search when validation data is limited.

### Search Space

- Lookback: [13, 25, 39, 52, 104] weeks
- Learning rate: [1e-5, 1e-3] log-uniform
- Dropout: [0.1, 0.5] uniform
- TC penalty λ: [0.01, 1.0] log-uniform
- CNN channels: [8, 16, 32, 64]
- LSTM hidden: [32, 64, 128, 256]

**Objective**: Maximize validation Sharpe ratio (net of costs) via Optuna TPE sampler with nested walk-forward cross-validation.

---

## Reproducibility

### Random Seeds
All experiments use `seed=42` for deterministic results:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

### Data
- Downloaded fresh from Yahoo Finance each run
- All adjustments (splits, dividends) handled by `yfinance`
- No local caching to ensure consistency

### Computational Requirements
- **CPU**: Any modern processor (Intel i7/AMD Ryzen/Apple M1)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (CPU execution supported)
- **Storage**: ~500MB

---

## Troubleshooting

**Issue**: `yfinance` download fails
**Solution**: Check internet connection. Yahoo Finance may rate-limit. Wait 1 minute and retry.

**Issue**: Out of memory
**Solution**: Reduce batch size from 64 to 32 in Cell 2

**Issue**: PyTorch not detecting GPU
**Solution**: Install GPU-specific PyTorch from https://pytorch.org/

**Issue**: Plots not displaying
**Solution**: Add `%matplotlib inline` to Cell 0

---

## Documentation

- **FINAL_REPORT.md**: Comprehensive academic report with literature review, methodology, results, and discussion
- **code.ipynb**: Fully documented implementation with markdown cells
- **requirements.txt**: All dependencies with version numbers

---

## References

1. Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium
2. Ross, S. A. (1976). The arbitrage theory of capital asset pricing
3. Fama, E. F., & French, K. R. (1993). Common risk factors in returns
4. Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers
5. Heaton, J. B., Polson, N. G., & Witte, J. H. (2017). Deep learning for finance
6. Harvey, C. R., Liu, Y., & Zhu, H. (2016). ...and the cross-section of expected returns

Complete bibliography in FINAL_REPORT.md.

---

**Version**: 1.0
**Last Updated**: November 2025
**Status**: On-going
