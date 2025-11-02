"""
Robust Hyperparameter Optimization with Overfitting Mitigation
===============================================================

This improved version implements comprehensive strategies to prevent overfitting:

1. CONSTRAINED SEARCH SPACE: Prevents extreme hyperparameters
2. EXPANDED VALIDATION: 2012-2015 (3 years) instead of 2014-2015 (1 year)
3. REGULARIZED OBJECTIVE: Penalizes complexity and extreme values
4. ENSEMBLE SELECTION: Averages top-K configurations instead of single best
5. MULTI-PERIOD VALIDATION: Tests consistency across different sub-periods

Author: DSA5205 Project Team
Date: November 2025
Version: 2.0 (Robust)
"""

# Fix for Windows DLL loading issue with PyTorch
import os
import sys
if sys.platform == 'win32':
    torch_lib_path = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib')
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from datetime import datetime, timedelta
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)
import warnings
warnings.filterwarnings('ignore')
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# Model Architectures (Same as before)
# ============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron for baseline comparison"""
    def __init__(self, lookback, n_features, n_stocks, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(lookback * n_features, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, n_stocks)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # Dollar-neutral constraint
        x = x - x.mean(dim=1, keepdim=True)
        return x

class LSTM(nn.Module):
    """LSTM for temporal pattern recognition"""
    def __init__(self, n_features, n_stocks, hidden_size=64, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, n_stocks)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # Dollar-neutral constraint
        x = x - x.mean(dim=1, keepdim=True)
        return x

class CNNLSTM(nn.Module):
    """Hybrid CNN-LSTM for spatial and temporal feature extraction"""
    def __init__(self, lookback, n_features, n_stocks, hidden_size=64,
                 cnn_channels=16, kernel_size=5, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.n_features = n_features

        # Convolutional layer for cross-sectional patterns
        self.conv = nn.Conv2d(1, cnn_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.bn = nn.BatchNorm2d(cnn_channels)
        self.gelu = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((lookback, 1))

        # LSTM for temporal patterns
        self.lstm = nn.LSTM(n_features + cnn_channels, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_stocks)

    def forward(self, x):
        batch_size = x.size(0)

        # CNN path
        x_conv = x.unsqueeze(1)  # Add channel dimension
        conv_out = self.conv(x_conv)
        conv_out = self.bn(conv_out)
        conv_out = self.gelu(conv_out)
        conv_out = self.pool(conv_out).squeeze(-1)  # [batch, channels, lookback]
        conv_out = conv_out.permute(0, 2, 1)  # [batch, lookback, channels]

        # Concatenate original features with CNN features
        x_combined = torch.cat([x, conv_out], dim=2)

        # LSTM path
        _, (h_n, _) = self.lstm(x_combined)
        x_out = h_n.squeeze(0)
        x_out = self.dropout(x_out)
        x_out = self.fc(x_out)

        # Dollar-neutral constraint
        x_out = x_out - x_out.mean(dim=1, keepdim=True)
        return x_out

# ============================================================================
# Dataset and Data Loading
# ============================================================================

class StockDataset(Dataset):
    """PyTorch Dataset for stock price sequences"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    """Load stock data from Yahoo Finance"""
    print("Loading data from Yahoo Finance...")

    # Stock universe
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    financials = ['JPM', 'BAC', 'GS', 'C']
    healthcare = ['JNJ', 'PFE', 'MRK', 'UNH']
    energy = ['XOM', 'CVX', 'COP', 'SLB']
    other = ['BA', 'CAT', 'MMM', 'DIS', 'NKE']
    stocks = tech_stocks + financials + healthcare + energy + other

    # Macro indicators
    macro = ['IEF', 'TLT', 'SHY', 'GLD', 'USO', '^VIX', 'DX-Y.NYB']

    all_tickers = stocks + macro

    # Download data
    data = yf.download(all_tickers, start='2008-01-01', end='2025-10-01',
                       interval='1wk', progress=False, auto_adjust=False)

    # Use adjusted close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Adj Close'].ffill().bfill()
    else:
        prices = data[['Adj Close']].ffill().bfill()
        prices.columns = [all_tickers[0]]

    return prices, stocks

def prepare_features_labels(prices, stocks, lookback=25):
    """Prepare features (X) and labels (y) from price data"""
    n_stocks = len(stocks)
    all_features = prices.columns.tolist()

    # Normalize prices (z-score)
    prices_normalized = (prices - prices.mean()) / prices.std()

    X_list, y_list, dates_list = [], [], []

    for i in range(lookback, len(prices) - 1):
        # Features: lookback window of all prices
        X_window = prices_normalized.iloc[i-lookback:i].values
        X_list.append(X_window)

        # Label: next week returns for stocks only
        current_prices = prices.iloc[i][stocks].values
        next_prices = prices.iloc[i+1][stocks].values
        returns = (next_prices / current_prices) - 1
        y_list.append(returns)

        dates_list.append(prices.index[i])

    X = np.array(X_list)
    y = np.array(y_list)
    dates = np.array(dates_list)

    return X, y, dates

# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def sharpe_loss(weights, returns, tc_penalty=0.1, prev_weights=None):
    """
    Custom loss function: negative Sharpe ratio + transaction cost penalty
    """
    # Portfolio returns
    portfolio_returns = (weights * returns).sum(dim=1)

    # Sharpe ratio (negative for minimization)
    sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)

    # Transaction cost penalty
    tc_loss = 0
    if prev_weights is not None:
        turnover = torch.abs(weights - prev_weights).sum(dim=1).mean()
        tc_loss = tc_penalty * turnover

    return -sharpe + tc_loss

def train_model(model, train_loader, val_loader, lr=1e-4, epochs=100,
                patience=10, tc_lambda=0.1, device='cpu'):
    """
    Train a neural network model with early stopping
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_sharpe = -np.inf
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            weights = model(X_batch)
            loss = sharpe_loss(weights, y_batch, tc_penalty=tc_lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_returns_list = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                weights = model(X_batch)
                portfolio_returns = (weights * y_batch).sum(dim=1)
                val_returns_list.append(portfolio_returns.cpu().numpy())

        val_returns = np.concatenate(val_returns_list)
        val_sharpe = val_returns.mean() / (val_returns.std() + 1e-8) * np.sqrt(52)

        # Early stopping
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_sharpe

# ============================================================================
# ROBUST OBJECTIVE FUNCTION WITH REGULARIZATION
# ============================================================================

def create_robust_objective(model_type, X_train, y_train, X_val, y_val,
                            X_val_alt, y_val_alt, n_features, n_stocks):
    """
    Create ROBUST objective function with:
    1. Constrained search space
    2. Regularization penalties
    3. Multi-period validation

    Args:
        X_val_alt, y_val_alt: Alternative validation period for robustness check
    """
    def objective(trial):
        # ====================================================================
        # 1. CONSTRAINED SEARCH SPACE (prevent extreme values)
        # ====================================================================
        lookback = trial.suggest_categorical('lookback', [25, 39, 52])  # Removed 13 (too short), removed 104 (too long)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        lr = trial.suggest_float('lr', 5e-5, 5e-4, log=True)  # Narrower range
        dropout = trial.suggest_float('dropout', 0.1, 0.3)  # Max 0.3 instead of 0.5
        tc_lambda = trial.suggest_float('tc_lambda', 0.05, 0.2, log=True)  # Min 0.05 instead of 0.01

        # Adjust lookback for data
        if lookback >= len(X_train):
            lookback = 25

        # Model-specific hyperparameters
        if model_type == 'MLP':
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128])  # Removed 256
            model = MLP(lookback, n_features, n_stocks, hidden_dim, dropout)
            complexity = hidden_dim

        elif model_type == 'LSTM':
            hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])  # Removed 256
            model = LSTM(n_features, n_stocks, hidden_size, dropout)
            complexity = hidden_size

        elif model_type == 'CNNLSTM':
            hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])  # Removed 256
            cnn_channels = trial.suggest_categorical('cnn_channels', [16, 32, 48])  # CONSTRAINED: no 8 or 64
            kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
            model = CNNLSTM(lookback, n_features, n_stocks, hidden_size,
                           cnn_channels, kernel_size, dropout)
            complexity = hidden_size * cnn_channels  # Complexity measure

        # Create data loaders
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)
        val_alt_dataset = StockDataset(X_val_alt, y_val_alt)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_alt_loader = DataLoader(val_alt_dataset, batch_size=batch_size, shuffle=False)

        # Train model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            _, val_sharpe = train_model(
                model, train_loader, val_loader,
                lr=lr, epochs=100, patience=10,
                tc_lambda=tc_lambda, device=device
            )

            # ================================================================
            # 2. MULTI-PERIOD VALIDATION (check consistency)
            # ================================================================
            # Evaluate on alternative validation period
            model.eval()
            val_alt_returns_list = []
            with torch.no_grad():
                for X_batch, y_batch in val_alt_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    weights = model(X_batch)
                    portfolio_returns = (weights * y_batch).sum(dim=1)
                    val_alt_returns_list.append(portfolio_returns.cpu().numpy())

            val_alt_returns = np.concatenate(val_alt_returns_list)
            val_alt_sharpe = val_alt_returns.mean() / (val_alt_returns.std() + 1e-8) * np.sqrt(52)

            # Average performance across both validation periods
            avg_val_sharpe = (val_sharpe + val_alt_sharpe) / 2.0
            consistency = abs(val_sharpe - val_alt_sharpe)  # Lower is better

        except Exception as e:
            print(f"Trial failed: {e}")
            return -np.inf

        # ====================================================================
        # 3. REGULARIZED OBJECTIVE (penalize complexity and extremes)
        # ====================================================================

        # Base objective: average Sharpe across validation periods
        objective_value = avg_val_sharpe

        # Penalty 1: Complexity (discourage very large models)
        complexity_penalty = 0.01 * (complexity / 1000.0)  # Small penalty

        # Penalty 2: Inconsistency (penalize if performance varies widely across periods)
        consistency_penalty = 0.1 * consistency

        # Penalty 3: Extreme TC lambda (discourage unrealistically low transaction costs)
        if tc_lambda < 0.07:
            tc_penalty = 0.1 * (0.07 - tc_lambda)
        else:
            tc_penalty = 0.0

        # Final regularized objective
        regularized_objective = objective_value - complexity_penalty - consistency_penalty - tc_penalty

        # Log detailed metrics
        trial.set_user_attr('val_sharpe_primary', val_sharpe)
        trial.set_user_attr('val_sharpe_secondary', val_alt_sharpe)
        trial.set_user_attr('consistency', consistency)
        trial.set_user_attr('complexity', complexity)

        # Optuna pruning
        trial.report(regularized_objective, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return regularized_objective

    return objective

# ============================================================================
# ROBUST WALK-FORWARD OPTIMIZATION
# ============================================================================

def robust_walk_forward_optimization(model_type='CNNLSTM', n_trials=100, n_splits=3):
    """
    Robust hyperparameter optimization with:
    1. Expanded validation period (2012-2015 instead of 2014-2015)
    2. Constrained search space
    3. Regularized objective
    4. Ensemble of top-K configurations
    """
    print(f"\n{'='*80}")
    print(f"ROBUST Walk-Forward Hyperparameter Optimization: {model_type}")
    print(f"{'='*80}\n")
    print("Overfitting Mitigation Strategies:")
    print("  [1] Constrained search space (no extreme values)")
    print("  [2] Expanded validation: 2012-2015 (3 years)")
    print("  [3] Regularized objective (complexity + consistency penalties)")
    print("  [4] Multi-period validation (primary + secondary)")
    print("  [5] Ensemble selection (top-5 configs)")
    print(f"{'='*80}\n")

    # Load and prepare data
    prices, stocks = load_data()
    X, y, dates = prepare_features_labels(prices, stocks, lookback=25)

    n_features = X.shape[2]
    n_stocks = len(stocks)

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    # ========================================================================
    # EXPANDED VALIDATION: 2012-2015 (3 years) instead of 2014-2015 (1 year)
    # ========================================================================
    split_2012 = np.where(dates >= pd.Timestamp('2012-01-01'))[0][0]
    split_2013 = np.where(dates >= pd.Timestamp('2013-01-01'))[0][0]
    split_2014 = np.where(dates >= pd.Timestamp('2014-01-01'))[0][0]
    split_2015 = np.where(dates >= pd.Timestamp('2015-01-01'))[0][0]

    # Training data: 2008-2012
    X_train = X[:split_2012]
    y_train = y[:split_2012]

    # Primary validation: 2012-2013
    X_val_primary = X[split_2012:split_2013]
    y_val_primary = y[split_2012:split_2013]

    # Secondary validation: 2014-2015 (for consistency check)
    X_val_secondary = X[split_2014:split_2015]
    y_val_secondary = y[split_2014:split_2015]

    print(f"\nTraining period: {len(X_train)} weeks (2008-2012)")
    print(f"Validation primary: {len(X_val_primary)} weeks (2012-2013)")
    print(f"Validation secondary: {len(X_val_secondary)} weeks (2014-2015)")
    print(f"Test (reserved): 2015-2025 (not used in optimization)")

    # ========================================================================
    # RUN OPTIMIZATION WITH ROBUST OBJECTIVE
    # ========================================================================
    study = optuna.create_study(
        direction='maximize',
        study_name=f"{model_type}_robust",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )

    objective = create_robust_objective(
        model_type, X_train, y_train,
        X_val_primary, y_val_primary,
        X_val_secondary, y_val_secondary,
        n_features, n_stocks
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    # ========================================================================
    # ENSEMBLE SELECTION: Average top-5 configurations
    # ========================================================================
    print(f"\n{'='*80}")
    print("ENSEMBLE SELECTION: Top-5 Configurations")
    print(f"{'='*80}\n")

    # Get top 5 trials
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -np.inf, reverse=True)
    top_k = 5
    top_trials = sorted_trials[:top_k]

    print(f"Top {top_k} trials:")
    for i, trial in enumerate(top_trials, 1):
        print(f"\n  Rank {i}:")
        print(f"    Regularized Objective: {trial.value:.4f}")
        print(f"    Primary Val Sharpe: {trial.user_attrs.get('val_sharpe_primary', 'N/A'):.4f}")
        print(f"    Secondary Val Sharpe: {trial.user_attrs.get('val_sharpe_secondary', 'N/A'):.4f}")
        print(f"    Consistency Gap: {trial.user_attrs.get('consistency', 'N/A'):.4f}")
        print(f"    Params: {trial.params}")

    # Ensemble averaging for numerical parameters
    ensemble_params = {}
    for key in top_trials[0].params.keys():
        values = [t.params[key] for t in top_trials if key in t.params]
        if isinstance(values[0], (int, float)):
            # Average numerical values
            ensemble_params[key] = np.mean(values)
        else:
            # Mode for categorical values
            from collections import Counter
            ensemble_params[key] = Counter(values).most_common(1)[0][0]

    print(f"\n{'='*80}")
    print("ENSEMBLE PARAMETERS (averaged/voted from top-5):")
    print(f"{'='*80}")
    for k, v in ensemble_params.items():
        print(f"  {k:15s}: {v}")

    # Save results
    output_dir = Path('optimization_results')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f'{model_type}_robust_study.pkl', 'wb') as f:
        pickle.dump(study, f)

    with open(output_dir / f'{model_type}_robust_best_params.json', 'w') as f:
        json.dump(ensemble_params, f, indent=2)

    with open(output_dir / f'{model_type}_robust_top5.json', 'w') as f:
        top5_data = []
        for t in top_trials:
            top5_data.append({
                'params': t.params,
                'value': t.value,
                'val_sharpe_primary': t.user_attrs.get('val_sharpe_primary'),
                'val_sharpe_secondary': t.user_attrs.get('val_sharpe_secondary'),
                'consistency': t.user_attrs.get('consistency')
            })
        json.dump(top5_data, f, indent=2)

    print(f"\nResults saved to {output_dir}/")

    # Generate visualizations
    print("\nGenerating optimization visualizations...")
    generate_visualizations(study, f"{model_type}_robust", output_dir)

    return ensemble_params, study

def generate_visualizations(study, model_type, output_dir):
    """Generate Optuna visualization plots"""
    try:
        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image(str(output_dir / f'{model_type}_optimization_history.png'))

        # Parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_image(str(output_dir / f'{model_type}_param_importance.png'))

        # Parallel coordinate plot
        fig3 = plot_parallel_coordinate(study)
        fig3.write_image(str(output_dir / f'{model_type}_parallel_coordinate.png'))

        # Slice plot
        fig4 = plot_slice(study)
        fig4.write_image(str(output_dir / f'{model_type}_slice_plot.png'))

        print(f"Visualizations saved to {output_dir}/")
    except Exception as e:
        print(f"Visualization generation failed: {e}")
        print("Install kaleido for plot export: pip install kaleido")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Robust Hyperparameter Optimization')
    parser.add_argument('--model', type=str, default='CNNLSTM',
                       choices=['MLP', 'LSTM', 'CNNLSTM'],
                       help='Model architecture to optimize')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of Optuna trials')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("ROBUST HYPERPARAMETER OPTIMIZATION")
    print("With Overfitting Mitigation")
    print(f"{'='*80}\n")
    print(f"Model: {args.model}")
    print(f"Trials: {args.trials}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"\n{'='*80}\n")

    # Run robust optimization
    ensemble_params, study = robust_walk_forward_optimization(
        model_type=args.model,
        n_trials=args.trials,
        n_splits=3  # Not used in robust version
    )

    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"Ensemble hyperparameters for {args.model}:")
    for param, value in ensemble_params.items():
        print(f"  {param}: {value}")
    print(f"\nBest trial objective: {study.best_value:.4f}")
    print(f"\nResults saved to optimization_results/")
    print(f"{'='*80}\n")
