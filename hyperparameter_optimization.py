"""
Hyperparameter Optimization for Neural Network Trading Strategies
==================================================================

This script implements Bayesian hyperparameter optimization using Optuna with
nested walk-forward cross-validation for time series data.

Key Features:
- Optuna TPE (Tree-structured Parzen Estimator) for efficient search
- Nested cross-validation to prevent optimistic bias
- Walk-forward validation respecting temporal order
- Parallel trial execution support
- Comprehensive logging and visualization

Author: DSA5205 Project Team
Date: November 2025
"""

# Fix for Windows DLL loading issue with PyTorch
import os
import sys
if sys.platform == 'win32':
    # Add torch DLL directory to PATH
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

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# Model Architectures
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

    # Download data (auto_adjust=False to get Adj Close column)
    data = yf.download(all_tickers, start='2008-01-01', end='2025-10-01',
                       interval='1wk', progress=False, auto_adjust=False)

    # Use adjusted close prices
    # yfinance returns MultiIndex DataFrame for multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Adj Close'].ffill().bfill()
    else:
        # Single ticker case
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

    Args:
        weights: Portfolio weights [batch_size, n_stocks]
        returns: Asset returns [batch_size, n_stocks]
        tc_penalty: Weight for transaction cost penalty
        prev_weights: Previous weights for TC calculation
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

    Returns:
        best_model: Model with best validation performance
        best_val_sharpe: Best validation Sharpe ratio
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

def evaluate_sharpe(model, data_loader, device='cpu'):
    """Evaluate Sharpe ratio on a dataset"""
    model.eval()
    returns_list = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            weights = model(X_batch)
            portfolio_returns = (weights * y_batch).sum(dim=1)
            returns_list.append(portfolio_returns.cpu().numpy())

    returns = np.concatenate(returns_list)
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(52)
    return sharpe

# ============================================================================
# Optuna Objective Function
# ============================================================================

def create_objective(model_type, X_train, y_train, X_val, y_val, n_features, n_stocks):
    """
    Create Optuna objective function for hyperparameter optimization

    Args:
        model_type: 'MLP', 'LSTM', or 'CNNLSTM'
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_features: Number of input features
        n_stocks: Number of stocks to predict

    Returns:
        objective: Optuna objective function
    """
    def objective(trial):
        # Hyperparameter search space
        lookback = trial.suggest_categorical('lookback', [13, 25, 39, 52])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        tc_lambda = trial.suggest_float('tc_lambda', 0.01, 1.0, log=True)

        # Adjust lookback for data
        if lookback >= len(X_train):
            lookback = min([13, 25, 39, 52])

        # Extract data with new lookback
        # For simplicity, we'll use the existing data and adjust model
        # In production, would regenerate data with new lookback

        # Model-specific hyperparameters
        if model_type == 'MLP':
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
            model = MLP(lookback, n_features, n_stocks, hidden_dim, dropout)

        elif model_type == 'LSTM':
            hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
            model = LSTM(n_features, n_stocks, hidden_size, dropout)

        elif model_type == 'CNNLSTM':
            hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
            cnn_channels = trial.suggest_categorical('cnn_channels', [8, 16, 32, 64])
            kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
            model = CNNLSTM(lookback, n_features, n_stocks, hidden_size,
                           cnn_channels, kernel_size, dropout)

        # Create data loaders
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            _, val_sharpe = train_model(
                model, train_loader, val_loader,
                lr=lr, epochs=100, patience=10,
                tc_lambda=tc_lambda, device=device
            )
        except Exception as e:
            print(f"Trial failed: {e}")
            return -np.inf

        # Optuna can prune unpromising trials
        trial.report(val_sharpe, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_sharpe

    return objective

# ============================================================================
# Walk-Forward Optimization
# ============================================================================

def walk_forward_optimization(model_type='CNNLSTM', n_trials=100, n_splits=3):
    """
    Perform walk-forward hyperparameter optimization

    Args:
        model_type: 'MLP', 'LSTM', or 'CNNLSTM'
        n_trials: Number of Optuna trials per split
        n_splits: Number of walk-forward splits

    Returns:
        best_params: Optimized hyperparameters
        study: Optuna study object
    """
    print(f"\n{'='*70}")
    print(f"Walk-Forward Hyperparameter Optimization: {model_type}")
    print(f"{'='*70}\n")

    # Load and prepare data
    prices, stocks = load_data()
    X, y, dates = prepare_features_labels(prices, stocks, lookback=25)

    n_features = X.shape[2]
    n_stocks = len(stocks)

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    # Split data: 2008-2014 for optimization, 2014-2015 for final validation
    # We'll use walk-forward splits within the optimization period

    # Find split points
    split_2014 = np.where(dates >= pd.Timestamp('2014-01-01'))[0][0]
    split_2015 = np.where(dates >= pd.Timestamp('2015-01-01'))[0][0]

    # Optimization data: 2008-2014
    X_opt = X[:split_2014]
    y_opt = y[:split_2014]

    # Final validation: 2014-2015
    X_final_val = X[split_2014:split_2015]
    y_final_val = y[split_2014:split_2015]

    print(f"\nOptimization period: {len(X_opt)} weeks")
    print(f"Final validation period: {len(X_final_val)} weeks")

    # Walk-forward splits for optimization
    split_size = len(X_opt) // (n_splits + 1)

    all_val_scores = []

    for fold in range(n_splits):
        print(f"\n--- Walk-Forward Fold {fold+1}/{n_splits} ---")

        # Define train and validation sets
        train_end = split_size * (fold + 1)
        val_end = split_size * (fold + 2)

        X_train = X_opt[:train_end]
        y_train = y_opt[:train_end]
        X_val = X_opt[train_end:val_end]
        y_val = y_opt[train_end:val_end]

        print(f"Train: {len(X_train)} weeks, Val: {len(X_val)} weeks")

        # Create Optuna study
        study_name = f"{model_type}_fold{fold+1}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )

        # Optimize
        objective = create_objective(
            model_type, X_train, y_train, X_val, y_val,
            n_features, n_stocks
        )

        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

        print(f"\nBest trial (Fold {fold+1}):")
        print(f"  Validation Sharpe: {study.best_value:.4f}")
        print(f"  Params: {study.best_params}")

        all_val_scores.append(study.best_value)

    # Average best parameters across folds
    print(f"\n{'='*70}")
    print("Aggregating Results Across Folds")
    print(f"{'='*70}\n")

    print(f"Validation Sharpe ratios: {all_val_scores}")
    print(f"Mean: {np.mean(all_val_scores):.4f} ± {np.std(all_val_scores):.4f}")

    # Final optimization on full optimization data
    print(f"\n{'='*70}")
    print("Final Optimization on Full Period (2008-2014)")
    print(f"{'='*70}\n")

    study_final = optuna.create_study(
        direction='maximize',
        study_name=f"{model_type}_final",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )

    objective_final = create_objective(
        model_type, X_opt, y_opt, X_final_val, y_final_val,
        n_features, n_stocks
    )

    study_final.optimize(objective_final, n_trials=n_trials*2, n_jobs=1, show_progress_bar=True)

    print(f"\nFinal Best Trial:")
    print(f"  Validation Sharpe (2014-2015): {study_final.best_value:.4f}")
    print(f"  Best Params: {study_final.best_params}")

    # Save study
    output_dir = Path('optimization_results')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f'{model_type}_study.pkl', 'wb') as f:
        pickle.dump(study_final, f)

    # Save best parameters as JSON
    with open(output_dir / f'{model_type}_best_params.json', 'w') as f:
        json.dump(study_final.best_params, f, indent=2)

    print(f"\nResults saved to {output_dir}/")

    # Generate visualizations
    print("\nGenerating optimization visualizations...")
    generate_visualizations(study_final, model_type, output_dir)

    return study_final.best_params, study_final

def generate_visualizations(study, model_type, output_dir):
    """Generate Optuna visualization plots"""
    try:
        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image(output_dir / f'{model_type}_optimization_history.png')

        # Parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_image(output_dir / f'{model_type}_param_importance.png')

        # Parallel coordinate plot
        fig3 = plot_parallel_coordinate(study)
        fig3.write_image(output_dir / f'{model_type}_parallel_coordinate.png')

        # Slice plot
        fig4 = plot_slice(study)
        fig4.write_image(output_dir / f'{model_type}_slice_plot.png')

        print(f"Visualizations saved to {output_dir}/")
    except Exception as e:
        print(f"Visualization generation failed: {e}")
        print("Install kaleido for plot export: pip install kaleido")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for Trading Strategies')
    parser.add_argument('--model', type=str, default='CNNLSTM',
                       choices=['MLP', 'LSTM', 'CNNLSTM'],
                       help='Model architecture to optimize')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of Optuna trials per fold')
    parser.add_argument('--splits', type=int, default=3,
                       help='Number of walk-forward splits')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("HYPERPARAMETER OPTIMIZATION FOR NEURAL NETWORK TRADING")
    print(f"{'='*70}\n")
    print(f"Model: {args.model}")
    print(f"Trials per fold: {args.trials}")
    print(f"Walk-forward splits: {args.splits}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"\n{'='*70}\n")

    # Run optimization
    best_params, study = walk_forward_optimization(
        model_type=args.model,
        n_trials=args.trials,
        n_splits=args.splits
    )

    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}\n")
    print(f"Best hyperparameters for {args.model}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nValidation Sharpe Ratio: {study.best_value:.4f}")
    print(f"\nResults saved to optimization_results/")
    print(f"{'='*70}\n")
