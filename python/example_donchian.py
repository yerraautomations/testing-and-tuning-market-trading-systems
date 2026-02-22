"""
Full 4-Step Validation Example: Donchian Channel Breakout

This script walks through the exact process described in the YouTube video:

  Step 1: In-Sample Excellence
    - Optimize the Donchian lookback on training data
    - Visually inspect the equity curve
    - Ask: "Is this excellent?" and "Is this obviously overfit?"

  Step 2: In-Sample MCPT
    - Permute price data, re-optimize on each permutation
    - Compare real performance vs. permutation distribution
    - Low p-value (< 1%) = pass

  Step 3: Walk-Forward Test
    - Re-optimize periodically on rolling windows
    - Test on unseen data
    - Ask: "Is this worth trading?"

  Step 4: Walk-Forward MCPT
    - Permute only the OOS data
    - Compare real walk-forward results vs. permutations
    - Low p-value (< 5% for 1 year, < 1% for 2+ years) = pass

Usage:
    1. Place your OHLC CSV data (with columns: Date, Open, High, Low, Close)
       in the same directory, or modify DATA_FILE below.
    2. Adjust the date ranges and parameters as needed.
    3. Run: python example_donchian.py

    If no CSV is found, the script generates synthetic data for demonstration.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies import donchian_signal, optimize_donchian
from objectives import profit_factor, sharpe_ratio, get_strategy_returns
from mcpt import insample_mcpt, walkforward_mcpt, walkforward_signal
from permutation import get_permutation


# ============================================================================
# Configuration
# ============================================================================

DATA_FILE = None  # Set to your CSV path, e.g., "btc_hourly.csv"
TRAIN_YEARS = 4   # Years of data for in-sample / training
TEST_YEARS = 1    # Years of data for walk-forward OOS

# Donchian lookback range to search
LOOKBACK_RANGE = range(5, 200)

# Walk-forward settings (assuming hourly data)
HOURS_PER_DAY = 24
TRAIN_BARS = TRAIN_YEARS * 365 * HOURS_PER_DAY   # ~35,040 bars
TRAIN_STEP = 30 * HOURS_PER_DAY                   # Re-optimize every 30 days

# MCPT settings
IS_PERMUTATIONS = 1000   # In-sample permutations (1000 recommended)
WF_PERMUTATIONS = 200    # Walk-forward permutations (slower, 200 is ok)


# ============================================================================
# Data Loading
# ============================================================================

def generate_synthetic_data(n_bars=5 * 365 * 24, seed=42):
    """Generate synthetic OHLC data for demonstration when no real data."""
    rng = np.random.default_rng(seed)

    # Geometric Brownian Motion with slight drift and volatility clustering
    dt = 1 / (365 * 24)
    mu = 0.5      # annual drift
    sigma = 0.8   # annual volatility

    log_returns = mu * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_bars)

    # Add some volatility clustering via GARCH-like effect
    vol = np.ones(n_bars) * sigma * np.sqrt(dt)
    for i in range(1, n_bars):
        vol[i] = np.sqrt(0.9 * vol[i - 1]**2 +
                         0.1 * log_returns[i - 1]**2)
        log_returns[i] = mu * dt + vol[i] * rng.standard_normal()

    # Build close prices
    close = 10000 * np.exp(np.cumsum(log_returns))

    # Build OHLC from close prices
    intrabar_noise = rng.uniform(0.001, 0.005, n_bars)
    open_prices = close * np.exp(rng.normal(0, 0.001, n_bars))
    high = np.maximum(open_prices, close) * (1 + intrabar_noise)
    low = np.minimum(open_prices, close) * (1 - intrabar_noise)

    dates = pd.date_range("2016-01-01", periods=n_bars, freq="h")

    return pd.DataFrame({
        "Date": dates,
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
    }).set_index("Date")


def load_data():
    """Load OHLC data from CSV or generate synthetic data."""
    if DATA_FILE and os.path.exists(DATA_FILE):
        print(f"Loading data from {DATA_FILE}")
        df = pd.read_csv(DATA_FILE, parse_dates=["Date"], index_col="Date")
        # Ensure column names are standardized
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if "open" in lower:
                col_map[col] = "Open"
            elif "high" in lower:
                col_map[col] = "High"
            elif "low" in lower:
                col_map[col] = "Low"
            elif "close" in lower:
                col_map[col] = "Close"
        df = df.rename(columns=col_map)
        return df
    else:
        print("No data file found. Generating synthetic OHLC data.")
        print("(Set DATA_FILE to your CSV path for real data.)\n")
        return generate_synthetic_data()


# ============================================================================
# Wrapper functions that match the MCPT interface
# ============================================================================

def optimize_on_data(data):
    """
    Optimize Donchian on a DataFrame. Returns (objective, signal).
    Used by insample_mcpt.
    """
    close = data["Close"].values
    best_lb, best_obj, best_sig = optimize_donchian(
        close, lookback_range=LOOKBACK_RANGE, objective_func=profit_factor
    )
    return best_obj, best_sig


def optimize_on_data_with_params(data):
    """
    Optimize Donchian on a DataFrame. Returns (objective, signal, params).
    Used by walkforward functions.
    """
    close = data["Close"].values
    best_lb, best_obj, best_sig = optimize_donchian(
        close, lookback_range=LOOKBACK_RANGE, objective_func=profit_factor
    )
    return best_obj, best_sig, best_lb


def signal_from_params(data, params):
    """
    Generate Donchian signal using specific parameters.
    Used by walkforward functions.
    """
    close = data["Close"].values
    return donchian_signal(close, lookback=params)


# ============================================================================
# Main: The 4-Step Validation Process
# ============================================================================

def main():
    print("=" * 70)
    print("  4-STEP TRADING STRATEGY VALIDATION FRAMEWORK")
    print("  Strategy: Donchian Channel Breakout (optimized lookback)")
    print("=" * 70)

    # Load data
    data = load_data()
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Split into training and test periods
    total_bars = len(data)
    train_end = TRAIN_BARS
    test_end = train_end + TEST_YEARS * 365 * HOURS_PER_DAY

    if test_end > total_bars:
        test_end = total_bars
    if train_end > total_bars:
        print("ERROR: Not enough data for the configured training period.")
        return

    train_data = data.iloc[:train_end].copy()
    full_data = data.iloc[:test_end].copy()

    print(f"\nTraining period: {train_data.index[0]} to {train_data.index[-1]}"
          f" ({len(train_data)} bars)")
    if test_end > train_end:
        print(f"OOS period:      {data.index[train_end]} to "
              f"{data.index[min(test_end, total_bars) - 1]}"
              f" ({test_end - train_end} bars)")

    # ------------------------------------------------------------------
    # STEP 1: In-Sample Excellence
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 1: IN-SAMPLE EXCELLENCE")
    print("=" * 70)

    close_train = train_data["Close"].values
    best_lb, best_obj, best_sig = optimize_donchian(
        close_train, lookback_range=LOOKBACK_RANGE
    )

    is_returns = get_strategy_returns(best_sig, close_train)
    is_equity = np.cumsum(is_returns)
    is_sharpe = sharpe_ratio(is_returns)

    print(f"\nBest lookback:    {best_lb}")
    print(f"Profit factor:    {best_obj:.4f}")
    print(f"Sharpe ratio:     {is_sharpe:.4f}")
    print(f"Total log return: {np.sum(is_returns):.4f}")
    print(f"Equity high:      {np.max(is_equity):.4f}")
    print(f"Equity low:       {np.min(is_equity):.4f}")

    print("\nQuestions to ask yourself:")
    print("  1. Is this excellent? (In-sample results SHOULD be good)")
    print("  2. Is this obviously overfit? (Suspiciously perfect = bad)")

    # ------------------------------------------------------------------
    # STEP 2: In-Sample MCPT
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 2: IN-SAMPLE MONTE-CARLO PERMUTATION TEST")
    print("=" * 70)
    print(f"\nRunning {IS_PERMUTATIONS} permutations...")
    print("(This may take a while. Each permutation requires a full "
          "optimization.)\n")

    is_results = insample_mcpt(
        train_data,
        optimize_func=optimize_on_data,
        objective_func=profit_factor,
        n_permutations=IS_PERMUTATIONS,
        seed=123,
        verbose=True,
    )

    if is_results["p_value"] > 0.01:
        print("\n** The strategy did NOT pass the in-sample MCPT. **")
        print("   Consider improving the strategy or trying a different one.")
        print("   Continuing anyway for demonstration...\n")
    else:
        print("\n** The strategy PASSED the in-sample MCPT. **")
        print("   Proceeding to walk-forward testing.\n")

    # ------------------------------------------------------------------
    # STEP 3: Walk-Forward Test
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  STEP 3: WALK-FORWARD TEST")
    print("=" * 70)

    wf_signal = walkforward_signal(
        full_data,
        optimize_func=optimize_on_data_with_params,
        signal_func=signal_from_params,
        train_bars=TRAIN_BARS,
        train_step=TRAIN_STEP,
    )

    wf_returns = get_strategy_returns(wf_signal, full_data["Close"].values)
    oos_returns = wf_returns[TRAIN_BARS:]
    wf_pf = profit_factor(oos_returns)
    wf_sharpe = sharpe_ratio(oos_returns)
    wf_equity = np.cumsum(oos_returns)

    print(f"\nWalk-Forward OOS Results:")
    print(f"Profit factor:    {wf_pf:.4f}")
    print(f"Sharpe ratio:     {wf_sharpe:.4f}")
    print(f"Total log return: {np.sum(oos_returns):.4f}")

    print("\nQuestion: Is this worth trading?")
    print("(WF results will typically be worse than in-sample, that's normal)")

    # ------------------------------------------------------------------
    # STEP 4: Walk-Forward MCPT
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 4: WALK-FORWARD MONTE-CARLO PERMUTATION TEST")
    print("=" * 70)
    print(f"\nRunning {WF_PERMUTATIONS} permutations...")
    print("(This is the slowest test. Each permutation requires a full "
          "walk-forward.)\n")

    wf_results = walkforward_mcpt(
        full_data,
        optimize_func=optimize_on_data_with_params,
        signal_func=signal_from_params,
        objective_func=profit_factor,
        train_bars=TRAIN_BARS,
        train_step=TRAIN_STEP,
        n_permutations=WF_PERMUTATIONS,
        seed=456,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Step 1 - In-Sample Profit Factor:    {best_obj:.4f}")
    print(f"  Step 2 - In-Sample MCPT p-value:     {is_results['p_value']:.4f}"
          f"  {'PASS' if is_results['p_value'] < 0.01 else 'FAIL'}")
    print(f"  Step 3 - Walk-Forward Profit Factor: {wf_pf:.4f}")
    print(f"  Step 4 - Walk-Forward MCPT p-value:  "
          f"{wf_results['p_value']:.4f}"
          f"  {'PASS' if wf_results['p_value'] < 0.05 else 'FAIL'}")

    both_pass = (is_results["p_value"] < 0.01 and
                 wf_results["p_value"] < 0.05)
    if both_pass:
        print("\n  VERDICT: Strategy passes both tests. Worth considering "
              "for live trading.")
    else:
        print("\n  VERDICT: Strategy did NOT pass both tests. Do NOT trade.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
