"""
A small library of vectorized trading strategies.

Each strategy exposes two module-level functions (so they are picklable for
the parallel MCPT process pool):

    <name>_signal(close, params)          -> np.ndarray of {+1, -1, 0}
    optimize_<name>(close, objective)     -> (best_params, best_obj, best_sig)

All signals use only information available up to and including the current
bar (no look-ahead). Returns are realized on the NEXT bar via
objectives.get_strategy_returns, which shifts the signal forward by one.

A STRATEGIES registry at the bottom lets other scripts iterate over all of
them uniformly.
"""

import numpy as np
import pandas as pd

from objectives import profit_factor, get_strategy_returns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _carry_forward(pos):
    """Forward-fill a position array (NaN = no new signal); leading -> 0."""
    return pd.Series(pos).ffill().fillna(0.0).values


def _best_over(close, param_grid, signal_fn, objective_func):
    """Generic grid search returning (best_params, best_obj, best_sig)."""
    if objective_func is None:
        objective_func = profit_factor
    best_obj = -np.inf
    best_params = None
    best_sig = None
    for params in param_grid:
        sig = signal_fn(close, params)
        rets = get_strategy_returns(sig, close)
        obj = objective_func(rets)
        if obj > best_obj:
            best_obj = obj
            best_params = params
            best_sig = sig
    return best_params, best_obj, best_sig


# ---------------------------------------------------------------------------
# 1. Donchian Channel Breakout (trend-following)
# ---------------------------------------------------------------------------

def donchian_signal(close, params):
    """Long on N-bar high breakout, short on N-bar low breakout."""
    lookback = int(params)
    close = np.asarray(close, dtype=float)
    s = pd.Series(close)
    roll_max = s.rolling(lookback + 1).max().values
    roll_min = s.rolling(lookback + 1).min().values
    pos = np.full(len(close), np.nan)
    with np.errstate(invalid="ignore"):
        pos[close <= roll_min] = -1.0
        pos[close >= roll_max] = 1.0
    return _carry_forward(pos)


def optimize_donchian(close, objective_func=None):
    grid = range(10, 205, 5)
    return _best_over(close, grid, donchian_signal, objective_func)


# ---------------------------------------------------------------------------
# 2. Moving Average Crossover (trend-following)
# ---------------------------------------------------------------------------

def ma_cross_signal(close, params):
    """Long when fast MA > slow MA, short when fast MA < slow MA."""
    fast, slow = int(params[0]), int(params[1])
    close = np.asarray(close, dtype=float)
    s = pd.Series(close)
    fast_ma = s.rolling(fast).mean().values
    slow_ma = s.rolling(slow).mean().values
    pos = np.full(len(close), np.nan)
    with np.errstate(invalid="ignore"):
        pos[fast_ma > slow_ma] = 1.0
        pos[fast_ma < slow_ma] = -1.0
    return _carry_forward(pos)


def optimize_ma_cross(close, objective_func=None):
    grid = [(f, s) for s in range(20, 210, 10)
            for f in range(5, 55, 5) if f < s]
    return _best_over(close, grid, ma_cross_signal, objective_func)


# ---------------------------------------------------------------------------
# 3. RSI Mean-Reversion (counter-trend)
# ---------------------------------------------------------------------------

def _rsi(close, period):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(period).mean().values
    avg_loss = pd.Series(loss).rolling(period).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[avg_loss == 0] = 100.0
    return rsi


def rsi_signal(close, params):
    """Long when oversold (RSI<lower), short when overbought (RSI>upper)."""
    period, lower = int(params[0]), float(params[1])
    upper = 100.0 - lower
    close = np.asarray(close, dtype=float)
    rsi = _rsi(close, period)
    pos = np.full(len(close), np.nan)
    with np.errstate(invalid="ignore"):
        pos[rsi < lower] = 1.0
        pos[rsi > upper] = -1.0
    return _carry_forward(pos)


def optimize_rsi(close, objective_func=None):
    grid = [(p, lo) for p in range(5, 30, 2) for lo in (10, 20, 30)]
    return _best_over(close, grid, rsi_signal, objective_func)


# ---------------------------------------------------------------------------
# 4. Bollinger Band Mean-Reversion (counter-trend)
# ---------------------------------------------------------------------------

def bollinger_signal(close, params):
    """Long when price closes below lower band, short above upper band."""
    n, k = int(params[0]), float(params[1])
    close = np.asarray(close, dtype=float)
    s = pd.Series(close)
    mid = s.rolling(n).mean().values
    sd = s.rolling(n).std().values
    upper = mid + k * sd
    lower = mid - k * sd
    pos = np.full(len(close), np.nan)
    with np.errstate(invalid="ignore"):
        pos[close < lower] = 1.0
        pos[close > upper] = -1.0
    return _carry_forward(pos)


def optimize_bollinger(close, objective_func=None):
    grid = [(n, k) for n in range(10, 60, 5) for k in (1.5, 2.0, 2.5)]
    return _best_over(close, grid, bollinger_signal, objective_func)


# ---------------------------------------------------------------------------
# 5. Momentum / Rate-of-Change (trend-following)
# ---------------------------------------------------------------------------

def momentum_signal(close, params):
    """Long when N-bar return is positive, short when negative."""
    n = int(params)
    close = np.asarray(close, dtype=float)
    roc = np.full(len(close), np.nan)
    roc[n:] = close[n:] / close[:-n] - 1.0
    pos = np.full(len(close), np.nan)
    with np.errstate(invalid="ignore"):
        pos[roc > 0] = 1.0
        pos[roc < 0] = -1.0
    return _carry_forward(pos)


def optimize_momentum(close, objective_func=None):
    grid = range(5, 125, 5)
    return _best_over(close, grid, momentum_signal, objective_func)


# ---------------------------------------------------------------------------
# Adapters for the MCPT framework (operate on DataFrames).
#
# These are module-level so functools.partial bindings of them remain
# picklable for the parallel process pool on BOTH fork and spawn platforms.
# ---------------------------------------------------------------------------

def opt_for_insample(data, optimize):
    """Adapt a strategy optimizer to the in-sample MCPT interface."""
    _, obj, sig = optimize(data["Close"].values)
    return obj, sig


def opt_for_walkforward(data, optimize):
    """Adapt a strategy optimizer to the walk-forward interface."""
    params, obj, sig = optimize(data["Close"].values)
    return obj, sig, params


def signal_for_walkforward(data, params, signal):
    """Adapt a strategy signal fn to the walk-forward interface."""
    return signal(data["Close"].values, params)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "donchian": {
        "signal": donchian_signal,
        "optimize": optimize_donchian,
        "kind": "trend-following",
        "params": "lookback",
    },
    "ma_cross": {
        "signal": ma_cross_signal,
        "optimize": optimize_ma_cross,
        "kind": "trend-following",
        "params": "(fast, slow)",
    },
    "rsi": {
        "signal": rsi_signal,
        "optimize": optimize_rsi,
        "kind": "mean-reversion",
        "params": "(period, lower_threshold)",
    },
    "bollinger": {
        "signal": bollinger_signal,
        "optimize": optimize_bollinger,
        "kind": "mean-reversion",
        "params": "(window, num_std)",
    },
    "momentum": {
        "signal": momentum_signal,
        "optimize": optimize_momentum,
        "kind": "trend-following",
        "params": "lookback",
    },
}


def register_strategy(name, signal, optimize, kind="custom", params=""):
    """
    Add a strategy to the registry so validate.py, crossmarket.py,
    mcpt_multi.py and propfirm.py can use it by name.

    signal(close, params) -> np.ndarray of positions, same length as close
    optimize(close, objective_func=None) -> (best_params, best_obj, best_sig)

    Both must be module-level functions (picklable for the process pool)
    and must not look ahead: the position at bar t may only use data up to
    and including bar t.
    """
    STRATEGIES[name] = {"signal": signal, "optimize": optimize,
                        "kind": kind, "params": params}


def load_user_strategies():
    """
    Import user_strategies.py (if present next to this file) so that the
    register_strategy() calls in it populate the registry. See
    user_strategies_template.py.
    """
    try:
        import user_strategies  # noqa: F401  (registers on import)
    except ImportError:
        pass


if __name__ == "__main__":
    # Quick self-check: run each strategy's optimizer on real BTC data.
    from data_loader import load
    df = load("BTC")
    close = df["Close"].values
    print(f"Self-check on BTC ({len(close)} bars):\n")
    print(f"{'strategy':12s} {'kind':16s} {'best params':22s} {'profit factor'}")
    print("-" * 70)
    for name, spec in STRATEGIES.items():
        params, obj, _ = spec["optimize"](close)
        print(f"{name:12s} {spec['kind']:16s} {str(params):22s} {obj:.4f}")
