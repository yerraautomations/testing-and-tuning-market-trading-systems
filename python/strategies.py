"""
Example trading strategies for use with the validation framework.

Each strategy function returns a signal array: +1 (long), -1 (short), 0 (flat).
Strategies that need optimization provide an optimize function that returns
the best parameters found via grid search.
"""

import numpy as np
import pandas as pd
from objectives import profit_factor, get_strategy_returns


# ---------------------------------------------------------------------------
# Donchian Channel Breakout
# ---------------------------------------------------------------------------

def donchian_signal(close, lookback):
    """
    Generate Donchian Channel Breakout signals.

    Goes long when the current close is the highest over the lookback period.
    Goes short when the current close is the lowest over the lookback period.
    Otherwise, maintains the previous position.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    lookback : int
        Number of bars to look back for the channel.

    Returns
    -------
    np.ndarray
        Signal array: +1 (long), -1 (short).
    """
    n = len(close)
    signal = np.zeros(n)
    position = 0  # start flat

    for i in range(lookback, n):
        window = close[i - lookback:i + 1]
        highest = np.max(window)
        lowest = np.min(window)

        if close[i] == highest:
            position = 1
        elif close[i] == lowest:
            position = -1

        signal[i] = position

    return signal


def optimize_donchian(close, lookback_range=None, objective_func=None):
    """
    Optimize the Donchian Channel lookback via grid search.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    lookback_range : range or list, optional
        Range of lookback values to test. Default is range(5, 200).
    objective_func : callable, optional
        Objective function. Default is profit_factor.

    Returns
    -------
    best_lookback : int
        The lookback that produced the best objective value.
    best_objective : float
        The best objective value found.
    best_signal : np.ndarray
        The signal array for the best lookback.
    """
    if lookback_range is None:
        lookback_range = range(5, 200)
    if objective_func is None:
        objective_func = profit_factor

    best_obj = -np.inf
    best_lb = None
    best_sig = None

    for lb in lookback_range:
        if lb >= len(close) - 1:
            continue
        sig = donchian_signal(close, lb)
        rets = get_strategy_returns(sig, close)
        obj = objective_func(rets)
        if obj > best_obj:
            best_obj = obj
            best_lb = lb
            best_sig = sig

    return best_lb, best_obj, best_sig


# ---------------------------------------------------------------------------
# Moving Average Crossover
# ---------------------------------------------------------------------------

def ma_crossover_signal(close, fast_period, slow_period):
    """
    Generate Moving Average Crossover signals.

    Goes long when the fast MA is above the slow MA.
    Goes short when the fast MA is below the slow MA.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    fast_period : int
        Period for the fast moving average.
    slow_period : int
        Period for the slow moving average.

    Returns
    -------
    np.ndarray
        Signal array: +1 (long), -1 (short), 0 (neutral).
    """
    n = len(close)
    signal = np.zeros(n)

    if slow_period >= n:
        return signal

    # Compute simple moving averages
    fast_ma = pd.Series(close).rolling(fast_period).mean().values
    slow_ma = pd.Series(close).rolling(slow_period).mean().values

    for i in range(slow_period - 1, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
            continue
        if fast_ma[i] > slow_ma[i]:
            signal[i] = 1
        elif fast_ma[i] < slow_ma[i]:
            signal[i] = -1

    return signal


def optimize_ma_crossover(close, max_slow=100, objective_func=None):
    """
    Optimize MA Crossover parameters via grid search.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    max_slow : int, optional
        Maximum slow MA period. Default is 100.
    objective_func : callable, optional
        Objective function. Default is profit_factor.

    Returns
    -------
    best_params : tuple
        (fast_period, slow_period) that produced the best objective.
    best_objective : float
        The best objective value found.
    best_signal : np.ndarray
        The signal for the best parameters.
    """
    if objective_func is None:
        objective_func = profit_factor

    best_obj = -np.inf
    best_params = None
    best_sig = None

    for slow in range(2, max_slow + 1):
        for fast in range(1, slow):
            sig = ma_crossover_signal(close, fast, slow)
            rets = get_strategy_returns(sig, close)
            obj = objective_func(rets)
            if obj > best_obj:
                best_obj = obj
                best_params = (fast, slow)
                best_sig = sig

    return best_params, best_obj, best_sig
