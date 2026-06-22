"""
Vectorized (fast) versions of the strategy functions.

These are drop-in replacements for the functions in strategies.py that
produce BYTE-FOR-BYTE IDENTICAL results, but run much faster by replacing
pure-Python loops with C-level pandas/numpy operations.

The originals in strategies.py are left untouched so you can always fall
back to them. See verify_fast.py for the equivalence proof.
"""

import numpy as np
import pandas as pd

from objectives import profit_factor, get_strategy_returns


def donchian_signal_fast(close, lookback):
    """
    Vectorized Donchian Channel Breakout signal.

    Identical output to strategies.donchian_signal, but uses pandas rolling
    max/min (C-level) instead of a Python loop, removing the O(lookback)
    inner cost.

    Logic (matches the original exactly):
      - Window is the last (lookback + 1) bars, including the current bar.
      - If close == window max -> go long (+1).
      - Elif close == window min -> go short (-1).
      - Else -> carry the previous position forward.
      - Long takes precedence on ties (matches the original if/elif order).
      - Bars before index `lookback` are 0 (flat), as in the original.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    s = pd.Series(close)

    # Rolling extremes over a window of (lookback + 1) bars.
    # min_periods defaults to the window size, so the first `lookback`
    # entries are NaN -- exactly the bars the original leaves flat.
    roll_max = s.rolling(lookback + 1).max().values
    roll_min = s.rolling(lookback + 1).min().values

    pos = np.full(n, np.nan)
    # Apply short first, then long, so long overwrites on ties.
    # (close >= roll_max) is True only when close == roll_max, since the
    # window includes the current bar; same for the min side.
    with np.errstate(invalid="ignore"):
        is_short = close <= roll_min
        is_long = close >= roll_max
    pos[is_short] = -1.0
    pos[is_long] = 1.0

    # Carry the position forward between breakouts; leading NaN -> flat (0).
    signal = pd.Series(pos).ffill().fillna(0.0).values
    return signal


def optimize_donchian_fast(close, lookback_range=None, objective_func=None):
    """
    Grid-search the Donchian lookback using the vectorized signal.

    Identical selection to strategies.optimize_donchian (same objective
    values, same tie-breaking: the first lookback achieving the max wins).
    """
    if lookback_range is None:
        lookback_range = range(5, 200)
    if objective_func is None:
        objective_func = profit_factor

    close = np.asarray(close, dtype=float)
    best_obj = -np.inf
    best_lb = None
    best_sig = None

    for lb in lookback_range:
        if lb >= len(close) - 1:
            continue
        sig = donchian_signal_fast(close, lb)
        rets = get_strategy_returns(sig, close)
        obj = objective_func(rets)
        if obj > best_obj:
            best_obj = obj
            best_lb = lb
            best_sig = sig

    return best_lb, best_obj, best_sig
