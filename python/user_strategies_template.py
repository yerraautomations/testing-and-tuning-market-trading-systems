"""
TEMPLATE: plug YOUR strategies into the validation platform.

How to use
----------
1. Copy this file to  user_strategies.py  (same folder). That filename is
   picked up automatically by validate.py / crossmarket.py / mcpt_multi.py.
2. For each strategy write two module-level functions (see the contract
   below) and call register_strategy() at the bottom.
3. Run, for example:
       python validate.py BTC my_breakout
       python crossmarket.py my_breakout BTC AAPL
       python propfirm.py            (edit the demo block, or import propfirm)

The contract
------------
signal(close, params) -> np.ndarray
    One position per bar, same length as `close`:
        +1 long, -1 short, 0 flat  (fractional sizes are allowed too).
    NO LOOK-AHEAD: the value at bar t may use only close[:t+1]. The framework
    realizes the return of position[t] on bar t+1 automatically.
    If your strategy needs more than close (e.g. High/Low, volume), keep the
    same signature and load the extra series inside the function from a
    module-level DataFrame you set beforehand -- or ask for a wider contract.

optimize(close, objective_func=None) -> (best_params, best_objective, best_signal)
    Grid-search (or any search) over your parameter space, scoring each
    candidate with objective_func on the per-bar strategy returns. Use the
    helper _best_over() for plain grid searches. Default objective is profit
    factor -- keep it so that results are comparable across strategies.

Anything you compute here is re-run on every permutation and every
walk-forward window, so keep it vectorized (pandas rolling / numpy) rather
than Python loops over bars.
"""

import numpy as np
import pandas as pd

from strategies_library import register_strategy, _carry_forward, _best_over


# ---------------------------------------------------------------------------
# Example: N-bar breakout with a buffer, exits on the opposite breakout.
# Replace with your own logic.
# ---------------------------------------------------------------------------

def my_breakout_signal(close, params):
    lookback, buffer = int(params[0]), float(params[1])
    close = np.asarray(close, dtype=float)
    s = pd.Series(close)
    # Prior N-bar high/low (shifted by one so the current bar is not included)
    hi = s.rolling(lookback).max().shift(1).values
    lo = s.rolling(lookback).min().shift(1).values
    pos = np.full(len(close), np.nan)
    with np.errstate(invalid="ignore"):
        pos[close < lo * (1.0 - buffer)] = -1.0
        pos[close > hi * (1.0 + buffer)] = 1.0
    return _carry_forward(pos)          # hold until the opposite breakout


def optimize_my_breakout(close, objective_func=None):
    grid = [(lb, b) for lb in range(10, 130, 10) for b in (0.0, 0.002, 0.005)]
    return _best_over(close, grid, my_breakout_signal, objective_func)


register_strategy(
    "my_breakout",
    signal=my_breakout_signal,
    optimize=optimize_my_breakout,
    kind="trend-following",
    params="(lookback, buffer)",
)
