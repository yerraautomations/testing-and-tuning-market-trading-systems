"""
Objective functions for evaluating trading strategy performance.

These operate on per-bar strategy returns (not per-trade), following the
approach advocated by Timothy Masters and the YouTuber's video.
Per-bar returns provide more stable and data-rich measurements.
"""

import numpy as np


def profit_factor(returns):
    """
    Compute the profit factor from per-bar strategy returns.

    Profit factor = sum of positive returns / abs(sum of negative returns).
    A profit factor > 1.0 means the strategy is profitable overall.

    Parameters
    ----------
    returns : np.ndarray
        Per-bar strategy returns (strategy position * price change).

    Returns
    -------
    float
        The profit factor. Returns 0.0 if there are no losing bars,
        and 0.0 if there are no winning bars.
    """
    gains = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0].sum())

    if losses == 0:
        return float("inf") if gains > 0 else 1.0
    if gains == 0:
        return 0.0
    return gains / losses


def sharpe_ratio(returns, periods_per_year=252 * 24):
    """
    Compute the annualized Sharpe ratio from per-bar strategy returns.

    Parameters
    ----------
    returns : np.ndarray
        Per-bar strategy returns.
    periods_per_year : int, optional
        Number of bars per year for annualization.
        Default is 252*24 = 6048 (hourly bars, 252 trading days).

    Returns
    -------
    float
        Annualized Sharpe ratio. Returns 0.0 if std is zero.
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)


def total_return(returns):
    """
    Compute the total log return from per-bar strategy returns.

    Parameters
    ----------
    returns : np.ndarray
        Per-bar strategy returns (log returns).

    Returns
    -------
    float
        Total cumulative log return.
    """
    return np.sum(returns)


def get_strategy_returns(signal, prices):
    """
    Compute per-bar strategy returns from a position signal and prices.

    This is the core function that converts any strategy's position signal
    into per-bar returns that can be fed to any objective function.

    The approach:
      1. Compute close-to-close log returns
      2. Shift returns forward by 1 bar (we trade AFTER the signal)
      3. Multiply by the signal (position size/direction)

    Parameters
    ----------
    signal : np.ndarray
        Position signal for each bar. Typical values:
        +1 = long, -1 = short, 0 = flat.
    prices : np.ndarray or pd.Series
        Close prices for each bar.

    Returns
    -------
    np.ndarray
        Per-bar strategy returns, same length as signal.
        First bar return is always 0.
    """
    prices = np.asarray(prices, dtype=float)
    signal = np.asarray(signal, dtype=float)

    # Log returns: log(close[i+1] / close[i])
    log_returns = np.diff(np.log(prices))

    # Shift forward: the return for bar i is the price change from i to i+1
    # Signal at bar i determines position, return is realized at bar i+1
    strategy_returns = np.zeros(len(signal))
    strategy_returns[1:] = signal[:-1] * log_returns

    return strategy_returns
