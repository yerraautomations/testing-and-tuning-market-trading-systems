"""
Bar permutation algorithm for Monte-Carlo Permutation Tests.

Generates permutations of OHLC price data that preserve statistical properties
(mean, std, skew, kurtosis of returns) while destroying temporal patterns.

Based on the methods from "Testing and Tuning Market Trading Systems"
by Timothy Masters, adapted from the C++ MCPT_BARS implementation
and the approach described by the YouTuber's walkthrough.
"""

import numpy as np
import pandas as pd


def get_permutation(data, start_index=0, seed=None):
    """
    Generate a permutation of OHLC bar data.

    The permutation preserves:
      - The first bar (before start_index, all data is unchanged)
      - Statistical properties of returns (mean, std, skew, kurtosis)
      - Cross-market correlation (when a list of DataFrames is passed)

    The permutation destroys:
      - Temporal ordering / serial correlation
      - Volatility clustering
      - Long memory

    Parameters
    ----------
    data : pd.DataFrame or list of pd.DataFrame
        OHLC data with columns: Open, High, Low, Close.
        If a list, all DataFrames must share the same index (for
        multi-market permutation that preserves cross-correlation).
    start_index : int, optional
        Index position where the permutation begins. Data before this
        index is left unchanged. Default is 0 (permute everything).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame or list of pd.DataFrame
        Permuted OHLC data in the same format as input.
    """
    rng = np.random.default_rng(seed)

    # Handle single DataFrame vs list of DataFrames
    single_market = False
    if isinstance(data, pd.DataFrame):
        single_market = True
        data = [data]

    n_markets = len(data)

    # Validate that all markets share the same index for multi-market case
    if n_markets > 1:
        for i in range(1, n_markets):
            if not data[0].index.equals(data[i].index):
                raise ValueError("All DataFrames must share the same index "
                                 "for multi-market permutation.")

    n_bars = len(data[0])
    n_permutable = n_bars - start_index

    if n_permutable < 2:
        if single_market:
            return data[0].copy()
        return [df.copy() for df in data]

    # Compute relative prices for each market
    # rel_intrabar[m][i] = (high - open, low - open, close - open) for bar i
    # rel_open[m][i] = open[i] - close[i-1]  (the gap)
    all_rel_intrabar = []
    all_rel_open = []
    all_first_bar = []

    for m in range(n_markets):
        df = data[m]
        log_open = np.log(df["Open"].values.astype(float))
        log_high = np.log(df["High"].values.astype(float))
        log_low = np.log(df["Low"].values.astype(float))
        log_close = np.log(df["Close"].values.astype(float))

        # Number of bars we will permute
        s = start_index

        # First bar of permutation zone
        first_bar = np.array([
            log_open[s],
            log_high[s] - log_open[s],
            log_low[s] - log_open[s],
            log_close[s] - log_open[s],
        ])
        all_first_bar.append(first_bar)

        # Relative intrabar: (high - open, low - open, close - open)
        rel_intrabar = np.column_stack([
            log_high[s + 1:] - log_open[s + 1:],
            log_low[s + 1:] - log_open[s + 1:],
            log_close[s + 1:] - log_open[s + 1:],
        ])
        all_rel_intrabar.append(rel_intrabar)

        # Relative open (gap): open[i] - close[i-1]
        rel_open = log_open[s + 1:] - log_close[s:s + (n_bars - s - 1)]
        all_rel_open.append(rel_open)

    # Generate shared shuffle indices (preserves cross-market correlation)
    n_shuffle = n_permutable - 1  # number of bars after the first

    if n_shuffle < 1:
        if single_market:
            return data[0].copy()
        return [df.copy() for df in data]

    # Shuffle 1: intrabar quantities (high, low, close relative to open)
    intrabar_indices = np.arange(n_shuffle)
    rng.shuffle(intrabar_indices)

    # Shuffle 2: gaps (open relative to prior close)
    gap_indices = np.arange(n_shuffle)
    rng.shuffle(gap_indices)

    # Reconstruct permuted prices for each market
    result = []
    for m in range(n_markets):
        df = data[m]
        perm = df.copy()

        log_open = np.log(df["Open"].values.astype(float))
        log_close = np.log(df["Close"].values.astype(float))

        s = start_index

        # Allocate arrays for permuted log prices
        perm_open = np.zeros(n_bars)
        perm_high = np.zeros(n_bars)
        perm_low = np.zeros(n_bars)
        perm_close = np.zeros(n_bars)

        # Copy unchanged portion before start_index
        perm_open[:s] = log_open[:s]
        perm_high[:s] = np.log(df["High"].values[:s].astype(float))
        perm_low[:s] = np.log(df["Low"].values[:s].astype(float))
        perm_close[:s] = log_close[:s]

        # Set the first bar of the permutation zone
        perm_open[s] = all_first_bar[m][0]
        perm_high[s] = all_first_bar[m][0] + all_first_bar[m][1]
        perm_low[s] = all_first_bar[m][0] + all_first_bar[m][2]
        perm_close[s] = all_first_bar[m][0] + all_first_bar[m][3]

        # Build remaining bars using shuffled relative prices
        shuffled_intrabar = all_rel_intrabar[m][intrabar_indices]
        shuffled_gaps = all_rel_open[m][gap_indices]

        for i in range(n_shuffle):
            bar_idx = s + 1 + i
            perm_open[bar_idx] = perm_close[bar_idx - 1] + shuffled_gaps[i]
            perm_high[bar_idx] = perm_open[bar_idx] + shuffled_intrabar[i, 0]
            perm_low[bar_idx] = perm_open[bar_idx] + shuffled_intrabar[i, 1]
            perm_close[bar_idx] = perm_open[bar_idx] + shuffled_intrabar[i, 2]

        # Convert back from log prices
        perm["Open"] = np.exp(perm_open)
        perm["High"] = np.exp(perm_high)
        perm["Low"] = np.exp(perm_low)
        perm["Close"] = np.exp(perm_close)

        result.append(perm)

    if single_market:
        return result[0]
    return result
