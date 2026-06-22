"""
Real market data loader.

Fetches real OHLC price data from public GitHub-hosted datasets and caches
it locally as CSV under python/data/. Once cached, no network is needed.

Why GitHub and not an exchange API? In many locked-down environments the
direct exchange endpoints (Binance/Coinbase/Kraken/Yahoo) are firewalled,
but raw.githubusercontent.com is reachable. On your own machine you can add
your own sources (e.g. ccxt, yfinance) -- see fetch_custom() at the bottom.

All loaders return a DataFrame indexed by date with columns:
    Open, High, Low, Close

For close-only sources (e.g. the CoinMetrics BTC series), Open=High=Low=Close.
This is intentional and correct: the bar-permutation then reduces to a
permutation of close-to-close returns, which is exactly the single-series
permutation test from Masters' book.
"""

import os
import io
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _download(url, timeout=20):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def _fetch_aapl():
    """Apple daily OHLC, 2015-2017 (~506 bars). True OHLC bars."""
    url = ("https://raw.githubusercontent.com/plotly/datasets/master/"
           "finance-charts-apple.csv")
    df = pd.read_csv(io.StringIO(_download(url)))
    out = pd.DataFrame({
        "Open": df["AAPL.Open"].values,
        "High": df["AAPL.High"].values,
        "Low": df["AAPL.Low"].values,
        "Close": df["AAPL.Close"].values,
    }, index=pd.to_datetime(df["Date"]))
    out.index.name = "Date"
    return out


def _fetch_btc():
    """Bitcoin daily close, 2010-present (~5800 bars). Close-only."""
    url = ("https://raw.githubusercontent.com/coinmetrics/data/master/"
           "csv/btc.csv")
    df = pd.read_csv(io.StringIO(_download(url)))
    df = df[["time", "PriceUSD"]].dropna()
    close = df["PriceUSD"].values.astype(float)
    out = pd.DataFrame({
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
    }, index=pd.to_datetime(df["time"]))
    out.index.name = "Date"
    return out


_SOURCES = {
    "AAPL": _fetch_aapl,
    "BTC": _fetch_btc,
}


def load(name, refresh=False):
    """
    Load a named dataset, using the local cache if available.

    Parameters
    ----------
    name : str
        One of: 'AAPL', 'BTC'.
    refresh : bool, optional
        If True, re-download even if a cache exists.

    Returns
    -------
    pd.DataFrame
        Indexed by date, columns: Open, High, Low, Close.
    """
    if name not in _SOURCES:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Available: {list(_SOURCES)}")

    cache = os.path.join(DATA_DIR, f"{name}.csv")
    if os.path.exists(cache) and not refresh:
        df = pd.read_csv(cache, index_col="Date", parse_dates=True)
        return df

    df = _SOURCES[name]()
    df.to_csv(cache)
    return df


def load_csv(path):
    """
    Load OHLC data from an arbitrary CSV on disk.

    The CSV must have a date column and Open/High/Low/Close columns
    (case-insensitive, common prefixes like 'AAPL.Open' are handled).

    Returns a DataFrame indexed by date with Open, High, Low, Close.
    """
    df = pd.read_csv(path)
    col_map = {}
    date_col = None
    for col in df.columns:
        lower = col.lower()
        if "date" in lower or "time" in lower:
            date_col = col
        elif lower.endswith("open") or lower == "o":
            col_map[col] = "Open"
        elif lower.endswith("high") or lower == "h":
            col_map[col] = "High"
        elif lower.endswith("low") or lower == "l":
            col_map[col] = "Low"
        elif lower.endswith("close") or lower == "c":
            col_map[col] = "Close"
    df = df.rename(columns=col_map)
    if date_col:
        df.index = pd.to_datetime(df[date_col])
        df.index.name = "Date"
    return df[["Open", "High", "Low", "Close"]]


def list_datasets():
    """Print available built-in datasets and their cache status."""
    print("Available datasets:")
    for name in _SOURCES:
        cache = os.path.join(DATA_DIR, f"{name}.csv")
        status = "cached" if os.path.exists(cache) else "not downloaded"
        print(f"  {name:6s} [{status}]")


if __name__ == "__main__":
    # Fetch and cache all built-in datasets, print a summary.
    for name in _SOURCES:
        try:
            df = load(name, refresh=True)
            print(f"{name}: {len(df)} bars, "
                  f"{df.index[0].date()} -> {df.index[-1].date()}  "
                  f"(cached to data/{name}.csv)")
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:100]}")
