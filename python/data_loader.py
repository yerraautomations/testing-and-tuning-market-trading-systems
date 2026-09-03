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


def load_mt5(path, tz="UTC"):
    """
    Load an MT5-style bar export (MetaTrader 5 "Export bars" or
    QuantDataManager "Export to MT5 data").

    Handles tab / comma / semicolon delimiters; headers written as
    <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD> or
    without angle brackets or with no header at all; a combined DateTime
    column; optional volume and spread columns.

    Parameters
    ----------
    path : str
    tz : str
        Timezone the timestamps were exported in (we recommend exporting in
        UTC). The result is always indexed in UTC.

    Returns
    -------
    pd.DataFrame indexed by UTC datetime with Open, High, Low, Close and,
    when present, Volume and Spread (spread in points, as exported).
    """
    import csv

    with open(path, "r", encoding="utf-8-sig") as f:
        sample = f.read(4096)
    try:
        sep = csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        sep = "\t" if "\t" in sample else ","
    first_line = sample.splitlines()[0]
    has_header = any(ch.isalpha() for ch in first_line)

    df = pd.read_csv(path, sep=sep, header=0 if has_header else None,
                     encoding="utf-8-sig")
    if has_header:
        cols = [str(c).strip().strip("<>").strip().lower() for c in df.columns]
    else:
        layouts = {
            9: ["date", "time", "open", "high", "low", "close",
                "tickvol", "vol", "spread"],
            8: ["date", "time", "open", "high", "low", "close",
                "tickvol", "vol"],
            7: ["date", "time", "open", "high", "low", "close", "tickvol"],
            6: ["date", "time", "open", "high", "low", "close"],
            5: ["datetime", "open", "high", "low", "close"],
        }
        if df.shape[1] not in layouts:
            raise ValueError(f"Unrecognized headerless layout with "
                             f"{df.shape[1]} columns in {path}")
        cols = layouts[df.shape[1]]
    df.columns = cols

    if "date" in cols and "time" in cols:
        ts = df["date"].astype(str) + " " + df["time"].astype(str)
    elif "datetime" in cols:
        ts = df["datetime"].astype(str)
    elif "date" in cols:
        ts = df["date"].astype(str)
    else:
        ts = df.iloc[:, 0].astype(str)

    idx = None
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y.%m.%d", "%Y-%m-%d"):
        try:
            idx = pd.to_datetime(ts, format=fmt)
            break
        except (ValueError, TypeError):
            continue
    if idx is None:
        idx = pd.to_datetime(ts)
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    idx = idx.tz_convert("UTC")

    out = pd.DataFrame({
        "Open": df["open"].astype(float).values,
        "High": df["high"].astype(float).values,
        "Low": df["low"].astype(float).values,
        "Close": df["close"].astype(float).values,
    }, index=idx)
    for vcol in ("volume", "vol", "tickvol"):
        if vcol in cols:
            out["Volume"] = df[vcol].astype(float).values
            break
    if "spread" in cols:
        out["Spread"] = df["spread"].astype(float).values
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


_OHLC_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}


def resample_ohlc(df, rule="1h", offset=None):
    """
    Resample M1 (or any) bars to a coarser timeframe: '5min', '15min', '1h',
    '4h', '1D'. `offset` shifts the bin boundaries (e.g. offset='22h' with
    rule='1D' gives daily bars that roll at 22:00 UTC).
    """
    agg = dict(_OHLC_AGG)
    if "Volume" in df:
        agg["Volume"] = "sum"
    if "Spread" in df:
        agg["Spread"] = "mean"
    # pandas ignores `offset` for calendar-day rules; express days in hours.
    if offset is not None and rule.strip().upper().endswith("D"):
        n_days = int(rule.strip()[:-1] or 1)
        rule = f"{24 * n_days}h"
    out = df.resample(rule, offset=offset, label="left", closed="left").agg(agg)
    return out.dropna(subset=["Open"])


def resample_daily_session(df, tz="America/New_York", session_start_hour=17):
    """
    Daily bars that roll at a wall-clock hour in a given timezone, DST-aware.
    The default (17:00 New York) is the FX-market "trading day" convention
    that brokers on GMT+2/+3 server time reproduce. The index is the
    session's opening timestamp in `tz`.
    """
    local = df.tz_convert(tz)
    shifted = local.copy()
    shifted.index = shifted.index - pd.Timedelta(hours=session_start_hour)
    agg = dict(_OHLC_AGG)
    if "Volume" in df:
        agg["Volume"] = "sum"
    if "Spread" in df:
        agg["Spread"] = "mean"
    daily = shifted.resample("1D").agg(agg).dropna(subset=["Open"])
    daily.index = daily.index + pd.Timedelta(hours=session_start_hour)
    daily.index.name = "Date"
    return daily


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
