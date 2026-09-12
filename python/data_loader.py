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

load_ticks() turns a broker tick export into M1 bid bars with the real
per-bar spread (see its docstring); all timestamps end up in UTC.

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


# ---------------------------------------------------------------------------
# Tick data -> M1 bars with the broker's *actual* spread
# ---------------------------------------------------------------------------

BROKER_TZ_ALIASES = {"broker", "eetus", "mt5", "ftmo"}


def broker_time_to_utc(idx, tz="broker"):
    """
    Convert naive timestamps to a UTC DatetimeIndex.

    tz='broker' (aliases: 'eetus', 'mt5', 'ftmo') is the MetaTrader server
    convention used by FTMO and most FX brokers: GMT+2 in winter, GMT+3 in
    summer, switching on the *US* DST dates so that the trading day always
    closes at 17:00 New York. That clock is exactly New York time + 7 hours,
    which is how it is implemented here (DST-safe: the ambiguous / skipped
    hours fall on Sunday morning when the market is closed).

    Any other value is passed to tz_localize (e.g. 'UTC', 'Europe/Prague').
    """
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        return idx.tz_convert("UTC")
    if str(tz).lower() in BROKER_TZ_ALIASES:
        ny = (idx - pd.Timedelta(hours=7)).tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT")
        return ny.tz_convert("UTC")
    return idx.tz_localize(tz).tz_convert("UTC")


def _ticks_to_minute_parts(chunk, dt_format):
    """Aggregate one chunk of ticks into per-minute partial statistics."""
    ts = pd.to_datetime(chunk["DateTime"], format=dt_format)
    minute = ts.dt.floor("min")
    bid = chunk["Bid"].astype("float64").values
    ask = chunk["Ask"].astype("float64").values
    vol = (chunk["Volume"].astype("float64").values if "Volume" in chunk
           else np.zeros(len(chunk)))
    frame = pd.DataFrame({"Bid": bid, "Spread": ask - bid, "Volume": vol},
                         index=minute.values)
    g = frame.groupby(level=0, sort=True)
    part = g.agg(Open=("Bid", "first"), High=("Bid", "max"),
                 Low=("Bid", "min"), Close=("Bid", "last"),
                 SpreadSum=("Spread", "sum"), SpreadMax=("Spread", "max"),
                 SpreadMin=("Spread", "min"), SpreadClose=("Spread", "last"),
                 Ticks=("Bid", "size"), Volume=("Volume", "sum"))
    return part


_PART_MERGE = {"Open": "first", "High": "max", "Low": "min", "Close": "last",
               "SpreadSum": "sum", "SpreadMax": "max", "SpreadMin": "min",
               "SpreadClose": "last", "Ticks": "sum", "Volume": "sum"}


def load_ticks(path, tz="broker", chunksize=5_000_000, cache=None,
               refresh=False, dt_format="%Y%m%d %H:%M:%S.%f", verbose=True):
    """
    Turn a tick export (QuantDataManager / MT5 style: DateTime,Bid,Ask,Volume)
    into M1 **bid** OHLC bars with the spread actually quoted in each bar.

    The file is streamed in chunks, so a 10+ GB export fits in a few GB of
    RAM. A minute that straddles two chunks is merged correctly.

    Parameters
    ----------
    path : str
        The tick CSV. Timestamps 'YYYYMMDD HH:MM:SS.fff' by default.
    tz : str
        Timezone of the timestamps. 'broker' (default) = MT5 server time
        GMT+2/+3 with US DST, which is what FTMO exports use (Friday's last
        tick is at 23:59 server time, Monday's first at 00:00). Use 'UTC'
        if you exported with target timezone UTC. See broker_time_to_utc().
    chunksize : int
        Ticks per chunk. 5M ticks ~ 1 GB RAM peak.
    cache : str, False or None
        CSV path to write the bars to / read them from. Default (None):
        data/<file stem>_M1.csv next to the other datasets (git-ignored).
        Pass cache=False to disable.
    refresh : bool
        Rebuild even if the cache exists.

    Returns
    -------
    pd.DataFrame indexed by UTC bar-open time with columns
        Open, High, Low, Close   bid prices
        Spread                   mean ask-bid over the bar's ticks (price units)
        SpreadMin, SpreadMax     best / worst spread quoted in the bar
        SpreadClose              spread at the bar's last tick (what a
                                 close-of-bar order actually pays)
        Ticks                    number of ticks (MT5 'tick volume')
        Volume                   sum of the exported volume column
    Spread columns are in price units (EURUSD 0.00010 = 1 pip); see
    spread_report() for pip conversion.
    """
    import time as _time

    if cache is None:
        stem = os.path.splitext(os.path.basename(path))[0]
        cache = os.path.join(DATA_DIR, f"{stem}_M1.csv")
    if cache and os.path.exists(cache) and not refresh:
        df = pd.read_csv(cache, index_col="Date", parse_dates=True)
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    parts = []
    n_ticks = 0
    t0 = _time.time()
    reader = pd.read_csv(path, chunksize=chunksize, encoding="utf-8-sig",
                         dtype={"Bid": "float64", "Ask": "float64",
                                "Volume": "float64"})
    for i, chunk in enumerate(reader):
        cols = [str(c).strip().strip("<>") for c in chunk.columns]
        cols = ["DateTime" if c.lower() in ("datetime", "time", "date")
                else c.title() for c in cols]
        chunk.columns = cols
        parts.append(_ticks_to_minute_parts(chunk, dt_format))
        n_ticks += len(chunk)
        if verbose:
            last = parts[-1].index[-1]
            print(f"  chunk {i + 1}: {n_ticks / 1e6:,.0f}M ticks, up to "
                  f"{last:%Y-%m-%d %H:%M} file time, "
                  f"{_time.time() - t0:,.0f}s", flush=True)

    merged = pd.concat(parts)
    bars = merged.groupby(level=0, sort=True).agg(_PART_MERGE)
    bars["Spread"] = bars["SpreadSum"] / bars["Ticks"]
    bars = bars.drop(columns="SpreadSum")
    bars["Ticks"] = bars["Ticks"].astype("int64")
    bars = bars[["Open", "High", "Low", "Close", "Spread", "SpreadMin",
                 "SpreadMax", "SpreadClose", "Ticks", "Volume"]]

    utc = broker_time_to_utc(bars.index, tz)
    bad = utc.isna()
    if bad.any():
        print(f"  WARNING: dropped {int(bad.sum())} bars in ambiguous / "
              f"non-existent DST hours")
        bars = bars[~bad]
        utc = utc[~bad]
    bars.index = utc
    bars.index.name = "Date"
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()

    if verbose:
        print(f"  {n_ticks:,} ticks -> {len(bars):,} M1 bars, "
              f"{bars.index[0]:%Y-%m-%d} .. {bars.index[-1]:%Y-%m-%d} UTC, "
              f"{_time.time() - t0:,.0f}s total")
    if cache:
        bars.to_csv(cache, float_format="%.6f")
        if verbose:
            print(f"  cached to {cache}")
    return bars


def pip_size(df):
    """0.01 for JPY-quoted pairs (price > 20), else 0.0001."""
    return 0.01 if float(df["Close"].median()) > 20 else 0.0001


def spread_report(df, pip=None, print_out=True):
    """
    Summarize the spread actually quoted in tick-derived M1 bars.

    Returns a dict: 'overall' (Series; percentiles are tick-weighted, i.e.
    "what a random tick paid"), 'by_hour' (UTC), 'by_year', 'by_weekday',
    'worst' (the 20 widest bars) and 'pip'. Spreads are reported in pips.
    """
    pip = pip or pip_size(df)
    d = df.copy()
    for c in ("Spread", "SpreadMin", "SpreadMax", "SpreadClose"):
        d[c] = d[c] / pip
    w = d["Ticks"].astype(float).values

    def wq(x, q):
        order = np.argsort(x.values)
        cw = np.cumsum(w[order])
        return float(x.values[order][min(np.searchsorted(cw, q * cw[-1]),
                                         len(cw) - 1)])

    s = d["Spread"]
    overall = pd.Series({
        "bars": len(d),
        "ticks": int(w.sum()),
        "mean (tick-weighted)": float((s * w).sum() / w.sum()),
        "mean (bar-weighted)": float(s.mean()),
        "mean of SpreadClose": float(d["SpreadClose"].mean()),
        "p10": wq(s, 0.10), "p25": wq(s, 0.25), "p50": wq(s, 0.50),
        "p75": wq(s, 0.75), "p90": wq(s, 0.90), "p99": wq(s, 0.99),
        "bars with mean > 1 pip (%)": 100 * float((s > 1).mean()),
        "bars with mean > 2 pips (%)": 100 * float((s > 2).mean()),
        "bars with max > 5 pips (%)": 100 * float((d["SpreadMax"] > 5).mean()),
    })

    def grp(key):
        g = d.groupby(key)
        return pd.DataFrame({
            "mean": g["Spread"].mean(),
            "median": g["Spread"].median(),
            "p90": g["Spread"].quantile(0.9),
            "max": g["SpreadMax"].max(),
            "bars": g.size(),
        })

    by_hour = grp(d.index.hour)
    by_hour.index.name = "hour_utc"
    by_year = grp(d.index.year)
    by_year.index.name = "year"
    by_wd = grp(d.index.dayofweek)
    by_wd.index = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i]
                   for i in by_wd.index]
    worst = d.sort_values("SpreadMax", ascending=False).head(20)[
        ["Close", "Spread", "SpreadMax", "Ticks"]]

    res = {"overall": overall, "by_hour": by_hour, "by_year": by_year,
           "by_weekday": by_wd, "worst": worst, "pip": pip}
    if print_out:
        pd.set_option("display.width", 120)
        print(f"Spread statistics (pips, 1 pip = {pip})")
        print(overall.to_string(float_format=lambda x: f"{x:,.2f}"))
        print("\nBy hour (UTC):")
        print(by_hour.round(2).to_string())
        print("\nBy year:")
        print(by_year.round(2).to_string())
        print("\nBy weekday:")
        print(by_wd.round(2).to_string())
        print("\nWidest bars:")
        print(worst.round(2).to_string())
    return res


def list_datasets():
    """Print available built-in datasets and their cache status."""
    print("Available datasets:")
    for name in _SOURCES:
        cache = os.path.join(DATA_DIR, f"{name}.csv")
        status = "cached" if os.path.exists(cache) else "not downloaded"
        print(f"  {name:6s} [{status}]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "ticks":
        # python data_loader.py ticks <tick.csv> [tz]
        #   -> build (or reuse) the M1 cache and print spread statistics
        tz = sys.argv[3] if len(sys.argv) > 3 else "broker"
        bars = load_ticks(sys.argv[2], tz=tz)
        spread_report(bars)
        sys.exit(0)
    # Fetch and cache all built-in datasets, print a summary.
    for name in _SOURCES:
        try:
            df = load(name, refresh=True)
            print(f"{name}: {len(df)} bars, "
                  f"{df.index[0].date()} -> {df.index[-1].date()}  "
                  f"(cached to data/{name}.csv)")
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:100]}")
