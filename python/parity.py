"""
Parity check: does the Python port of a strategy trade like the MT5 EA?

The EA cannot be tested by the honesty machine, so the port must be proven
faithful. Run the EA in the MT5 Strategy Tester on a period, export the
report (HTML) and the bars of that symbol/period (Symbols -> Bars -> Export),
then:

    python parity.py --report ReportTester-123.html --bars EURUSD_H1.csv \
        --strategy my_breakout --params "[40, 0.002]" --tz broker

It aligns the EA's entry deals with the Python signal's entries (same side,
within --tolerance bars), reports the match rate, lists mismatches, and
correlates trade-by-trade P/L of the matched pairs. A faithful port shows
match rates near 100% and P/L correlation above 0.99; anything else lists
exactly which trades differ so the port can be fixed.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OPT_DIR = os.path.join(os.path.dirname(HERE), "portfolio_optimizer")
if OPT_DIR not in sys.path:
    sys.path.insert(0, OPT_DIR)

from data_loader import load_mt5, broker_time_to_utc, resample_ohlc
from strategies_library import STRATEGIES, load_user_strategies
from to_optimizer import signal_to_strategy
from mt5_parser import MT5Parser


def _deals(strategy):
    """DataFrame of deals with naive times (broker time for MT5 reports)."""
    rows = [{"time": t.time, "direction": t.direction.lower(), "side": t.trade_type.lower(),
             "price": t.price, "profit": t.profit} for t in strategy.trades]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df


def compare(mt5_strategy, py_strategy, bar_seconds, tolerance_bars=1):
    """
    Align entries ("in" deals) by side and time. Returns a dict with match
    statistics, the unmatched deals of each side, and P/L correlation over
    matched entry->exit pairs.
    """
    a = _deals(mt5_strategy)
    b = _deals(py_strategy)
    tol = pd.Timedelta(seconds=bar_seconds * tolerance_bars)
    a_in = a[a.direction.isin(["in", "in/out"])].reset_index(drop=True)
    b_in = b[b.direction.isin(["in", "in/out"])].reset_index(drop=True)
    a_out = a[a.direction.isin(["out", "in/out"])].reset_index(drop=True)
    b_out = b[b.direction.isin(["out", "in/out"])].reset_index(drop=True)

    used = set()
    pairs = []
    for i, ra in a_in.iterrows():
        cand = b_in[(b_in.side == ra.side) & ((b_in.time - ra.time).abs() <= tol)]
        cand = cand[~cand.index.isin(used)]
        if len(cand):
            j = (cand.time - ra.time).abs().idxmin()
            used.add(j)
            pairs.append((i, j))
    matched_a = {i for i, _ in pairs}
    matched_b = {j for _, j in pairs}

    # P/L per entry: sum of the following out-deal profits until next entry
    def pnl_after_entries(ins, outs):
        out = []
        for k, r in ins.iterrows():
            nxt = ins.time.iloc[k + 1] if k + 1 < len(ins) else pd.Timestamp.max
            out.append(outs[(outs.time >= r.time) & (outs.time < nxt)].profit.sum())
        return np.array(out)
    pa = pnl_after_entries(a_in, a_out) if len(a_in) else np.array([])
    pb = pnl_after_entries(b_in, b_out) if len(b_in) else np.array([])
    corr = None
    if len(pairs) > 2:
        x = np.array([pa[i] for i, _ in pairs]); y = np.array([pb[j] for _, j in pairs])
        if x.std() > 0 and y.std() > 0:
            corr = float(np.corrcoef(x, y)[0, 1])

    return {
        "mt5_entries": int(len(a_in)), "py_entries": int(len(b_in)),
        "matched": len(pairs),
        "match_rate_mt5": len(pairs) / len(a_in) if len(a_in) else 0.0,
        "match_rate_py": len(pairs) / len(b_in) if len(b_in) else 0.0,
        "pnl_correlation": corr,
        "mt5_net": float(a_out.profit.sum()) if len(a_out) else 0.0,
        "py_net": float(b_out.profit.sum()) if len(b_out) else 0.0,
        "unmatched_mt5": a_in[~a_in.index.isin(matched_a)][["time", "side", "price"]].head(25),
        "unmatched_py": b_in[~b_in.index.isin(matched_b)][["time", "side", "price"]].head(25),
    }


def run(report, bars, strategy, params, tz="broker", tolerance_bars=1,
        notional=100_000.0, spread_frac=0.0, verbose=True):
    load_user_strategies()
    spec = STRATEGIES[strategy]
    df = load_mt5(bars, tz="UTC")
    naive = df.index.tz_localize(None)
    if tz == "broker":
        df.index = broker_time_to_utc(naive, "broker")
        df = df[~df.index.isna()]
    close = df["Close"].values
    signal = spec["signal"](close, params)
    # Deal times must be comparable with the MT5 report (broker time, naive):
    # rebuild the naive broker-time index for the Python deals.
    if tz == "broker":
        idx_naive = pd.DatetimeIndex(df.index.tz_convert("America/New_York").tz_localize(None)) + pd.Timedelta(hours=7)
    else:
        idx_naive = df.index.tz_convert(tz).tz_localize(None)
    py = signal_to_strategy(f"{strategy} (python)", "", "", signal, close, idx_naive,
                            notional=notional, spread_frac=spread_frac)
    mt5 = MT5Parser().parse_file(report)
    bar_seconds = int(pd.Series(df.index).diff().dt.total_seconds().median())
    res = compare(mt5, py, bar_seconds, tolerance_bars)
    if verbose:
        print(f"MT5 report : {mt5.name} {mt5.symbol} {mt5.timeframe}, {res['mt5_entries']} entries, net {res['mt5_net']:,.2f}")
        print(f"Python     : {strategy} {params}, {res['py_entries']} entries, net {res['py_net']:,.2f}")
        print(f"matched    : {res['matched']}  (MT5 {res['match_rate_mt5']:.1%}, Python {res['match_rate_py']:.1%})")
        print(f"P/L corr   : {res['pnl_correlation']}")
        verdict = ("FAITHFUL" if res["match_rate_mt5"] >= 0.98 and res["match_rate_py"] >= 0.98
                   and (res["pnl_correlation"] or 0) >= 0.99 else "MISMATCH")
        print(f"verdict    : {verdict}")
        if len(res["unmatched_mt5"]):
            print("\nEA entries with no Python match (first 25):")
            print(res["unmatched_mt5"].to_string(index=False))
        if len(res["unmatched_py"]):
            print("\nPython entries with no EA match (first 25):")
            print(res["unmatched_py"].to_string(index=False))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="MT5 Strategy Tester HTML report")
    ap.add_argument("--bars", required=True, help="MT5 bars export of the same symbol/period")
    ap.add_argument("--strategy", required=True, help="registered strategy name")
    ap.add_argument("--params", required=True, help="JSON params, e.g. 40 or [40, 0.002]")
    ap.add_argument("--tz", default="broker", help="timezone of the bars export (broker | UTC | zone name)")
    ap.add_argument("--tolerance", type=int, default=1, help="bars of timing tolerance")
    a = ap.parse_args(argv)
    params = json.loads(a.params)
    if isinstance(params, list):
        params = tuple(params)
    run(a.report, a.bars, a.strategy, params, a.tz, a.tolerance)


if __name__ == "__main__":
    main()
