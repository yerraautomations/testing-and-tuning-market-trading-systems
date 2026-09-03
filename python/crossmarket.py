"""
Cross-market robustness report.

Runs ONE strategy across MANY markets and reports, per market:
  - in-sample optimized params and profit factor
  - in-sample MCPT p-value            (is the edge real on this market?)
  - walk-forward OOS profit factor    (does it generalize on this market?)
  - optional walk-forward MCPT p-value

Then gives a verdict on whether the edge generalizes or looks like a
single-market fluke. Passing on exactly one of many related markets is the
classic fingerprint of luck surviving the single-market test (with 7 markets,
~1 in 20 at p<0.05 pass by chance). A genuine edge with an economic rationale
usually shows up, at least directionally, on related instruments.

Rule of thumb applied here:
  PASS on a market  = in-sample p < 0.01 AND walk-forward OOS PF > 1
  directional       = walk-forward OOS PF > 1 (right sign, maybe not significant)
"""

from functools import partial

import numpy as np
import pandas as pd

from data_loader import load
from strategies_library import (STRATEGIES, opt_for_insample,
                                opt_for_walkforward, signal_for_walkforward,
                                load_user_strategies)
from objectives import profit_factor, get_strategy_returns
from mcpt import walkforward_signal
from mcpt_fast import insample_mcpt_parallel, walkforward_mcpt_parallel


def evaluate_market(spec, df, train_frac=0.7, train_bars=1000, train_step=126,
                    n_permutations=200, wf_permutations=0, seed=0,
                    n_jobs=None):
    """Full in-sample + walk-forward evaluation of one strategy on one market."""
    is_opt = partial(opt_for_insample, optimize=spec["optimize"])
    wf_opt = partial(opt_for_walkforward, optimize=spec["optimize"])
    wf_sig = partial(signal_for_walkforward, signal=spec["signal"])

    n = len(df)
    split = int(n * train_frac)
    train = df.iloc[:split].reset_index(drop=True)

    params, is_pf, _ = spec["optimize"](train["Close"].values)
    is_res = insample_mcpt_parallel(train, is_opt, profit_factor,
                                    n_permutations=n_permutations, seed=seed,
                                    n_jobs=n_jobs, verbose=False)

    # Shrink the walk-forward windows for short series.
    tb = min(train_bars, split)
    ts = max(10, min(train_step, tb // 4))
    wf = walkforward_signal(df, wf_opt, wf_sig, tb, ts)
    wf_rets = get_strategy_returns(wf, df["Close"].values)[tb:]
    wf_pf = profit_factor(wf_rets)

    row = {
        "bars": n,
        "params": str(params),
        "is_pf": is_pf,
        "is_p": is_res["p_value"],
        "wf_pf": wf_pf,
        "wf_bars": n - tb,
    }
    if wf_permutations:
        wf_res = walkforward_mcpt_parallel(
            df, wf_opt, wf_sig, profit_factor, tb, ts,
            n_permutations=wf_permutations, seed=seed + 1, n_jobs=n_jobs,
            verbose=False)
        row["wf_p"] = wf_res["p_value"]
    return row


def cross_market_report(strategy, datasets, verbose=True, **kwargs):
    """
    Parameters
    ----------
    strategy : str or dict
        Name in STRATEGIES, or a strategy spec dict.
    datasets : dict
        market name -> OHLC DataFrame (DatetimeIndex).
    **kwargs : forwarded to evaluate_market.

    Returns
    -------
    table : pd.DataFrame (one row per market)
    verdict : str
    """
    load_user_strategies()
    spec = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    name = strategy if isinstance(strategy, str) else spec.get("name", "?")

    rows = {}
    for mkt, df in datasets.items():
        if verbose:
            print(f"  evaluating {name} on {mkt} ({len(df)} bars)...")
        rows[mkt] = evaluate_market(spec, df, **kwargs)
    table = pd.DataFrame(rows).T
    for c in ("is_pf", "is_p", "wf_pf"):
        table[c] = table[c].astype(float)
    if "wf_p" in table:
        table["wf_p"] = table["wf_p"].astype(float)

    n_mkts = len(table)
    passed = table[(table["is_p"] < 0.01) & (table["wf_pf"] > 1.0)]
    n_pass = len(passed)
    n_dir = int((table["wf_pf"] > 1.0).sum())

    if n_pass == 0:
        verdict = ("FAIL: no market passes both the in-sample MCPT and a "
                   "profitable walk-forward.")
    elif n_pass == 1:
        verdict = (f"CAUTION: passes on only 1 of {n_mkts} markets "
                   f"({passed.index[0]}). Single-market pass = fluke risk. "
                   f"Treat as noise unless you can name, in advance, why it "
                   f"should be specific to that market. "
                   f"({n_dir}/{n_mkts} markets are at least directionally "
                   f"profitable OOS.)")
    elif n_pass * 2 >= n_mkts:
        verdict = (f"PASS: edge generalizes -- passes on {n_pass}/{n_mkts} "
                   f"markets ({', '.join(passed.index)}); "
                   f"{n_dir}/{n_mkts} directionally profitable OOS.")
    else:
        verdict = (f"PARTIAL: passes on {n_pass}/{n_mkts} markets "
                   f"({', '.join(passed.index)}); {n_dir}/{n_mkts} "
                   f"directionally profitable OOS. Look for a reason "
                   f"(spread, session, volatility) before trusting it.")

    if verbose:
        pd.set_option("display.width", 140)
        print(f"\n=== Cross-market report: {name} ===")
        print(table.round(4).to_string())
        print(f"\nVerdict: {verdict}")
    return table, verdict


def load_many(names):
    """Load several built-in datasets into a dict."""
    return {n: load(n) for n in names}


if __name__ == "__main__":
    import sys
    strat = sys.argv[1] if len(sys.argv) > 1 else "momentum"
    markets = sys.argv[2:] or ["BTC", "AAPL"]
    cross_market_report(strat, load_many(markets), n_permutations=100, seed=3)
