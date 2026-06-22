"""
Full 4-step strategy validation on REAL market data.

This ties everything together:
  - data_loader        -> real OHLC prices (cached locally)
  - strategies_library -> a library of vectorized strategies
  - mcpt_fast          -> parallel Monte-Carlo Permutation Tests

The four steps (from the video / Masters' book):
  1. In-Sample Excellence       - optimize, inspect the result
  2. In-Sample MCPT             - is it real, or just data-mining bias?
  3. Walk-Forward Test          - does it hold up on unseen data?
  4. Walk-Forward MCPT          - or were the OOS results just luck?

Usage:
    python validate.py                      # BTC + donchian (defaults)
    python validate.py BTC ma_cross
    python validate.py AAPL momentum

Available datasets:   BTC, AAPL   (see data_loader.list_datasets())
Available strategies: donchian, ma_cross, rsi, bollinger, momentum
"""

import sys
from functools import partial

import numpy as np

from data_loader import load
from strategies_library import (
    STRATEGIES, opt_for_insample, opt_for_walkforward, signal_for_walkforward,
)
from objectives import profit_factor, sharpe_ratio, get_strategy_returns
from mcpt import walkforward_signal
from mcpt_fast import insample_mcpt_parallel, walkforward_mcpt_parallel


# Defaults tuned for DAILY data (both BTC and AAPL are daily here).
DEFAULTS = dict(
    train_frac=0.70,        # fraction of data used for the in-sample stage
    train_bars=1000,        # walk-forward training window (~4 trading years)
    train_step=126,         # re-optimize every ~6 months
    is_permutations=300,    # in-sample permutations
    wf_permutations=200,    # walk-forward permutations
)


def run(dataset="BTC", strategy="donchian", periods_per_year=252,
        seed=42, **overrides):
    cfg = dict(DEFAULTS)
    cfg.update(overrides)

    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from {list(STRATEGIES)}")
    spec = STRATEGIES[strategy]

    # Picklable adapters bound to the chosen strategy (work on fork & spawn).
    is_opt = partial(opt_for_insample, optimize=spec["optimize"])
    wf_opt = partial(opt_for_walkforward, optimize=spec["optimize"])
    wf_sig = partial(signal_for_walkforward, signal=spec["signal"])

    # ---- Load real data ----
    df = load(dataset)
    close = df["Close"].values
    n = len(df)
    print("=" * 70)
    print(f"  VALIDATING: {strategy} ({spec['kind']}) on {dataset}")
    print("=" * 70)
    print(f"Data: {n} bars, {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Strategy params: {spec['params']}")

    split = int(n * cfg["train_frac"])
    train_df = df.iloc[:split].reset_index(drop=True)
    print(f"In-sample: {split} bars | Out-of-sample: {n - split} bars\n")

    # ---- STEP 1: In-Sample Excellence ----
    print("-" * 70)
    print("STEP 1: In-Sample Excellence")
    print("-" * 70)
    params, is_pf, is_sig = spec["optimize"](train_df["Close"].values)
    is_rets = get_strategy_returns(is_sig, train_df["Close"].values)
    print(f"Best params:      {params}")
    print(f"Profit factor:    {is_pf:.4f}")
    print(f"Sharpe (ann.):    {sharpe_ratio(is_rets, periods_per_year):.3f}")
    print(f"Total log return: {np.sum(is_rets):.4f}")

    # ---- STEP 2: In-Sample MCPT ----
    print("\n" + "-" * 70)
    print(f"STEP 2: In-Sample MCPT ({cfg['is_permutations']} permutations)")
    print("-" * 70)
    is_res = insample_mcpt_parallel(
        train_df, is_opt, profit_factor,
        n_permutations=cfg["is_permutations"], seed=seed, verbose=True)
    is_pass = is_res["p_value"] < 0.01
    print(f"Verdict: {'PASS' if is_pass else 'FAIL'} "
          f"(want p < 0.01)")

    # ---- STEP 3: Walk-Forward Test ----
    print("\n" + "-" * 70)
    print("STEP 3: Walk-Forward Test")
    print("-" * 70)
    tb, ts = cfg["train_bars"], cfg["train_step"]
    if tb >= n:
        tb = split  # fall back for short series
    wf = walkforward_signal(df, wf_opt, wf_sig, tb, ts)
    wf_rets = get_strategy_returns(wf, close)[tb:]
    wf_pf = profit_factor(wf_rets)
    print(f"Train window: {tb} bars | Re-optimize every: {ts} bars")
    print(f"OOS profit factor: {wf_pf:.4f}")
    print(f"OOS Sharpe (ann.): {sharpe_ratio(wf_rets, periods_per_year):.3f}")
    print(f"OOS total log ret: {np.sum(wf_rets):.4f}")

    # ---- STEP 4: Walk-Forward MCPT ----
    print("\n" + "-" * 70)
    print(f"STEP 4: Walk-Forward MCPT ({cfg['wf_permutations']} permutations)")
    print("-" * 70)
    wf_res = walkforward_mcpt_parallel(
        df, wf_opt, wf_sig, profit_factor, tb, ts,
        n_permutations=cfg["wf_permutations"], seed=seed + 1, verbose=True)
    wf_pass = wf_res["p_value"] < 0.05
    print(f"Verdict: {'PASS' if wf_pass else 'FAIL'} "
          f"(want p < 0.05 for ~1 yr OOS, p < 0.01 for 2+ yrs)")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Step 1  In-sample PF:        {is_pf:.4f}")
    print(f"  Step 2  In-sample MCPT p:    {is_res['p_value']:.4f}  "
          f"{'PASS' if is_pass else 'FAIL'}")
    print(f"  Step 3  Walk-forward PF:     {wf_pf:.4f}")
    print(f"  Step 4  Walk-forward MCPT p: {wf_res['p_value']:.4f}  "
          f"{'PASS' if wf_pass else 'FAIL'}")
    if is_pass and wf_pass:
        print("\n  VERDICT: Passes both permutation tests.")
    else:
        print("\n  VERDICT: Did NOT pass both tests -- do not trade as-is.")
    print("=" * 70)

    return {"insample": is_res, "walkforward": wf_res,
            "is_pf": is_pf, "wf_pf": wf_pf}


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    strat = sys.argv[2] if len(sys.argv) > 2 else "donchian"
    run(ds, strat)
