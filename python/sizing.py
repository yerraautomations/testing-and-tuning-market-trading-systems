"""
Risk-sizing layer: volatility targeting.

For prop-firm challenges the binding constraint is the daily-loss limit, and
what breaches it is not a bad entry signal but a position that is too large
for the market's current volatility. Volatility targeting scales the
position so the account's expected volatility stays near a chosen level,
automatically shrinking exposure when the market gets wild.

    sized_signal[t] = signal[t] * clip(target_vol / realized_vol[t], 0, max_lev)

realized_vol[t] uses only returns up to and including bar t, and the sized
position taken at bar t is applied to bar t+1 -- no look-ahead.

The sized signal is fractional (e.g. +0.37) and plugs straight into
propfirm.equity_path() and objectives.get_strategy_returns().

Choosing the target: target_vol_for_daily_limit() derives an annualized
target from the firm's daily-loss rule, e.g. "a 3-sigma down day must stay
inside 5%" -> daily sigma 1.67% -> ~26% annualized (252 periods/yr).
"""

import numpy as np
import pandas as pd


def realized_vol(prices, lookback=20, periods_per_year=252):
    """Trailing annualized volatility of log returns (NaN during warm-up)."""
    p = np.asarray(prices, dtype=float)
    r = np.full(len(p), np.nan)
    r[1:] = np.diff(np.log(p))
    return pd.Series(r).rolling(lookback).std().values * np.sqrt(periods_per_year)


def target_vol_for_daily_limit(max_daily_loss, sigmas=3.0, periods_per_year=252):
    """
    Annualized volatility such that a `sigmas`-sigma daily move stays inside
    the daily-loss limit. Higher `sigmas` = more conservative.
    """
    return (max_daily_loss / sigmas) * np.sqrt(periods_per_year)


def vol_target(signal, prices, target_vol=0.20, lookback=20,
               periods_per_year=252, max_leverage=3.0, min_change=0.10):
    """
    Scale a +1 / -1 / 0 signal to a target annualized volatility.

    Parameters
    ----------
    signal : array of {+1, -1, 0}
    prices : close prices aligned to signal
    target_vol : annualized volatility target for the account (0.20 = 20%)
    lookback : bars used for realized vol
    periods_per_year : 252 for daily FX/stocks, 365 for daily crypto,
                       252*24 for hourly FX, etc.
    max_leverage : cap on the scale factor
    min_change : only re-size an open position when the desired size differs
                 from the current size by more than this fraction, to avoid
                 paying costs on tiny daily rebalances. Direction changes
                 and entries/exits always execute.

    Returns
    -------
    np.ndarray of fractional positions (0 during vol warm-up).
    """
    signal = np.asarray(signal, dtype=float)
    vol = realized_vol(prices, lookback, periods_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = target_vol / vol
    scale = np.where(np.isfinite(scale), scale, 0.0)
    scale = np.clip(scale, 0.0, max_leverage)

    out = np.zeros(len(signal))
    cur = 0.0
    for t in range(len(signal)):
        desired = signal[t] * scale[t]
        if desired == 0.0:
            cur = 0.0
        elif (cur == 0.0 or np.sign(cur) != np.sign(desired)
              or abs(desired - cur) > min_change * abs(cur)):
            cur = desired
        out[t] = cur
    return out


if __name__ == "__main__":
    # Demo: does vol targeting rescue the FTMO pass rate of momentum on BTC?
    from functools import partial
    from data_loader import load
    from strategies_library import (STRATEGIES, opt_for_walkforward,
                                    signal_for_walkforward)
    from mcpt import walkforward_signal
    from propfirm import (FTMO_2STEP, CostModel, equity_path, daily_table,
                          historical_pass_rate, bootstrap_pass_rate)

    df = load("BTC")
    spec = STRATEGIES["momentum"]
    wf_opt = partial(opt_for_walkforward, optimize=spec["optimize"])
    wf_sig = partial(signal_for_walkforward, signal=spec["signal"])
    TB, TS = 1000, 126
    sig = walkforward_signal(df, wf_opt, wf_sig, TB, TS)
    oos = df.iloc[TB:]
    close = oos["Close"].values
    raw = sig[TB:]
    costs = CostModel(spread_frac=0.0005)
    PPY = 365  # crypto trades every day

    variants = {
        "fixed 1.0x": raw * 1.0,
        "fixed 0.25x": raw * 0.25,
    }
    for sigmas in (3.0, 4.0):
        tv = target_vol_for_daily_limit(FTMO_2STEP.max_daily_loss, sigmas, PPY)
        variants[f"vol-target {tv:.0%} ({sigmas:.0f}-sigma)"] = vol_target(
            raw, close, tv, lookback=20, periods_per_year=PPY, max_leverage=2.0)

    print("=" * 78)
    print("  SIZING vs FTMO 2-Step pass rate  (momentum on BTC, walk-forward OOS)")
    print("=" * 78)
    print(f"{'sizing':30s} {'final eq':>9s} {'hist P(pass)':>13s} "
          f"{'P(daily)':>9s} {'P(max)':>7s} {'boot365 P(pass)':>16s} "
          f"{'med days':>9s}")
    for name, s in variants.items():
        eq, opened = equity_path(s, close, 1.0, costs)
        days = daily_table(eq, opened, oos.index, FTMO_2STEP.daily_reset_tz)
        h = historical_pass_rate(days, FTMO_2STEP, horizon=365)
        b = bootstrap_pass_rate(days, FTMO_2STEP, 1000, 365, seed=1)
        md = b["median_days_to_pass"]
        print(f"{name:30s} {eq[-1]:9.2f} {h['p_pass']:13.1%} "
              f"{h['p_fail_daily']:9.1%} {h['p_fail_max']:7.1%} "
              f"{b['p_pass']:16.1%} "
              f"{(f'{md:.0f}' if not np.isnan(md) else 'n/a'):>9s}")
