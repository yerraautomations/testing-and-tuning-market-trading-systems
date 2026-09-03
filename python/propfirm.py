"""
Prop-firm challenge simulator.

Answers the question the rest of the framework can't: "Would this strategy
PASS a prop-firm evaluation (FTMO etc.) -- and with what probability?"

Pipeline
--------
  signal + prices + timestamps
      -> equity_path()      apply leverage + transaction costs, bar by bar
      -> daily_table()      collapse to one row per trading day in the firm's
                            timezone: day return, intra-day worst point,
                            whether a position was opened
      -> run_challenge()    walk the day table phase by phase, checking every
                            rule (daily loss, max loss, profit target,
                            min trading days, optional time limit)
      -> historical_pass_rate() / bootstrap_pass_rate()
                            P(pass), P(fail by daily limit), P(fail by max
                            loss), median days to fund
      -> sweep_leverage()   the "trade smaller, take longer" trade-off

Everything is normalized to an initial balance of 1.0, so a 5% daily-loss
rule is simply "day_start - intraday_low > 0.05".

Granularity note: with daily bars the intra-day low is just the close, so the
daily-loss check is close-to-close. With M1 data it becomes a true intra-day
check. The code is identical in both cases.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropFirmRules:
    name: str
    profit_targets: Tuple[float, ...]   # per phase, fraction of initial balance
    max_daily_loss: float               # fraction of initial balance
    max_total_loss: float               # fraction of initial balance
    max_loss_mode: str = "static"       # "static" (from initial) | "trailing" (from high-water mark)
    min_trading_days: int = 4           # per phase
    max_days: Optional[int] = None      # per phase; None = no time limit
    daily_reset_tz: str = "Europe/Prague"   # FTMO measures the day in CE(S)T


# FTMO rules as of 2026 (see ftmo.com/en/trading-objectives). Some accounts
# use an 8% phase-1 target; override profit_targets if yours does.
FTMO_2STEP = PropFirmRules(
    name="FTMO 2-Step", profit_targets=(0.10, 0.05),
    max_daily_loss=0.05, max_total_loss=0.10, max_loss_mode="static",
    min_trading_days=4, max_days=None, daily_reset_tz="Europe/Prague")

FTMO_1STEP = PropFirmRules(
    name="FTMO 1-Step", profit_targets=(0.10,),
    max_daily_loss=0.03, max_total_loss=0.10, max_loss_mode="trailing",
    min_trading_days=4, max_days=None, daily_reset_tz="Europe/Prague")


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """
    Transaction costs, all as fractions of price/notional.

    spread_frac     full bid-ask spread as a fraction of price
                    (use pips_to_frac() for FX)
    commission_frac commission per side as a fraction of notional
    stress          multiplier for sensitivity tests (1x, 2x, 3x costs)
    """
    spread_frac: float = 0.0
    commission_frac: float = 0.0
    stress: float = 1.0

    def per_unit_change(self):
        # Half the spread is paid on entry and half on exit, so charging
        # half-spread per unit of position change is exact for a round trip
        # and also correct for a flip (+1 -> -1 = close + open = full spread).
        return (self.spread_frac / 2.0 + self.commission_frac) * self.stress


def pips_to_frac(pips, price, pip_size=0.0001):
    """Convert a spread in pips to a fraction of price (JPY pairs: pip_size=0.01)."""
    return pips * pip_size / price


# ---------------------------------------------------------------------------
# Equity path and daily table
# ---------------------------------------------------------------------------

def equity_path(signal, prices, leverage=1.0, costs=None):
    """
    Bar-by-bar account equity for a position signal.

    The position set at bar i (signal[i]) earns the price move of bar i+1.
    Position changes are executed at the close of the bar in which the
    signal changes and are charged transaction costs then.

    Parameters
    ----------
    signal : array of {+1, -1, 0} (or fractional sizes)
    prices : close prices aligned to signal
    leverage : notional exposure per unit signal, as a multiple of equity
    costs : CostModel or None

    Returns
    -------
    equity : np.ndarray, starts at 1.0
    opened : np.ndarray of bool, True on bars where a position was opened
    """
    signal = np.asarray(signal, dtype=float)
    prices = np.asarray(prices, dtype=float)
    n = len(signal)

    price_ret = np.zeros(n)
    price_ret[1:] = prices[1:] / prices[:-1] - 1.0        # simple returns
    held = np.zeros(n)
    held[1:] = signal[:-1]                                  # position during bar i
    gross = 1.0 + leverage * held * price_ret

    dpos = np.abs(np.diff(signal, prepend=0.0))             # size of each trade
    unit_cost = costs.per_unit_change() if costs is not None else 0.0
    net = gross * (1.0 - leverage * dpos * unit_cost)

    equity = np.cumprod(net)
    opened = (dpos > 0) & (signal != 0)
    return equity, opened


def daily_table(equity, opened, index, tz="Europe/Prague"):
    """
    Collapse a bar-level equity path to one row per trading day.

    Columns (all relative to that day's starting equity):
      ret    : end-of-day equity / day-start equity - 1
      min    : lowest intra-day equity / day-start equity - 1  (<= 0)
      opened : True if a position was opened during the day
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    day = idx.tz_convert(tz).date

    df = pd.DataFrame({"equity": equity, "opened": opened, "day": day})
    g = df.groupby("day", sort=True)
    end = g["equity"].last()
    low = g["equity"].min()
    opened_any = g["opened"].any()

    start = end.shift(1)
    start.iloc[0] = 1.0
    out = pd.DataFrame({
        "ret": end / start - 1.0,
        "min": np.minimum(low / start - 1.0, 0.0),
        "opened": opened_any,
    })
    out.index.name = "day"
    return out


# ---------------------------------------------------------------------------
# Challenge simulation
# ---------------------------------------------------------------------------

def _run(rets, mins, opened, rules, start=0, horizon=None):
    """
    Walk the day arrays from row `start`, phase by phase.

    Returns dict with outcome in {PASS, FAIL_DAILY, FAIL_MAX, FAIL_TIME,
    INCOMPLETE}, per-phase details, total_days, and end_row.
    INCOMPLETE = ran out of data (or exceeded `horizon`) before resolving.
    """
    n = len(rets)
    i = start
    total = 0
    phases = []

    for target in rules.profit_targets:
        eq = 1.0
        hwm = 1.0
        tdays = 0
        d = 0
        outcome = "INCOMPLETE"

        while i < n:
            if horizon is not None and total >= horizon:
                break
            d += 1
            total += 1
            day_start = eq
            worst = day_start * (1.0 + mins[i])

            # Daily loss: measured from the day's starting equity, sized as a
            # fraction of the initial balance (normalized to 1.0).
            if day_start - worst > rules.max_daily_loss + 1e-12:
                outcome = "FAIL_DAILY"
                i += 1
                break

            # Max loss: static floor from initial, or trailing from the
            # high-water mark of end-of-day equity.
            if rules.max_loss_mode == "trailing":
                floor = hwm - rules.max_total_loss
            else:
                floor = 1.0 - rules.max_total_loss
            if worst <= floor + 1e-12:
                outcome = "FAIL_MAX"
                i += 1
                break

            eq = day_start * (1.0 + rets[i])
            hwm = max(hwm, eq)
            if opened[i]:
                tdays += 1
            i += 1

            if eq >= 1.0 + target - 1e-12 and tdays >= rules.min_trading_days:
                outcome = "PASS"
                break
            if rules.max_days is not None and d >= rules.max_days:
                outcome = "FAIL_TIME"
                break

        phases.append({"outcome": outcome, "days": d,
                       "trading_days": tdays, "equity": eq})
        if outcome != "PASS":
            break

    if len(phases) == len(rules.profit_targets) and \
            all(p["outcome"] == "PASS" for p in phases):
        overall = "PASS"
    else:
        overall = phases[-1]["outcome"]

    return {"outcome": overall, "phases": phases,
            "total_days": total, "end_row": i}


def run_challenge(days, rules, start=0, horizon=None):
    """Simulate one challenge from row `start` of a daily_table() frame."""
    return _run(days["ret"].values, days["min"].values,
                days["opened"].values, rules, start, horizon)


def _summarize(outcomes):
    n = len(outcomes)
    counts = Counter(o["outcome"] for o in outcomes)
    pass_days = [o["total_days"] for o in outcomes if o["outcome"] == "PASS"]
    return {
        "n": n,
        "p_pass": counts["PASS"] / n if n else np.nan,
        "p_fail_daily": counts["FAIL_DAILY"] / n if n else np.nan,
        "p_fail_max": counts["FAIL_MAX"] / n if n else np.nan,
        "p_fail_time": counts["FAIL_TIME"] / n if n else np.nan,
        "p_incomplete": counts["INCOMPLETE"] / n if n else np.nan,
        "median_days_to_pass": float(np.median(pass_days)) if pass_days else np.nan,
        "counts": dict(counts),
    }


def historical_pass_rate(days, rules, horizon=None, step=1):
    """
    Start a challenge on every historical day (stride `step`) and tally the
    outcomes. This preserves real path dependence (regimes, clustering).

    `horizon` caps the total days a challenge may take; runs that don't
    resolve within it (or before the data ends) count as INCOMPLETE.
    """
    rets = days["ret"].values
    mins = days["min"].values
    opened = days["opened"].values
    outcomes = [_run(rets, mins, opened, rules, s, horizon)
                for s in range(0, len(rets), step)]
    return _summarize(outcomes)


def bootstrap_pass_rate(days, rules, n_sims=1000, horizon=365, block=10,
                        seed=None):
    """
    Moving-block bootstrap of the daily table: rebuild `n_sims` synthetic
    day sequences of length `horizon` from random contiguous blocks of
    `block` real days, and run a challenge on each.

    Blocks preserve short-range clustering (streaks, volatility bursts) that
    a plain shuffle would destroy. `horizon` is the window you care about --
    e.g. 60 to ask "P(funded within ~2 months)".
    """
    rng = np.random.default_rng(seed)
    rets = days["ret"].values
    mins = days["min"].values
    opened = days["opened"].values
    n = len(rets)
    block = min(block, n)

    outcomes = []
    for _ in range(n_sims):
        idx = []
        while len(idx) < horizon:
            s = rng.integers(0, n - block + 1)
            idx.extend(range(s, s + block))
        idx = np.asarray(idx[:horizon])
        outcomes.append(_run(rets[idx], mins[idx], opened[idx], rules,
                             0, horizon))
    return _summarize(outcomes)


# ---------------------------------------------------------------------------
# Leverage sweep -- the "trade smaller, take longer" trade-off
# ---------------------------------------------------------------------------

def sweep_leverage(signal, prices, index, rules, leverages, costs=None,
                   horizon=365, n_sims=1000, seed=0):
    """
    For each leverage, report historical and bootstrap pass statistics.

    Returns a DataFrame with one row per leverage.
    """
    rows = []
    for lev in leverages:
        eq, opened = equity_path(signal, prices, lev, costs)
        days = daily_table(eq, opened, index, rules.daily_reset_tz)
        h = historical_pass_rate(days, rules, horizon)
        b = bootstrap_pass_rate(days, rules, n_sims, horizon, seed=seed)
        rows.append({
            "leverage": lev,
            "hist_p_pass": h["p_pass"],
            "hist_p_fail_daily": h["p_fail_daily"],
            "hist_p_fail_max": h["p_fail_max"],
            "hist_p_incomplete": h["p_incomplete"],
            "hist_median_days": h["median_days_to_pass"],
            "boot_p_pass": b["p_pass"],
            "boot_p_fail_daily": b["p_fail_daily"],
            "boot_p_fail_max": b["p_fail_max"],
            "boot_median_days": b["median_days_to_pass"],
        })
    return pd.DataFrame(rows)


def print_summary(title, s):
    print(f"--- {title} ---")
    print(f"  challenges:      {s['n']}")
    print(f"  P(pass):         {s['p_pass']:.1%}")
    print(f"  P(fail daily):   {s['p_fail_daily']:.1%}")
    print(f"  P(fail max):     {s['p_fail_max']:.1%}")
    if s["p_fail_time"] > 0:
        print(f"  P(fail time):    {s['p_fail_time']:.1%}")
    print(f"  P(unresolved):   {s['p_incomplete']:.1%}")
    md = s["median_days_to_pass"]
    print(f"  median days:     {md:.0f}" if not np.isnan(md) else
          "  median days:     n/a")


# ---------------------------------------------------------------------------
# Demo: momentum on BTC, walk-forward OOS signal, FTMO 2-step
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from functools import partial
    from data_loader import load
    from strategies_library import (STRATEGIES, opt_for_walkforward,
                                    signal_for_walkforward)
    from mcpt import walkforward_signal

    df = load("BTC")
    spec = STRATEGIES["momentum"]
    wf_opt = partial(opt_for_walkforward, optimize=spec["optimize"])
    wf_sig = partial(signal_for_walkforward, signal=spec["signal"])
    TRAIN_BARS, TRAIN_STEP = 1000, 126

    # Honest input: the walk-forward signal, OOS portion only.
    sig = walkforward_signal(df, wf_opt, wf_sig, TRAIN_BARS, TRAIN_STEP)
    oos = df.iloc[TRAIN_BARS:]
    sig_oos = sig[TRAIN_BARS:]
    costs = CostModel(spread_frac=0.0005)   # ~5 bps round-trip spread

    print("=" * 70)
    print(f"  PROP-FIRM SIMULATION: momentum on BTC (walk-forward OOS)")
    print(f"  Rules: {FTMO_2STEP.name} | OOS: {len(oos)} days "
          f"{oos.index[0].date()} -> {oos.index[-1].date()}")
    print("=" * 70)

    for lev in (1.0,):
        eq, opened = equity_path(sig_oos, oos["Close"].values, lev, costs)
        days = daily_table(eq, opened, oos.index, FTMO_2STEP.daily_reset_tz)
        print(f"\nLeverage {lev}x  |  final equity {eq[-1]:.2f}x  |  "
              f"trading days {int(days['opened'].sum())}")
        print_summary("Historical (start any day, no horizon)",
                      historical_pass_rate(days, FTMO_2STEP))
        print_summary("Bootstrap, 60-day horizon (\"funded in ~2 months?\")",
                      bootstrap_pass_rate(days, FTMO_2STEP, 2000, 60, seed=1))
        print_summary("Bootstrap, 365-day horizon",
                      bootstrap_pass_rate(days, FTMO_2STEP, 2000, 365, seed=1))

    print("\n" + "=" * 70)
    print("  LEVERAGE SWEEP (365-day horizon)")
    print("=" * 70)
    table = sweep_leverage(sig_oos, oos["Close"].values, oos.index,
                           FTMO_2STEP, [0.25, 0.5, 1.0, 2.0, 3.0],
                           costs, horizon=365, n_sims=1000)
    pd.set_option("display.width", 140)
    print(table.round(3).to_string(index=False))
