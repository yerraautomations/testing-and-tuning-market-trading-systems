"""
Prop-firm challenge simulator (generic rules, any firm).

Answers the question the rest of the framework can't: "Would this strategy
PASS a prop-firm evaluation -- and with what probability?"

Pipeline
--------
  signal + prices + timestamps
      -> equity_path()        apply leverage + transaction costs, bar by bar
      -> daily_table()        one row per trading day in the firm's timezone:
                              day return, intra-day low and high, traded flag
      -> run_challenge()      walk the day table phase by phase, checking every
                              rule (daily loss, static/trailing max loss,
                              targets or payout threshold, min days,
                              consistency, time limit)
      -> historical_pass_rate() / bootstrap_pass_rate()
                              P(pass), P(fail daily), P(fail max), median days
      -> sweep_leverage()     the "trade smaller, take longer" trade-off

Trade lists (MT5 reports) can be simulated too: daily_table_from_pnl().

Everything is normalized to an initial balance of 1.0, so a 5% daily-loss
rule is simply "day_start - intraday_low > 0.05".

Rules live in PropFirmRules; presets are read from config/propfirms.json
(FTMO verified; other firms approximate -- verify before relying on them)
and can be edited or extended there.
"""

import json
import os
from collections import Counter
from dataclasses import dataclass, asdict, replace
from typing import Optional, Tuple

import numpy as np
import pandas as pd

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropFirmRules:
    name: str
    profit_targets: Tuple[float, ...] = (0.10, 0.05)  # per phase, fraction of account; () = no target (instant funding)
    max_daily_loss: float = 0.05        # fraction of account, measured from the day's starting equity; 0 = no daily limit
    max_total_loss: float = 0.10        # fraction of account
    max_loss_mode: str = "static"       # "static" (from initial) | "trailing" (from high-water mark)
    trailing_basis: str = "eod"         # "eod" (end-of-day equity) | "intraday" (intraday high) for the high-water mark
    trailing_lock_at_initial: bool = False   # trailing floor never rises above the initial balance
    min_trading_days: int = 4           # per phase
    max_days: Optional[int] = None      # per phase; None = no time limit
    consistency_max_day_pct: Optional[float] = None  # best day's profit / total profit must be <= this to pass
    payout_threshold: float = 0.0       # no-target accounts: profit fraction that counts as "funded/paid"
    daily_reset_tz: str = "Europe/Prague"
    account_size: float = 100_000.0
    fee: float = 0.0                    # evaluation / activation fee, same currency as account
    profit_split: float = 0.8
    asset_class: str = "any"            # info only: forex | futures | any
    verified: bool = False              # True only when the rules were checked against the firm's current terms
    notes: str = ""

    def targets(self):
        if self.profit_targets:
            return tuple(self.profit_targets)
        return (self.payout_threshold,) if self.payout_threshold > 0 else ()


FTMO_2STEP = PropFirmRules(
    name="FTMO 2-Step", profit_targets=(0.10, 0.05), max_daily_loss=0.05,
    max_total_loss=0.10, max_loss_mode="static", min_trading_days=4,
    daily_reset_tz="Europe/Prague", account_size=100_000, fee=540,
    profit_split=0.8, verified=True,
    notes="Rules checked 2026-09 (ftmo.com/en/trading-objectives). Some accounts use an 8% phase-1 target.")

FTMO_1STEP = PropFirmRules(
    name="FTMO 1-Step", profit_targets=(0.10,), max_daily_loss=0.03,
    max_total_loss=0.10, max_loss_mode="trailing", trailing_basis="eod",
    min_trading_days=4, daily_reset_tz="Europe/Prague", account_size=100_000,
    fee=540, profit_split=0.8, verified=False,
    notes="Trailing details approximate; verify.")


def load_presets(path=None):
    """Read config/propfirms.json -> {name: PropFirmRules}. Built-ins if missing."""
    path = path or os.path.join(CONFIG_DIR, "propfirms.json")
    presets = {FTMO_2STEP.name: FTMO_2STEP, FTMO_1STEP.name: FTMO_1STEP}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("presets", []):
            item = dict(item)
            item["profit_targets"] = tuple(item.get("profit_targets", ()))
            presets[item["name"]] = PropFirmRules(**item)
    return presets


def get_rules(name, **overrides):
    """Preset by name with optional field overrides (e.g. account_size=50_000)."""
    rules = load_presets()[name]
    return replace(rules, **overrides) if overrides else rules


def rules_to_dict(rules):
    return asdict(rules)


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """
    Transaction costs as fractions of price / notional.

    spread_frac     full bid-ask spread (or slippage) as a fraction of price;
                    the 1x baseline. For FX use pips_to_frac(); for futures
                    use instruments.InstrumentProfile.cost_frac().
    commission_frac commission per side as a fraction of notional
    stress          multiplier for sensitivity tests (1x, 2x, 3x costs)
    """
    spread_frac: float = 0.0
    commission_frac: float = 0.0
    stress: float = 1.0

    def per_unit_change(self, spread_frac=None):
        # Half the spread is paid on entry and half on exit, so charging
        # half-spread per unit of position change is exact for a round trip
        # and also correct for a flip (+1 -> -1 = close + open = full spread).
        s = self.spread_frac if spread_frac is None else spread_frac
        return (s / 2.0 + self.commission_frac) * self.stress


def pips_to_frac(pips, price, pip_size=0.0001):
    """Convert a spread in pips to a fraction of price (JPY pairs: pip_size=0.01)."""
    return pips * pip_size / price


# ---------------------------------------------------------------------------
# Equity path and daily table
# ---------------------------------------------------------------------------

def equity_path(signal, prices, leverage=1.0, costs=None, spread_series=None):
    """
    Bar-by-bar account equity for a position signal.

    The position set at bar i (signal[i]) earns the price move of bar i+1.
    Position changes execute at the close of the bar in which the signal
    changes and are charged transaction costs then.

    spread_series : optional per-bar spread as a fraction of price (e.g. the
                    Spread column of tick-derived bars divided by price); it
                    replaces costs.spread_frac bar by bar.

    Returns (equity starting at 1.0, opened: bool array of bars where a
    position was opened).
    """
    signal = np.asarray(signal, dtype=float)
    prices = np.asarray(prices, dtype=float)
    n = len(signal)

    price_ret = np.zeros(n)
    price_ret[1:] = prices[1:] / prices[:-1] - 1.0
    held = np.zeros(n)
    held[1:] = signal[:-1]
    gross = 1.0 + leverage * held * price_ret

    dpos = np.abs(np.diff(signal, prepend=0.0))
    if costs is None:
        unit_cost = 0.0
    elif spread_series is not None:
        unit_cost = costs.per_unit_change(np.asarray(spread_series, dtype=float))
        unit_cost = np.where(np.isfinite(unit_cost), unit_cost, costs.per_unit_change())
    else:
        unit_cost = costs.per_unit_change()
    net = gross * (1.0 - leverage * dpos * unit_cost)

    equity = np.cumprod(net)
    opened = (dpos > 0) & (signal != 0)
    return equity, opened


def daily_table(equity, opened, index, tz="Europe/Prague"):
    """
    Collapse a bar-level equity path to one row per trading day.

    Columns (relative to that day's starting equity):
      ret    : end-of-day equity / day-start equity - 1
      min    : lowest intra-day equity / day-start - 1   (<= 0)
      max    : highest intra-day equity / day-start - 1  (>= 0)
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
    high = g["equity"].max()
    opened_any = g["opened"].any()

    start = end.shift(1)
    start.iloc[0] = 1.0
    out = pd.DataFrame({
        "ret": end / start - 1.0,
        "min": np.minimum(low / start - 1.0, 0.0),
        "max": np.maximum(high / start - 1.0, 0.0),
        "opened": opened_any,
    })
    out.index.name = "day"
    return out


def daily_table_from_pnl(times, pnl, account_size, tz="Europe/Prague"):
    """
    Day table from a list of closed-trade P/Ls (e.g. an MT5 report), for
    strategies that exist only as trade lists (Tier B). Intraday path is
    unknown, so min/max are the close-to-close values.

    times : datetimes of the closing deals (naive = broker time is fine as
            long as `tz` matches; tz-aware is converted)
    pnl   : P/L per closed trade in account currency
    """
    idx = pd.DatetimeIndex(times)
    if idx.tz is not None:
        idx = idx.tz_convert(tz)
    day = idx.date
    s = pd.Series(np.asarray(pnl, dtype=float), index=day).groupby(level=0).sum().sort_index()
    equity = account_size + s.cumsum()
    start = equity.shift(1).fillna(account_size)
    ret = equity / start - 1.0
    out = pd.DataFrame({"ret": ret, "min": np.minimum(ret, 0.0),
                        "max": np.maximum(ret, 0.0), "opened": True})
    out.index.name = "day"
    return out


# ---------------------------------------------------------------------------
# Challenge simulation
# ---------------------------------------------------------------------------

def _run(rets, mins, maxs, opened, rules, start=0, horizon=None):
    """
    Walk the day arrays from row `start`, phase by phase.

    Outcome in {PASS, FAIL_DAILY, FAIL_MAX, FAIL_TIME, INCOMPLETE}.
    INCOMPLETE = ran out of data (or exceeded `horizon`) before resolving.
    """
    n = len(rets)
    i = start
    total = 0
    phases = []
    targets = rules.targets()
    if not targets:
        return {"outcome": "INCOMPLETE", "phases": [], "total_days": 0,
                "end_row": start, "note": "no target and no payout threshold"}

    for target in targets:
        eq = 1.0
        hwm = 1.0
        tdays = 0
        d = 0
        best_day = 0.0
        outcome = "INCOMPLETE"

        while i < n:
            if horizon is not None and total >= horizon:
                break
            d += 1
            total += 1
            day_start = eq
            worst = day_start * (1.0 + mins[i])
            high = day_start * (1.0 + maxs[i])

            if rules.max_daily_loss > 0 and day_start - worst > rules.max_daily_loss + 1e-12:
                outcome = "FAIL_DAILY"
                i += 1
                break

            if rules.max_loss_mode == "trailing":
                if rules.trailing_basis == "intraday":
                    hwm = max(hwm, high)
                floor = hwm - rules.max_total_loss
                if rules.trailing_lock_at_initial:
                    floor = min(floor, 1.0)
            else:
                floor = 1.0 - rules.max_total_loss
            if worst <= floor + 1e-12:
                outcome = "FAIL_MAX"
                i += 1
                break

            eq = day_start * (1.0 + rets[i])
            best_day = max(best_day, eq - day_start)
            if rules.trailing_basis == "eod":
                hwm = max(hwm, eq)
            if opened[i]:
                tdays += 1
            i += 1

            if eq >= 1.0 + target - 1e-12 and tdays >= rules.min_trading_days:
                if (rules.consistency_max_day_pct is not None and eq > 1.0
                        and best_day / (eq - 1.0) > rules.consistency_max_day_pct + 1e-12):
                    pass   # target hit but one day too dominant: keep trading
                else:
                    outcome = "PASS"
                    break
            if rules.max_days is not None and d >= rules.max_days:
                outcome = "FAIL_TIME"
                break

        phases.append({"outcome": outcome, "days": d,
                       "trading_days": tdays, "equity": eq})
        if outcome != "PASS":
            break

    if len(phases) == len(targets) and all(p["outcome"] == "PASS" for p in phases):
        overall = "PASS"
    else:
        overall = phases[-1]["outcome"]
    return {"outcome": overall, "phases": phases, "total_days": total, "end_row": i}


def _arrays(days):
    maxs = days["max"].values if "max" in days else np.maximum(days["ret"].values, 0.0)
    return (days["ret"].values, days["min"].values, maxs, days["opened"].values)


def run_challenge(days, rules, start=0, horizon=None):
    """Simulate one challenge from row `start` of a daily_table() frame."""
    return _run(*_arrays(days), rules, start, horizon)


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
    """Start a challenge on every historical day (stride `step`); tally outcomes."""
    arrays = _arrays(days)
    outcomes = [_run(*arrays, rules, s, horizon) for s in range(0, len(days), step)]
    return _summarize(outcomes)


def bootstrap_pass_rate(days, rules, n_sims=1000, horizon=365, block=10, seed=None):
    """
    Moving-block bootstrap of the day table: `n_sims` synthetic sequences of
    `horizon` days built from random contiguous blocks of `block` real days.
    """
    rng = np.random.default_rng(seed)
    rets, mins, maxs, opened = _arrays(days)
    n = len(rets)
    block = min(block, n)
    outcomes = []
    for _ in range(n_sims):
        idx = []
        while len(idx) < horizon:
            s = rng.integers(0, n - block + 1)
            idx.extend(range(s, s + block))
        idx = np.asarray(idx[:horizon])
        outcomes.append(_run(rets[idx], mins[idx], maxs[idx], opened[idx], rules, 0, horizon))
    return _summarize(outcomes)


def evaluate(days, rules, horizon=365, n_sims=1000, seed=0):
    """Historical + bootstrap summaries plus fee economics, in one dict."""
    h = historical_pass_rate(days, rules, horizon)
    b = bootstrap_pass_rate(days, rules, n_sims, horizon, seed=seed)
    p = b["p_pass"] if not np.isnan(b["p_pass"]) else h["p_pass"]
    econ = {}
    if rules.fee > 0 and not np.isnan(p):
        econ = {
            "expected_attempts": (1.0 / p) if p > 0 else np.inf,
            "expected_fees_to_pass": (rules.fee / p) if p > 0 else np.inf,
            "funded_capital_per_fee": p * rules.account_size / rules.fee,
        }
    return {"historical": h, "bootstrap": b, "economics": econ,
            "rules": rules.name, "horizon": horizon}


# ---------------------------------------------------------------------------
# Leverage sweep -- the "trade smaller, take longer" trade-off
# ---------------------------------------------------------------------------

def sweep_leverage(signal, prices, index, rules, leverages, costs=None,
                   horizon=365, n_sims=1000, seed=0, spread_series=None):
    rows = []
    for lev in leverages:
        eq, opened = equity_path(signal, prices, lev, costs, spread_series)
        days = daily_table(eq, opened, index, rules.daily_reset_tz)
        h = historical_pass_rate(days, rules, horizon)
        b = bootstrap_pass_rate(days, rules, n_sims, horizon, seed=seed)
        rows.append({
            "leverage": lev,
            "hist_p_pass": h["p_pass"], "hist_p_fail_daily": h["p_fail_daily"],
            "hist_p_fail_max": h["p_fail_max"], "hist_p_incomplete": h["p_incomplete"],
            "hist_median_days": h["median_days_to_pass"],
            "boot_p_pass": b["p_pass"], "boot_p_fail_daily": b["p_fail_daily"],
            "boot_p_fail_max": b["p_fail_max"], "boot_median_days": b["median_days_to_pass"],
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
    print(f"  median days:     {md:.0f}" if not np.isnan(md) else "  median days:     n/a")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from functools import partial
    from data_loader import load
    from strategies_library import STRATEGIES, opt_for_walkforward, signal_for_walkforward
    from mcpt import walkforward_signal

    df = load("BTC")
    spec = STRATEGIES["momentum"]
    wf_opt = partial(opt_for_walkforward, optimize=spec["optimize"])
    wf_sig = partial(signal_for_walkforward, signal=spec["signal"])
    TRAIN_BARS, TRAIN_STEP = 1000, 126
    sig = walkforward_signal(df, wf_opt, wf_sig, TRAIN_BARS, TRAIN_STEP)
    oos = df.iloc[TRAIN_BARS:]
    sig_oos = sig[TRAIN_BARS:]
    costs = CostModel(spread_frac=0.0005)

    print("Presets available:", ", ".join(load_presets()))
    for name in ("FTMO 2-Step", "FTMO 1-Step"):
        rules = get_rules(name)
        eq, opened = equity_path(sig_oos * 0.25, oos["Close"].values, 1.0, costs)
        days = daily_table(eq, opened, oos.index, rules.daily_reset_tz)
        res = evaluate(days, rules, horizon=365, n_sims=500)
        print(f"\n{name} (momentum on BTC at 0.25x):")
        print_summary("bootstrap 365d", res["bootstrap"])
        if res["economics"]:
            e = res["economics"]
            print(f"  expected fees to pass: {e['expected_fees_to_pass']:,.0f}  "
                  f"funded capital per fee unit: {e['funded_capital_per_fee']:.0f}")
