"""
The gate: everything one strategy must go through before it enters the bank.

    result = run_gate("momentum", df, instrument, rules, cfg)
    result["validation"]  -> numbers the criteria are checked against
    result["strategy"]    -> optimizer Strategy (deals) for the bank / builder

Steps (Tier A, a Python strategy from the registry):
  1. dev / lockbox split (the last `lockbox_years` are the lockbox)
  2. in-sample optimization on the first `train_frac` of the dev data
  3. in-sample permutation test (parallel)                -> is_p
  4. walk-forward over the dev data, OOS profit factor    -> wf_pf
  5. walk-forward permutation test                        -> wf_p
  6. lockbox: the walk-forward continued into the lockbox, scored once
  7. optional cross-market check on extra datasets        -> markets_passed
  8. prop-firm simulation of the vol-targeted OOS signal  -> p_pass ...
  9. the OOS signal converted to deals for the optimizer

Tier B (an MT5 report): steps 8 only, from the report's trade list.

The multiple-system correction (corrected_p) is added by pipeline.py,
because it needs all candidates of a run at once.
"""

import os
import sys
import time
from functools import partial

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data_loader import load, load_mt5, load_ticks, resample_ohlc, broker_time_to_utc
from strategies_library import (STRATEGIES, opt_for_insample, opt_for_walkforward,
                                signal_for_walkforward, load_user_strategies)
from objectives import profit_factor, sharpe_ratio, get_strategy_returns
from mcpt import walkforward_signal
from mcpt_fast import insample_mcpt_parallel, walkforward_mcpt_parallel
from sizing import vol_target, target_vol_for_daily_limit
from propfirm import CostModel, equity_path, daily_table, daily_table_from_pnl, evaluate
from to_optimizer import signal_to_strategy

DEFAULT_GATE = dict(
    train_frac=0.7,        # in-sample share of the dev data
    train_bars=1000,       # walk-forward training window (bars)
    train_step=126,        # bars between re-optimizations
    is_perms=300,          # in-sample permutations
    wf_perms=200,          # walk-forward permutations
    xm_perms=100,          # cross-market in-sample permutations
    horizon=365,           # prop-firm bootstrap horizon (days)
    n_sims=1000,           # prop-firm bootstrap simulations
    target_vol_sigmas=4.0, # daily-loss limit = this many daily sigmas
    vol_lookback=20,
    max_leverage=3.0,
    cost_stress=1.0,       # multiply costs (2.0 / 3.0 for stress runs)
    n_jobs=None,
    seed=42,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(spec):
    """
    spec examples:
      {"name": "BTC"}
      {"mt5": "data/raw/MES_continuous.csv", "tz": "UTC", "timeframe": "1h"}
      {"mt5": "data/raw/EURUSD_M1_2023.csv", "tz": "broker", "timeframe": "4h"}
      {"ticks": "data/raw/EURUSD_ftmo.csv", "timeframe": "1h"}
    Optional "since"/"until" (ISO dates) slice the result. Paths are relative
    to python/ unless absolute. Returns OHLC(+Spread) with a UTC index.
    """
    if "name" in spec:
        df = load(spec["name"])
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    elif "mt5" in spec:
        path = spec["mt5"] if os.path.isabs(spec["mt5"]) else os.path.join(HERE, spec["mt5"])
        tz = spec.get("tz", "UTC")
        if tz == "broker":
            df = load_mt5(path, tz="UTC")
            df.index = broker_time_to_utc(df.index.tz_localize(None), "broker")
            df = df[~df.index.isna()]
        else:
            df = load_mt5(path, tz=tz)
    elif "ticks" in spec:
        path = spec["ticks"] if os.path.isabs(spec["ticks"]) else os.path.join(HERE, spec["ticks"])
        df = load_ticks(path, tz=spec.get("tz", "broker"), verbose=False)
    else:
        raise ValueError(f"dataset spec needs 'name', 'mt5' or 'ticks': {spec}")

    tf = spec.get("timeframe")
    if tf and tf.lower() not in ("1min", "native"):
        df = resample_ohlc(df, tf)
    if spec.get("since"):
        df = df[df.index >= pd.Timestamp(spec["since"], tz="UTC")]
    if spec.get("until"):
        df = df[df.index < pd.Timestamp(spec["until"], tz="UTC")]
    return df


def dataset_label(spec):
    tf = spec.get("timeframe", "native")
    src = spec.get("name") or os.path.basename(spec.get("mt5") or spec.get("ticks") or "?")
    return f"{src}@{tf}"


def split_dev_lockbox(df, lockbox_years=0.0):
    """Last `lockbox_years` of bars become the lockbox."""
    if not lockbox_years:
        return df, df.iloc[0:0]
    cut = df.index[-1] - pd.DateOffset(days=int(365.25 * lockbox_years))
    return df[df.index < cut], df[df.index >= cut]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _funcs(spec):
    return (partial(opt_for_insample, optimize=spec["optimize"]),
            partial(opt_for_walkforward, optimize=spec["optimize"]),
            partial(signal_for_walkforward, signal=spec["signal"]))


def _windows(n_bars, cfg):
    tb = min(cfg["train_bars"], max(50, int(n_bars * 0.5)))
    ts = max(10, min(cfg["train_step"], tb // 4))
    return tb, ts


def _cost_inputs(df, instrument, cfg):
    """(CostModel, per-bar spread series or None) for the prop-firm stage."""
    price = float(np.median(df["Close"].values))
    base = instrument.cost_frac(price)
    costs = CostModel(spread_frac=base, stress=cfg["cost_stress"])
    spread_series = None
    if "Spread" in df.columns:
        # tick-derived bars: Spread in price units -> fraction, plus commission
        comm = 2.0 * instrument.commission / instrument.notional(price)
        spread_series = (df["Spread"].values / df["Close"].values) + comm
    return costs, spread_series


# ---------------------------------------------------------------------------
# Tier A gate
# ---------------------------------------------------------------------------

def run_gate(strategy_name, df, instrument, rules, cfg=None, timeframe="1D",
             extra_markets=None, verbose=True):
    """Full gate for one registered strategy on one dataset."""
    load_user_strategies()
    cfg = {**DEFAULT_GATE, **(cfg or {})}
    spec = STRATEGIES[strategy_name]
    is_opt, wf_opt, wf_sig = _funcs(spec)
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)

    ppy = instrument.bars_per_year(timeframe)
    dev, lockbox = split_dev_lockbox(df, cfg.get("lockbox_years", 0.0))
    n = len(dev)
    split = int(n * cfg["train_frac"])
    train = dev.iloc[:split].reset_index(drop=True)
    tb, ts = _windows(n, cfg)
    v = {"strategy": strategy_name, "kind": spec.get("kind", ""), "bars_dev": n,
         "bars_lockbox": len(lockbox), "train_bars": tb, "train_step": ts}

    # 2. in-sample optimization
    params, is_pf, _ = spec["optimize"](train["Close"].values)
    v["best_params"] = str(params)
    v["is_pf"] = float(is_pf)
    say(f"  [{strategy_name}] in-sample PF {is_pf:.3f} params {params}")

    # 3. in-sample MCPT
    r = insample_mcpt_parallel(train, is_opt, profit_factor, n_permutations=cfg["is_perms"],
                               seed=cfg["seed"], n_jobs=cfg["n_jobs"], verbose=False)
    v["is_p"] = float(r["p_value"])
    say(f"  [{strategy_name}] in-sample p {v['is_p']:.4f}")

    # 4-6. walk-forward over dev + lockbox (the lockbox is never trained on ahead of time)
    full = pd.concat([dev, lockbox])
    wf = walkforward_signal(full, wf_opt, wf_sig, tb, ts)
    close = full["Close"].values
    rets = get_strategy_returns(wf, close)
    dev_oos = slice(tb, n)
    v["wf_pf"] = float(profit_factor(rets[dev_oos]))
    v["sharpe"] = float(sharpe_ratio(rets[dev_oos], ppy))
    eq = np.cumsum(rets[dev_oos])
    dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
    v["return_dd"] = float(eq[-1] / dd) if len(eq) and dd > 0 else 0.0
    say(f"  [{strategy_name}] walk-forward PF {v['wf_pf']:.3f}  Sharpe {v['sharpe']:.2f}")

    r = walkforward_mcpt_parallel(dev, wf_opt, wf_sig, profit_factor, tb, ts,
                                  n_permutations=cfg["wf_perms"], seed=cfg["seed"] + 1,
                                  n_jobs=cfg["n_jobs"], verbose=False)
    v["wf_p"] = float(r["p_value"])
    say(f"  [{strategy_name}] walk-forward p {v['wf_p']:.4f}")

    if len(lockbox):
        v["lockbox_pf"] = float(profit_factor(rets[n:]))
        say(f"  [{strategy_name}] LOCKBOX PF {v['lockbox_pf']:.3f} (single look)")
    else:
        v["lockbox_pf"] = None

    # 7. cross-market
    if extra_markets:
        passed, detail = 0, {}
        for label, xdf in extra_markets.items():
            xn = len(xdf)
            xtrain = xdf.iloc[:int(xn * cfg["train_frac"])].reset_index(drop=True)
            xr = insample_mcpt_parallel(xtrain, is_opt, profit_factor,
                                        n_permutations=cfg["xm_perms"], seed=cfg["seed"],
                                        n_jobs=cfg["n_jobs"], verbose=False)
            xtb, xts = _windows(xn, cfg)
            xwf = walkforward_signal(xdf, wf_opt, wf_sig, xtb, xts)
            xpf = float(profit_factor(get_strategy_returns(xwf, xdf["Close"].values)[xtb:]))
            ok = xr["p_value"] < 0.05 and xpf > 1.0
            passed += int(ok)
            detail[label] = {"is_p": float(xr["p_value"]), "wf_pf": xpf, "pass": ok}
        v["markets_passed"] = passed
        v["markets"] = detail
        say(f"  [{strategy_name}] cross-market: {passed}/{len(extra_markets)} pass")
    else:
        v["markets_passed"] = None

    # 8. prop-firm simulation on the OOS signal (dev OOS + lockbox), vol-targeted
    oos = full.iloc[tb:]
    sig_oos = wf[tb:]
    daily_limit = rules.max_daily_loss if rules.max_daily_loss > 0 else 0.05
    tv = target_vol_for_daily_limit(daily_limit, cfg["target_vol_sigmas"],
                                    instrument.bars_per_year("1D"))
    sized = vol_target(sig_oos, oos["Close"].values, tv, lookback=cfg["vol_lookback"],
                       periods_per_year=ppy, max_leverage=cfg["max_leverage"])
    costs, spread_series = _cost_inputs(oos, instrument, cfg)
    equity, opened = equity_path(sized, oos["Close"].values, 1.0, costs, spread_series)
    days = daily_table(equity, opened, oos.index, rules.daily_reset_tz)
    pf_res = evaluate(days, rules, horizon=cfg["horizon"], n_sims=cfg["n_sims"], seed=cfg["seed"])
    b = pf_res["bootstrap"]
    v.update({"target_vol": float(tv), "p_pass": float(b["p_pass"]),
              "p_fail_daily": float(b["p_fail_daily"]), "p_fail_max": float(b["p_fail_max"]),
              "median_days": (None if np.isnan(b["median_days_to_pass"]) else float(b["median_days_to_pass"])),
              "p_pass_historical": float(pf_res["historical"]["p_pass"]),
              "economics": pf_res["economics"], "rules": rules.name})
    say(f"  [{strategy_name}] {rules.name}: P(pass) {v['p_pass']:.1%}  "
        f"fail daily {v['p_fail_daily']:.1%}  fail max {v['p_fail_max']:.1%}")

    # 9. deals for the optimizer (account-sized)
    strategy = signal_to_strategy(
        f"{strategy_name} {timeframe} WF", instrument.symbol, timeframe, sized,
        oos["Close"].values, oos.index, notional=rules.account_size,
        initial_deposit=rules.account_size, spread_frac=costs.spread_frac * cfg["cost_stress"],
        file_path=f"gate:{strategy_name}:{instrument.symbol}:{timeframe}")
    v["trades"] = int(strategy.metrics.total_trades)
    v["seconds"] = round(time.time() - t0, 1)
    return {"validation": v, "strategy": strategy, "signal_oos": sized,
            "oos_index": oos.index, "days": days}


# ---------------------------------------------------------------------------
# Tier B: MT5 report -> prop-firm simulation only
# ---------------------------------------------------------------------------

def run_tier_b(html_path, rules, cfg=None, verbose=True):
    """Prop-firm + Monte Carlo evaluation of an MT5 report's trade list."""
    cfg = {**DEFAULT_GATE, **(cfg or {})}
    opt_dir = os.path.join(os.path.dirname(HERE), "portfolio_optimizer")
    if opt_dir not in sys.path:
        sys.path.insert(0, opt_dir)
    from mt5_parser import MT5Parser
    s = MT5Parser().parse_file(html_path)
    outs = [t for t in s.trades if t.direction.lower() in ("out", "in/out")]
    if not outs:
        raise ValueError(f"no closed deals in {html_path}")
    # Scale the report's P/L to the firm's account size using the report's deposit.
    scale = rules.account_size / s.metrics.initial_deposit if s.metrics.initial_deposit else 1.0
    days = daily_table_from_pnl([t.time for t in outs], [t.profit * scale for t in outs],
                                rules.account_size, tz=rules.daily_reset_tz)
    res = evaluate(days, rules, horizon=cfg["horizon"], n_sims=cfg["n_sims"], seed=cfg["seed"])
    b = res["bootstrap"]
    m = s.metrics
    v = {"strategy": s.name, "tier": "B", "trades": int(m.total_trades),
         "is_pf": float(m.profit_factor), "wf_pf": None, "is_p": None, "wf_p": None,
         "lockbox_pf": None, "sharpe": float(m.sharpe_ratio),
         "return_dd": float(m.total_net_profit / m.equity_dd_maximal) if m.equity_dd_maximal else 0.0,
         "p_pass": float(b["p_pass"]), "p_fail_daily": float(b["p_fail_daily"]),
         "p_fail_max": float(b["p_fail_max"]),
         "median_days": (None if np.isnan(b["median_days_to_pass"]) else float(b["median_days_to_pass"])),
         "p_pass_historical": float(res["historical"]["p_pass"]), "economics": res["economics"],
         "rules": rules.name, "scale_to_account": scale, "markets_passed": None}
    if verbose:
        print(f"  [B] {s.name}: PF {m.profit_factor:.2f}, {rules.name} P(pass) {v['p_pass']:.1%}")
    return {"validation": v, "strategy": s, "days": days}
