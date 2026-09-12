"""
Bridge from the validation platform to the portfolio optimizer.

A strategy that survives the permutation / walk-forward / prop-firm tests is
just a position signal over prices. The optimizer, however, thinks in MT5
terms: a Strategy with a list of deals (in / out, with P/L) and report-style
metrics. This module converts one into the other and saves the result as a
databank that both the Streamlit app and portfolio_optimizer/headless.py load.

    from to_optimizer import signal_to_strategy, export_databank
    s = signal_to_strategy("momentum WF", "BTC", "D1", signal, close, index,
                           notional=100_000, spread_frac=0.0005)
    export_databank([s, ...], "platform_btc_demo")
    # then:  python portfolio_optimizer/headless.py --databank platform_btc_demo

Conventions (identical to propfirm.equity_path):
  * signal[t] is the position after bar t, executed at prices[t]; it earns
    the move of bar t+1. Fractional sizes (vol targeting) are fine.
  * P/L in account currency = notional * size * (exit / entry - 1).
  * Round-trip cost = notional * |size| * (spread_frac + 2 * commission_frac),
    charged on the closing deal. Position increases add to a volume-weighted
    entry price; decreases close part of the position.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OPT_DIR = os.path.join(os.path.dirname(HERE), "portfolio_optimizer")
if OPT_DIR not in sys.path:
    sys.path.insert(0, OPT_DIR)

from mt5_parser import Strategy, StrategyMetrics, Trade   # noqa: E402


def _max_drawdown(curve):
    """(max absolute drawdown, max drawdown % of the peak) of a 1-D array."""
    curve = np.asarray(curve, dtype=float)
    if len(curve) == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(curve)
    dd = peak - curve
    i = int(np.argmax(dd))
    pct = 100.0 * dd[i] / peak[i] if peak[i] > 0 else 0.0
    return float(dd[i]), float(pct)


def signal_to_strategy(name, symbol, timeframe, signal, prices, index,
                       notional=100_000.0, initial_deposit=100_000.0,
                       spread_frac=0.0, commission_frac=0.0, file_path=""):
    """
    Convert a position signal into an optimizer Strategy with MT5-style deals.

    Parameters
    ----------
    name, symbol, timeframe : labels shown in the optimizer
    signal : array of positions (+1 / -1 / 0 or fractional)
    prices : close prices aligned with signal
    index : DatetimeIndex aligned with signal (tz-aware is converted to UTC)
    notional : account currency exposed per unit of signal
    initial_deposit : starting balance for the balance / equity curves
    spread_frac, commission_frac : costs as fractions of price / notional
    file_path : optional provenance string stored on the strategy
    """
    sig = np.asarray(signal, dtype=float)
    px = np.asarray(prices, dtype=float)
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    times = idx.to_pydatetime()
    n = len(sig)
    if not (len(px) == n == len(times)):
        raise ValueError("signal, prices and index must have the same length")

    rt_cost = spread_frac + 2.0 * commission_frac      # per unit, round trip
    trades = []
    deal_id = 1
    balance = initial_deposit
    pos = 0.0
    entry = 0.0
    equity = np.empty(n)

    def deal(t, direction, size, price, profit):
        nonlocal deal_id, balance
        balance += profit
        # MT5 labels a deal by the side that executes it: closing a long is a
        # "sell" deal, closing a short is a "buy" deal.
        if direction == "in":
            side = "buy" if size > 0 else "sell"
        else:
            side = "sell" if size > 0 else "buy"
        trades.append(Trade(time=t, deal_id=deal_id, symbol=symbol,
                            trade_type=side, direction=direction,
                            volume=abs(size), price=float(price),
                            profit=float(profit), balance=float(balance)))
        deal_id += 1

    def close_part(t, size, price):
        """Close `size` units (same sign as pos) at price."""
        pnl = notional * size * (price / entry - 1.0) - notional * abs(size) * rt_cost
        deal(t, "out", size, price, pnl)

    for t in range(n):
        p = px[t]
        # mark to market before acting on this bar's signal
        equity[t] = balance + (notional * pos * (p / entry - 1.0) if pos else 0.0)
        target = sig[t] if t < n - 1 else 0.0        # flatten on the last bar
        if target == pos:
            continue
        if pos and (target == 0 or np.sign(target) != np.sign(pos)):
            close_part(times[t], pos, p)
            pos, entry = 0.0, 0.0
        if target:
            if pos == 0:
                pos, entry = target, p
                deal(times[t], "in", target, p, 0.0)
            elif abs(target) > abs(pos):
                add = target - pos
                entry = (entry * pos + p * add) / target
                pos = target
                deal(times[t], "in", add, p, 0.0)
            else:
                close_part(times[t], pos - target, p)
                pos = target
        equity[t] = balance + (notional * pos * (p / entry - 1.0) if pos else 0.0)

    outs = np.array([tr.profit for tr in trades if tr.direction == "out"])
    m = StrategyMetrics()
    m.initial_deposit = float(initial_deposit)
    m.total_net_profit = float(outs.sum()) if len(outs) else 0.0
    m.gross_profit = float(outs[outs > 0].sum()) if len(outs) else 0.0
    m.gross_loss = float(outs[outs < 0].sum()) if len(outs) else 0.0   # negative, MT5 style
    m.profit_factor = (m.gross_profit / abs(m.gross_loss) if m.gross_loss < 0
                       else (float("inf") if m.gross_profit > 0 else 0.0))
    m.total_trades = int(len(outs))
    m.profit_trades = int((outs > 0).sum()) if len(outs) else 0
    m.loss_trades = m.total_trades - m.profit_trades
    m.win_rate = 100.0 * m.profit_trades / m.total_trades if m.total_trades else 0.0
    m.expected_payoff = float(outs.mean()) if len(outs) else 0.0
    m.sharpe_ratio = (float(outs.mean() / outs.std()) if len(outs) > 1 and outs.std() > 0
                      else 0.0)                                   # MT5: per-trade
    bal_curve = initial_deposit + np.concatenate([[0.0], np.cumsum(outs)])
    m.balance_dd_maximal, m.balance_dd_maximal_pct = _max_drawdown(bal_curve)
    m.balance_dd_absolute = float(max(0.0, initial_deposit - bal_curve.min()))
    m.equity_dd_maximal, m.equity_dd_maximal_pct = _max_drawdown(equity)
    m.equity_dd_absolute = float(max(0.0, initial_deposit - equity.min()))
    m.recovery_factor = (m.total_net_profit / m.equity_dd_maximal
                         if m.equity_dd_maximal > 0 else 0.0)
    if len(outs) > 2:
        m.lr_correlation = float(np.corrcoef(np.arange(len(bal_curve)), bal_curve)[0, 1])

    return Strategy(name=name, symbol=symbol, timeframe=timeframe,
                    period_start=times[0], period_end=times[-1],
                    metrics=m, trades=trades, file_path=file_path)


def export_databank(strategies, name, locked=()):
    """Save strategies as portfolio_optimizer/databanks/<name>.json."""
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    import streamlit_app as app   # bare-mode import (no UI runs)
    return app.save_databank(name, list(strategies), set(locked))


def summarize(strategies):
    print(f"{'strategy':32s} {'trades':>6} {'net':>11} {'PF':>6} {'win%':>6} "
          f"{'eqDD':>10} {'eqDD%':>6} {'sharpe':>7}")
    for s in strategies:
        m = s.metrics
        print(f"{s.name[:32]:32s} {m.total_trades:6d} {m.total_net_profit:11,.0f} "
              f"{m.profit_factor:6.2f} {m.win_rate:6.1f} {m.equity_dd_maximal:10,.0f} "
              f"{m.equity_dd_maximal_pct:6.1f} {m.sharpe_ratio:7.2f}")


if __name__ == "__main__":
    # Demo: three walk-forward strategies on BTC -> databank -> headless run.
    from functools import partial
    sys.path.insert(0, HERE)
    from data_loader import load
    from strategies_library import (STRATEGIES, opt_for_walkforward,
                                    signal_for_walkforward)
    from mcpt import walkforward_signal
    from sizing import vol_target

    df = load("BTC")
    TB, TS = 1000, 126
    oos = df.iloc[TB:]
    built = []
    for strat in ("momentum", "donchian", "ma_cross"):
        spec = STRATEGIES[strat]
        wf = walkforward_signal(df, partial(opt_for_walkforward, optimize=spec["optimize"]),
                                partial(signal_for_walkforward, signal=spec["signal"]), TB, TS)
        # size to ~24% annualized vol, as the prop-firm layer recommends
        sized = vol_target(wf[TB:], oos["Close"].values, target_vol=0.24,
                           lookback=20, periods_per_year=365, max_leverage=2.0)
        built.append(signal_to_strategy(f"{strat} WF vol24", "BTC", "D1", sized,
                                        oos["Close"].values, oos.index,
                                        notional=100_000, spread_frac=0.0005,
                                        file_path=f"platform:{strat}"))
    summarize(built)
    path = export_databank(built, "platform_btc_demo")
    print(f"\ndatabank written: {path}")

    import headless
    headless.run(databanks=["platform_btc_demo"], name="platform_btc_demo",
                 settings=dict(min_strategies=2, max_strategies=3, max_per_symbol=3),
                 filters=dict(min_trades=10), export_top=1)
