"""
Instrument profiles: the facts that differ between forex, futures and crypto
so one tester can serve all of them.

    from instruments import get_instrument
    mes = get_instrument("MES")
    mes.cost_frac(price)            # round-trip cost as a fraction of price
    mes.bars_per_year("1h")         # for annualizing Sharpe / vol targeting
    mes.contracts(equity, leverage, price)   # whole contracts for a target exposure

Profiles are read from config/instruments.json (edit or add there). Costs
are deliberately conservative defaults; the tick-derived Spread column, when
available, overrides them bar by bar.
"""

import json
import math
import os
from dataclasses import dataclass, asdict, replace

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


@dataclass(frozen=True)
class InstrumentProfile:
    symbol: str
    asset_class: str            # forex | futures | crypto | stock
    tick_size: float            # smallest price increment (forex: pip size)
    multiplier: float           # account currency per 1.0 price unit per contract/lot
                                #   forex: contract_size (100000 per standard lot)
                                #   futures: point value (MES 5, MNQ 2, M6E 12500)
    spread_ticks: float = 1.0   # typical spread (forex, in pips) or slippage (futures, in ticks)
    commission: float = 0.0     # per side, per contract/lot, account currency
    quote_currency: str = "USD"
    hours_per_day: float = 24.0
    days_per_week: float = 5.0
    weeks_per_year: float = 52.0
    session_tz: str = "UTC"
    quantity_step: float = 1.0  # minimum tradeable size (0.01 lot for forex, 1 contract for futures)
    notes: str = ""

    # -- economics ---------------------------------------------------------
    def notional(self, price, quantity=1.0):
        """Account-currency exposure of `quantity` contracts / lots at price."""
        return price * self.multiplier * quantity

    def cost_frac(self, price):
        """Round-trip cost as a fraction of price: spread + 2 commissions."""
        spread_cost = self.spread_ticks * self.tick_size * self.multiplier
        return (spread_cost + 2.0 * self.commission) / self.notional(price)

    def contracts(self, equity, leverage, price):
        """Whole contracts/lots giving `leverage` x equity exposure (floored)."""
        raw = leverage * equity / self.notional(price)
        return math.floor(raw / self.quantity_step) * self.quantity_step

    # -- time --------------------------------------------------------------
    def bars_per_year(self, timeframe):
        """Bars per year for a pandas-style timeframe ('1min','5min','1h','4h','1D')."""
        tf = timeframe.strip().lower()
        hours = self.hours_per_day * self.days_per_week * self.weeks_per_year
        if tf.endswith("min"):
            return hours * 60 / int(tf[:-3])
        if tf.endswith("h"):
            return hours / int(tf[:-1])
        if tf.endswith("d"):
            n = int(tf[:-1] or 1)
            return self.days_per_week * self.weeks_per_year / n
        raise ValueError(f"unknown timeframe {timeframe!r}")


def load_profiles(path=None):
    path = path or os.path.join(CONFIG_DIR, "instruments.json")
    profiles = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for item in json.load(f).get("instruments", []):
                profiles[item["symbol"]] = InstrumentProfile(**item)
    return profiles


def get_instrument(symbol, **overrides):
    """Profile by symbol (case-insensitive) with optional field overrides."""
    profiles = load_profiles()
    key = next((k for k in profiles if k.lower() == symbol.lower()), None)
    if key is None:
        raise KeyError(f"No instrument profile for {symbol!r}; add it to config/instruments.json")
    p = profiles[key]
    return replace(p, **overrides) if overrides else p


def profile_to_dict(p):
    return asdict(p)


if __name__ == "__main__":
    for sym, p in load_profiles().items():
        px = {"EURUSD": 1.10, "GBPUSD": 1.27, "USDJPY": 150.0, "MES": 5000.0, "MNQ": 18000.0,
              "M2K": 2100.0, "MYM": 40000.0, "M6E": 1.10, "M6A": 0.66, "M6B": 1.27,
              "BTC": 60000.0}.get(sym, 100.0)
        print(f"{sym:7s} {p.asset_class:8s} cost/round-trip {p.cost_frac(px)*1e4:6.2f} bp  "
              f"notional/1 {p.notional(px):>12,.0f}  H1 bars/yr {p.bars_per_year('1h'):,.0f}")
