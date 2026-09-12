"""
A fake slave platform for tests and dry runs.

It behaves like a real broker as far as the engine can tell: opens return an
id only once the position exists, closes only succeed once it is gone, symbol
specs enforce minimum and step sizes, and failures can be injected.
"""

import itertools
import time
from typing import Dict, List, Optional

from .adapter_base import SlaveAdapter


class FakeAdapter(SlaveAdapter):
    def __init__(self, name="fake", specs=None, balance=100_000.0, fail_opens=0,
                 fail_closes=0, latency=0.0):
        self.name = name
        self.positions: Dict[str, Dict] = {}
        self.specs = specs or {}
        self.balance = balance
        self.fail_opens = fail_opens          # number of upcoming opens that fail
        self.fail_closes = fail_closes
        self.latency = latency
        self.calls: List[tuple] = []
        self._ids = itertools.count(1000)
        self._connected = False

    # -- lifecycle ----------------------------------------------------------
    def connect(self):
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def get_account_info(self):
        return {"balance": self.balance, "equity": self.balance, "currency": "USD"}

    # -- market -------------------------------------------------------------
    def get_positions(self):
        self.calls.append(("positions",))
        return [dict(p) for p in self.positions.values()]

    def get_symbol_spec(self, symbol):
        return self.specs.get(symbol, {"min_volume": 0.01, "volume_step": 0.01, "max_volume": 100.0})

    def open_position(self, symbol, side, volume, sl=0.0, tp=0.0, comment=""):
        self.calls.append(("open", symbol, side, volume, sl, tp))
        time.sleep(self.latency)
        if self.fail_opens > 0:
            self.fail_opens -= 1
            return None
        pid = str(next(self._ids))
        self.positions[pid] = {"id": pid, "symbol": symbol, "side": side, "volume": volume,
                               "open_price": 1.0, "sl": sl, "tp": tp,
                               "open_time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        return dict(self.positions[pid])

    def close_position(self, position_id, volume=None):
        self.calls.append(("close", position_id, volume))
        time.sleep(self.latency)
        if self.fail_closes > 0:
            self.fail_closes -= 1
            return False
        p = self.positions.get(str(position_id))
        if p is None:
            return False
        if volume is not None and volume < p["volume"] - 1e-9:
            p["volume"] = round(p["volume"] - volume, 8)
            return True
        del self.positions[str(position_id)]
        return True

    def modify_position(self, position_id, sl=None, tp=None):
        self.calls.append(("modify", position_id, sl, tp))
        p = self.positions.get(str(position_id))
        if p is None:
            return False
        if sl is not None:
            p["sl"] = sl
        if tp is not None:
            p["tp"] = tp
        return True

    # -- test helpers -------------------------------------------------------
    def inject_position(self, symbol, side, volume):
        """A position the copier did not open (an orphan)."""
        pid = str(next(self._ids))
        self.positions[pid] = {"id": pid, "symbol": symbol, "side": side, "volume": volume,
                               "open_price": 1.0, "sl": 0.0, "tp": 0.0, "open_time": ""}
        return pid
