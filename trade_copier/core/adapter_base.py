"""
The contract every slave adapter must fulfil.

The engine relies on two guarantees that the old browser adapter did not give:

  * open_position() returns the platform's position record (with its id)
    ONLY after the position is visible in get_positions(); otherwise None.
  * close_position() returns True ONLY after the position is gone (or, for a
    partial close, reduced) in get_positions().

Everything the engine does to stay consistent -- ticket mapping, restart
reconciliation, orphan detection -- depends on those two rules.

Position dicts returned by get_positions():
    {"id": str, "symbol": str, "side": "BUY"|"SELL", "volume": float,
     "open_price": float, "sl": float, "tp": float, "open_time": str}
"""

from typing import Dict, List, Optional


class SlaveAdapter:
    name: str = "adapter"

    def connect(self) -> bool:
        raise NotImplementedError

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        raise NotImplementedError

    def get_account_info(self) -> Optional[Dict]:
        """{"balance": float, "equity": float, "currency": str} or None."""
        return None

    def get_positions(self) -> List[Dict]:
        raise NotImplementedError

    def get_symbol_spec(self, symbol: str) -> Optional[Dict]:
        """{"min_volume", "volume_step", "max_volume"} or None if unknown."""
        return None

    def open_position(self, symbol: str, side: str, volume: float,
                      sl: float = 0.0, tp: float = 0.0, comment: str = "") -> Optional[Dict]:
        raise NotImplementedError

    def close_position(self, position_id: str, volume: Optional[float] = None) -> bool:
        raise NotImplementedError

    def modify_position(self, position_id: str, sl: Optional[float] = None,
                        tp: Optional[float] = None) -> bool:
        return False
