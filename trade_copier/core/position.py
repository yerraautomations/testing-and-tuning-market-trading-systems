"""Position model and master-side change detection."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("trade_copier")


class PositionType(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Position:
    ticket: int
    symbol: str
    position_type: PositionType
    volume: float
    open_price: float
    open_time: datetime
    stop_loss: float = 0.0
    take_profit: float = 0.0
    magic_number: int = 0
    comment: str = ""
    slave_tickets: Dict[str, int] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.ticket)

    def __eq__(self, other):
        return isinstance(other, Position) and self.ticket == other.ticket

    @property
    def direction(self) -> str:
        return self.position_type.value

    def to_dict(self) -> dict:
        return {"ticket": self.ticket, "symbol": self.symbol, "type": self.position_type.value,
                "volume": self.volume, "open_price": self.open_price,
                "open_time": self.open_time.isoformat(), "stop_loss": self.stop_loss,
                "take_profit": self.take_profit, "magic_number": self.magic_number,
                "comment": self.comment, "slave_tickets": self.slave_tickets}


class PositionManager:
    """
    Tracks master positions between polls and reports what changed:
      opened   -> new tickets
      closed   -> tickets that disappeared
      reduced  -> (position, closed_volume) when the volume went down (partial close)
      modified -> SL/TP changed
    """

    def __init__(self):
        self._positions: Dict[int, Position] = {}
        self._last_update: Optional[datetime] = None

    @property
    def positions(self) -> Dict[int, Position]:
        return self._positions.copy()

    @property
    def position_count(self) -> int:
        return len(self._positions)

    def get_position(self, ticket: int) -> Optional[Position]:
        return self._positions.get(ticket)

    def seed(self, positions: List[Position]):
        """Register positions without reporting them as opened (startup)."""
        for p in positions:
            self._positions[p.ticket] = p

    def update_positions(self, current_positions: List[Position]) -> dict:
        current = {p.ticket: p for p in current_positions}
        previous = set(self._positions)
        result = {"opened": [], "closed": [], "reduced": [], "modified": []}

        for ticket, pos in current.items():
            if ticket not in previous:
                self._positions[ticket] = pos
                result["opened"].append(pos)
                logger.info(f"New position: {pos.symbol} {pos.direction} {pos.volume} (ticket {ticket})")

        for ticket in previous - set(current):
            closed = self._positions.pop(ticket)
            result["closed"].append(closed)
            logger.info(f"Position closed: {closed.symbol} (ticket {ticket})")

        for ticket, pos in current.items():
            if ticket in previous and ticket in self._positions:
                old = self._positions[ticket]
                if pos.volume < old.volume - 1e-9:
                    result["reduced"].append((pos, round(old.volume - pos.volume, 8)))
                    logger.info(f"Position reduced: {pos.symbol} {old.volume} -> {pos.volume} (ticket {ticket})")
                if old.stop_loss != pos.stop_loss or old.take_profit != pos.take_profit:
                    result["modified"].append(pos)
                    logger.info(f"Position modified: {pos.symbol} SL={pos.stop_loss} TP={pos.take_profit}")
                pos.slave_tickets = old.slave_tickets
                self._positions[ticket] = pos

        self._last_update = datetime.now()
        return result

    def add_slave_ticket(self, master_ticket: int, slave_name: str, slave_ticket):
        if master_ticket in self._positions:
            self._positions[master_ticket].slave_tickets[slave_name] = slave_ticket

    def get_slave_ticket(self, master_ticket: int, slave_name: str):
        pos = self._positions.get(master_ticket)
        return pos.slave_tickets.get(slave_name) if pos else None

    def clear(self):
        self._positions.clear()

    def get_symbols(self) -> List[str]:
        return list({p.symbol for p in self._positions.values()})
