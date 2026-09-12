"""MetaTrader 5 position monitor."""

import logging
from typing import List, Optional
from datetime import datetime
import time

logger = logging.getLogger("trade_copier")

# Import MT5 - only available on Windows
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not available. Install with: pip install MetaTrader5")

from core.position import Position, PositionType


class MT5Monitor:
    """
    Monitors positions in MetaTrader 5.

    Uses the official MetaTrader5 Python package for direct IPC
    with the MT5 terminal.
    """

    def __init__(
        self,
        terminal_path: str = None,
        account: int = None,
        password: str = None,
        server: str = None
    ):
        """
        Initialize MT5 monitor.

        Args:
            terminal_path: Path to MT5 terminal (optional, auto-detected)
            account: Account number (optional if already logged in)
            password: Account password (optional if already logged in)
            server: Broker server (optional if already logged in)
        """
        self.terminal_path = terminal_path
        self.account = account
        self.password = password
        self.server = server
        self._initialized = False
        self._last_positions: List[Position] = []

    def initialize(self) -> bool:
        """
        Initialize connection to MT5 terminal.

        Returns:
            True if successful
        """
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package not installed")
            return False

        # Build initialization kwargs
        init_kwargs = {}
        if self.terminal_path:
            init_kwargs['path'] = self.terminal_path

        # Initialize MT5
        if not mt5.initialize(**init_kwargs):
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            return False

        # Login if credentials provided
        if self.account and self.password and self.server:
            if not mt5.login(self.account, password=self.password, server=self.server):
                error = mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                mt5.shutdown()
                return False
            logger.info(f"Logged in to MT5 account {self.account}")
        else:
            # Check if already logged in
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Not logged in to MT5 and no credentials provided")
                mt5.shutdown()
                return False
            logger.info(f"Connected to existing MT5 session: Account {account_info.login}")

        self._initialized = True
        return True

    def shutdown(self):
        """Shutdown MT5 connection."""
        if MT5_AVAILABLE and self._initialized:
            mt5.shutdown()
            self._initialized = False
            logger.info("MT5 connection closed")

    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        if not MT5_AVAILABLE or not self._initialized:
            return False

        terminal_info = mt5.terminal_info()
        return terminal_info is not None and terminal_info.connected

    def get_account_info(self) -> Optional[dict]:
        """
        Get current account information.

        Returns:
            Dict with account info or None
        """
        if not self._check_connection():
            return None

        info = mt5.account_info()
        if info is None:
            return None

        return {
            "login": info.login,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "leverage": info.leverage,
            "currency": info.currency
        }

    def get_positions(self) -> List[Position]:
        """
        Get all open positions from MT5.

        Returns:
            List of Position objects
        """
        if not self._check_connection():
            return []

        mt5_positions = mt5.positions_get()
        if mt5_positions is None:
            error = mt5.last_error()
            if error[0] != 0:  # 0 = no error, just no positions
                logger.error(f"Failed to get positions: {error}")
            return []

        positions = []
        for pos in mt5_positions:
            position = Position(
                ticket=pos.ticket,
                symbol=pos.symbol,
                position_type=PositionType.BUY if pos.type == mt5.ORDER_TYPE_BUY else PositionType.SELL,
                volume=pos.volume,
                open_price=pos.price_open,
                open_time=datetime.fromtimestamp(pos.time),
                stop_loss=pos.sl,
                take_profit=pos.tp,
                magic_number=pos.magic,
                comment=pos.comment
            )
            positions.append(position)

        return positions

    def get_position_by_ticket(self, ticket: int) -> Optional[Position]:
        """
        Get a specific position by ticket.

        Args:
            ticket: Position ticket number

        Returns:
            Position object or None
        """
        if not self._check_connection():
            return None

        mt5_positions = mt5.positions_get(ticket=ticket)
        if mt5_positions is None or len(mt5_positions) == 0:
            return None

        pos = mt5_positions[0]
        return Position(
            ticket=pos.ticket,
            symbol=pos.symbol,
            position_type=PositionType.BUY if pos.type == mt5.ORDER_TYPE_BUY else PositionType.SELL,
            volume=pos.volume,
            open_price=pos.price_open,
            open_time=datetime.fromtimestamp(pos.time),
            stop_loss=pos.sl,
            take_profit=pos.tp,
            magic_number=pos.magic,
            comment=pos.comment
        )

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """
        Get positions for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of Position objects
        """
        if not self._check_connection():
            return []

        mt5_positions = mt5.positions_get(symbol=symbol)
        if mt5_positions is None:
            return []

        positions = []
        for pos in mt5_positions:
            position = Position(
                ticket=pos.ticket,
                symbol=pos.symbol,
                position_type=PositionType.BUY if pos.type == mt5.ORDER_TYPE_BUY else PositionType.SELL,
                volume=pos.volume,
                open_price=pos.price_open,
                open_time=datetime.fromtimestamp(pos.time),
                stop_loss=pos.sl,
                take_profit=pos.tp,
                magic_number=pos.magic,
                comment=pos.comment
            )
            positions.append(position)

        return positions

    def get_positions_by_magic(self, magic_number: int) -> List[Position]:
        """
        Get positions with a specific magic number.

        Args:
            magic_number: EA magic number

        Returns:
            List of Position objects
        """
        all_positions = self.get_positions()
        return [p for p in all_positions if p.magic_number == magic_number]

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Get symbol information.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with symbol info or None
        """
        if not self._check_connection():
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        return {
            "name": info.name,
            "description": info.description,
            "digits": info.digits,
            "point": info.point,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread
        }

    def _check_connection(self) -> bool:
        """Check and re-establish connection if needed."""
        if not MT5_AVAILABLE:
            return False

        if not self._initialized:
            logger.warning("MT5 not initialized, attempting to connect...")
            return self.initialize()

        if not self.is_connected():
            logger.warning("MT5 disconnected, attempting to reconnect...")
            self.shutdown()
            return self.initialize()

        return True

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
