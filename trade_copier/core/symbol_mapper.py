"""Symbol mapping between different brokers/platforms."""

import logging
from typing import Dict, Optional, List
import re

logger = logging.getLogger("trade_copier")


# Common symbol aliases across brokers
COMMON_ALIASES = {
    # Metals
    "GOLD": ["XAUUSD", "XAU/USD", "XAUUSD.pro", "GOLD.pro"],
    "SILVER": ["XAGUSD", "XAG/USD", "XAGUSD.pro"],
    "XAUUSD": ["GOLD", "XAU/USD"],
    "XAGUSD": ["SILVER", "XAG/USD"],

    # Indices
    "US30": ["DJ30", "DJI", "DOW", "US30.pro"],
    "US100": ["NAS100", "USTEC", "NDX", "US100.pro"],
    "US500": ["SPX500", "SP500", "SPX", "US500.pro"],
    "GER40": ["DAX40", "DE40", "GER30", "DAX"],
    "UK100": ["FTSE100", "FTSE", "UK100.pro"],
    "JP225": ["NIKKEI", "JPN225", "NI225"],

    # Oil
    "USOIL": ["WTI", "CRUDEOIL", "OIL", "USOIL.pro", "XTIUSD"],
    "UKOIL": ["BRENT", "BRENTOIL", "XBRUSD"],

    # Crypto
    "BTCUSD": ["BITCOIN", "BTC/USD", "BTCUSD.pro"],
    "ETHUSD": ["ETHEREUM", "ETH/USD", "ETHUSD.pro"],
}


class SymbolMapper:
    """
    Maps symbols between master and slave accounts.

    Handles:
    - Direct symbol mapping (GOLD -> XAUUSD)
    - Suffix addition (EURUSD -> EURUSD.pro)
    - Auto-mapping based on common aliases
    """

    def __init__(
        self,
        custom_mapping: Dict[str, str] = None,
        suffix: str = "",
        auto_map: bool = True
    ):
        """
        Initialize symbol mapper.

        Args:
            custom_mapping: Dict of master_symbol -> slave_symbol
            suffix: Suffix to add to symbols (e.g., ".pro")
            auto_map: Whether to use automatic symbol mapping
        """
        self.custom_mapping = custom_mapping or {}
        self.suffix = suffix
        self.auto_map = auto_map
        self._reverse_mapping: Dict[str, str] = {}

        # Build reverse mapping
        for master, slave in self.custom_mapping.items():
            self._reverse_mapping[slave] = master

    def map_symbol(self, master_symbol: str) -> str:
        """
        Map a master symbol to slave symbol.

        Args:
            master_symbol: Symbol from master account

        Returns:
            Mapped symbol for slave account
        """
        # Clean the symbol (remove any existing suffix)
        clean_symbol = self._clean_symbol(master_symbol)

        # 1. Check custom mapping first
        if clean_symbol in self.custom_mapping:
            mapped = self.custom_mapping[clean_symbol]
            logger.debug(f"Symbol mapped via custom mapping: {master_symbol} -> {mapped}")
            return mapped

        # Also check with original symbol (including suffix)
        if master_symbol in self.custom_mapping:
            mapped = self.custom_mapping[master_symbol]
            logger.debug(f"Symbol mapped via custom mapping: {master_symbol} -> {mapped}")
            return mapped

        # 2. Try auto-mapping based on common aliases
        if self.auto_map:
            mapped = self._auto_map(clean_symbol)
            if mapped:
                # Add suffix if needed
                if self.suffix and not mapped.endswith(self.suffix):
                    mapped = mapped + self.suffix
                logger.debug(f"Symbol auto-mapped: {master_symbol} -> {mapped}")
                return mapped

        # 3. Just add suffix to original symbol
        if self.suffix:
            mapped = clean_symbol + self.suffix
            logger.debug(f"Symbol suffixed: {master_symbol} -> {mapped}")
            return mapped

        # 4. Return as-is
        return clean_symbol

    def reverse_map(self, slave_symbol: str) -> str:
        """
        Map a slave symbol back to master symbol.

        Args:
            slave_symbol: Symbol from slave account

        Returns:
            Original master symbol
        """
        clean = self._clean_symbol(slave_symbol)

        # Check reverse mapping
        if slave_symbol in self._reverse_mapping:
            return self._reverse_mapping[slave_symbol]
        if clean in self._reverse_mapping:
            return self._reverse_mapping[clean]

        return clean

    def _clean_symbol(self, symbol: str) -> str:
        """Remove common suffixes from symbol."""
        # Common suffixes to remove
        suffixes = ['.pro', '.c', '.r', '.m', '_SB', '.std', '.raw']

        clean = symbol
        for suffix in suffixes:
            if clean.lower().endswith(suffix.lower()):
                clean = clean[:-len(suffix)]
                break

        return clean.upper()

    def _auto_map(self, symbol: str) -> Optional[str]:
        """
        Try to auto-map a symbol using common aliases.

        Args:
            symbol: Clean symbol to map

        Returns:
            Mapped symbol or None
        """
        symbol_upper = symbol.upper()

        # Check if this symbol has known aliases
        if symbol_upper in COMMON_ALIASES:
            # Return the first alias (most common)
            return COMMON_ALIASES[symbol_upper][0]

        # Check if this symbol IS an alias for something
        for base, aliases in COMMON_ALIASES.items():
            if symbol_upper in [a.upper() for a in aliases]:
                return base

        return None

    def add_mapping(self, master_symbol: str, slave_symbol: str):
        """Add a custom symbol mapping."""
        self.custom_mapping[master_symbol] = slave_symbol
        self._reverse_mapping[slave_symbol] = master_symbol
        logger.info(f"Added symbol mapping: {master_symbol} -> {slave_symbol}")

    def is_symbol_allowed(
        self,
        symbol: str,
        whitelist: List[str] = None,
        blacklist: List[str] = None
    ) -> bool:
        """
        Check if a symbol is allowed based on whitelist/blacklist.

        Args:
            symbol: Symbol to check
            whitelist: If provided, only these symbols are allowed
            blacklist: These symbols are blocked

        Returns:
            True if symbol is allowed
        """
        clean = self._clean_symbol(symbol)

        # Blacklist takes priority
        if blacklist:
            for blocked in blacklist:
                if self._clean_symbol(blocked) == clean:
                    return False

        # If whitelist is set, symbol must be in it
        if whitelist:
            for allowed in whitelist:
                if self._clean_symbol(allowed) == clean:
                    return True
            return False

        return True
