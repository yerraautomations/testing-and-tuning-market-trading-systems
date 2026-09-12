"""MatchTrader REST API adapter for prop firm trading."""

import logging
import requests
from typing import Optional, Dict, List, Any
from datetime import datetime
import time
import json

logger = logging.getLogger("trade_copier")


class MatchTraderAdapter:
    """
    Adapter for MatchTrader REST API.

    Handles authentication, position management, and order execution
    for MatchTrader-based prop firms.
    """

    def __init__(
        self,
        server_url: str,
        email: str,
        password: str,
        account_number: str = None
    ):
        """
        Initialize MatchTrader adapter.

        Args:
            server_url: MatchTrader server URL (e.g., https://platform.instantfunding.com)
            email: Account email/username
            password: Account password
            account_number: Trading account number (optional, will use first if not specified)
        """
        self.server_url = server_url.rstrip('/')
        self.email = email
        self.password = password
        self.account_number = account_number

        self._session = requests.Session()
        self._token: str = None
        self._token_expiry: datetime = None
        self._account_id: str = None
        self._connected = False

        # API endpoints (common MatchTrader endpoints)
        self._endpoints = {
            "login": "/api/auth/login",
            "refresh": "/api/auth/refresh",
            "accounts": "/api/accounts",
            "positions": "/api/trading/positions",
            "orders": "/api/trading/orders",
            "open_position": "/api/trading/positions/open",
            "close_position": "/api/trading/positions/close",
            "modify_position": "/api/trading/positions/modify",
            "symbols": "/api/trading/symbols",
            "quotes": "/api/trading/quotes",
        }

        # Set default headers
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "TradeCopier/1.0"
        })

    def connect(self) -> bool:
        """
        Connect and authenticate with MatchTrader server.

        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to MatchTrader: {self.server_url}")

            # Attempt login
            login_data = {
                "email": self.email,
                "password": self.password
            }

            response = self._session.post(
                f"{self.server_url}{self._endpoints['login']}",
                json=login_data,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                self._process_login_response(data)
                logger.info(f"Connected to MatchTrader successfully")
                return True

            elif response.status_code == 401:
                logger.error("MatchTrader authentication failed: Invalid credentials")
                return False

            else:
                logger.error(f"MatchTrader login failed: HTTP {response.status_code}")
                logger.debug(f"Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"MatchTrader connection error: {e}")
            return False

    def _process_login_response(self, data: dict):
        """Process login response and extract token/account info."""
        # Token can be in different fields depending on MatchTrader version
        self._token = (
            data.get("token") or
            data.get("accessToken") or
            data.get("access_token") or
            data.get("jwt")
        )

        if self._token:
            self._session.headers["Authorization"] = f"Bearer {self._token}"
            self._connected = True

            # Try to get account ID
            if "accounts" in data and len(data["accounts"]) > 0:
                if self.account_number:
                    # Find specific account
                    for acc in data["accounts"]:
                        if str(acc.get("accountNumber", acc.get("login", ""))) == str(self.account_number):
                            self._account_id = acc.get("id", acc.get("accountId"))
                            break
                else:
                    # Use first account
                    self._account_id = data["accounts"][0].get("id", data["accounts"][0].get("accountId"))
                    self.account_number = str(data["accounts"][0].get("accountNumber", data["accounts"][0].get("login", "")))

            logger.debug(f"Account ID: {self._account_id}, Account Number: {self.account_number}")

    def disconnect(self):
        """Disconnect from MatchTrader."""
        self._session.close()
        self._token = None
        self._connected = False
        logger.info("Disconnected from MatchTrader")

    def is_connected(self) -> bool:
        """Check if connected to MatchTrader."""
        return self._connected and self._token is not None

    def _ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if needed."""
        if not self.is_connected():
            logger.warning("MatchTrader not connected, attempting to reconnect...")
            return self.connect()
        return True

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict = None,
        params: dict = None,
        retry_count: int = 3
    ) -> Optional[dict]:
        """
        Make HTTP request to MatchTrader API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            retry_count: Number of retries on failure

        Returns:
            Response data or None
        """
        if not self._ensure_connected():
            return None

        url = f"{self.server_url}{endpoint}"

        for attempt in range(retry_count):
            try:
                if method.upper() == "GET":
                    response = self._session.get(url, params=params, timeout=30)
                elif method.upper() == "POST":
                    response = self._session.post(url, json=data, params=params, timeout=30)
                elif method.upper() == "PUT":
                    response = self._session.put(url, json=data, params=params, timeout=30)
                elif method.upper() == "DELETE":
                    response = self._session.delete(url, params=params, timeout=30)
                else:
                    logger.error(f"Unknown HTTP method: {method}")
                    return None

                if response.status_code == 200:
                    return response.json() if response.text else {}

                elif response.status_code == 401:
                    # Token expired, try to reconnect
                    logger.warning("Token expired, reconnecting...")
                    self._connected = False
                    if self.connect():
                        continue  # Retry request
                    return None

                else:
                    logger.error(f"API request failed: HTTP {response.status_code}")
                    logger.debug(f"URL: {url}")
                    logger.debug(f"Response: {response.text}")

                    if attempt < retry_count - 1:
                        time.sleep(1)
                        continue
                    return None

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                if attempt < retry_count - 1:
                    time.sleep(1)
                    continue
                return None

        return None

    def get_account_info(self) -> Optional[dict]:
        """
        Get account information.

        Returns:
            Dict with account info or None
        """
        response = self._make_request("GET", self._endpoints["accounts"])
        if response is None:
            return None

        # Handle different response formats
        if isinstance(response, list) and len(response) > 0:
            for acc in response:
                if str(acc.get("accountNumber", acc.get("login", ""))) == str(self.account_number):
                    return self._normalize_account_info(acc)
            return self._normalize_account_info(response[0])

        return self._normalize_account_info(response)

    def _normalize_account_info(self, data: dict) -> dict:
        """Normalize account info response."""
        return {
            "account_number": data.get("accountNumber", data.get("login", "")),
            "balance": float(data.get("balance", 0)),
            "equity": float(data.get("equity", 0)),
            "margin": float(data.get("usedMargin", data.get("margin", 0))),
            "free_margin": float(data.get("freeMargin", data.get("marginFree", 0))),
            "profit": float(data.get("profit", data.get("unrealizedPnl", 0))),
            "currency": data.get("currency", "USD"),
            "leverage": data.get("leverage", 100)
        }

    def get_positions(self) -> List[dict]:
        """
        Get all open positions.

        Returns:
            List of position dicts
        """
        endpoint = self._endpoints["positions"]
        if self._account_id:
            endpoint = f"{endpoint}?accountId={self._account_id}"

        response = self._make_request("GET", endpoint)
        if response is None:
            return []

        # Handle different response formats
        positions = response if isinstance(response, list) else response.get("positions", response.get("data", []))

        return [self._normalize_position(p) for p in positions]

    def _normalize_position(self, data: dict) -> dict:
        """Normalize position data."""
        return {
            "ticket": data.get("id", data.get("positionId", data.get("ticket", 0))),
            "symbol": data.get("symbol", data.get("instrument", "")),
            "type": "BUY" if data.get("side", data.get("type", "")).upper() in ["BUY", "LONG", "0"] else "SELL",
            "volume": float(data.get("volume", data.get("lots", data.get("quantity", 0)))),
            "open_price": float(data.get("openPrice", data.get("entryPrice", data.get("price", 0)))),
            "current_price": float(data.get("currentPrice", data.get("marketPrice", 0))),
            "stop_loss": float(data.get("stopLoss", data.get("sl", 0)) or 0),
            "take_profit": float(data.get("takeProfit", data.get("tp", 0)) or 0),
            "profit": float(data.get("profit", data.get("pnl", data.get("unrealizedPnl", 0)))),
            "open_time": data.get("openTime", data.get("createdAt", ""))
        }

    def open_position(
        self,
        symbol: str,
        direction: str,
        volume: float,
        stop_loss: float = 0,
        take_profit: float = 0,
        comment: str = ""
    ) -> Optional[dict]:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            direction: "BUY" or "SELL"
            volume: Position size in lots
            stop_loss: Stop loss price (0 for none)
            take_profit: Take profit price (0 for none)
            comment: Position comment

        Returns:
            Position data or None
        """
        logger.info(f"Opening position: {symbol} {direction} {volume} lots")

        order_data = {
            "symbol": symbol,
            "side": direction.upper(),
            "volume": volume,
            "type": "MARKET"
        }

        # Add account ID if available
        if self._account_id:
            order_data["accountId"] = self._account_id

        # Add SL/TP if provided
        if stop_loss > 0:
            order_data["stopLoss"] = stop_loss
        if take_profit > 0:
            order_data["takeProfit"] = take_profit
        if comment:
            order_data["comment"] = comment

        response = self._make_request("POST", self._endpoints["open_position"], data=order_data)

        if response:
            logger.info(f"Position opened successfully: {response}")
            return response
        else:
            logger.error(f"Failed to open position: {symbol} {direction} {volume}")
            return None

    def close_position(self, ticket: int, volume: float = None) -> bool:
        """
        Close a position.

        Args:
            ticket: Position ticket/ID
            volume: Volume to close (None = close all)

        Returns:
            True if successful
        """
        logger.info(f"Closing position: {ticket}")

        close_data = {
            "positionId": ticket
        }

        if self._account_id:
            close_data["accountId"] = self._account_id

        if volume is not None:
            close_data["volume"] = volume

        response = self._make_request("POST", self._endpoints["close_position"], data=close_data)

        if response is not None:
            logger.info(f"Position closed: {ticket}")
            return True
        else:
            logger.error(f"Failed to close position: {ticket}")
            return False

    def modify_position(
        self,
        ticket: int,
        stop_loss: float = None,
        take_profit: float = None
    ) -> bool:
        """
        Modify position SL/TP.

        Args:
            ticket: Position ticket/ID
            stop_loss: New stop loss (None = don't change)
            take_profit: New take profit (None = don't change)

        Returns:
            True if successful
        """
        logger.info(f"Modifying position {ticket}: SL={stop_loss}, TP={take_profit}")

        modify_data = {
            "positionId": ticket
        }

        if self._account_id:
            modify_data["accountId"] = self._account_id

        if stop_loss is not None:
            modify_data["stopLoss"] = stop_loss
        if take_profit is not None:
            modify_data["takeProfit"] = take_profit

        response = self._make_request("POST", self._endpoints["modify_position"], data=modify_data)

        if response is not None:
            logger.info(f"Position modified: {ticket}")
            return True
        else:
            logger.error(f"Failed to modify position: {ticket}")
            return False

    def get_symbols(self) -> List[dict]:
        """
        Get available trading symbols.

        Returns:
            List of symbol info dicts
        """
        response = self._make_request("GET", self._endpoints["symbols"])
        if response is None:
            return []

        symbols = response if isinstance(response, list) else response.get("symbols", response.get("data", []))
        return symbols

    def get_quote(self, symbol: str) -> Optional[dict]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote dict with bid/ask or None
        """
        response = self._make_request("GET", f"{self._endpoints['quotes']}/{symbol}")
        if response:
            return {
                "symbol": symbol,
                "bid": float(response.get("bid", 0)),
                "ask": float(response.get("ask", 0)),
                "spread": float(response.get("spread", 0))
            }
        return None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
