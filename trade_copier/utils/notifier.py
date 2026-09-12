"""Notification system for Trade Copier."""

import logging
from typing import Optional
import requests

logger = logging.getLogger("trade_copier")


class Notifier:
    """Send notifications via Discord, Telegram, etc."""

    def __init__(
        self,
        discord_webhook: str = None,
        telegram_bot_token: str = None,
        telegram_chat_id: str = None
    ):
        self.discord_webhook = discord_webhook
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id

    def send(self, message: str, level: str = "info"):
        """
        Send notification to all configured channels.

        Args:
            message: Message to send
            level: Notification level (info, warning, error)
        """
        if self.discord_webhook:
            self._send_discord(message, level)

        if self.telegram_bot_token and self.telegram_chat_id:
            self._send_telegram(message)

    def _send_discord(self, message: str, level: str = "info"):
        """Send notification to Discord webhook."""
        if not self.discord_webhook:
            return

        # Color based on level
        colors = {
            "info": 3447003,      # Blue
            "warning": 16776960,  # Yellow
            "error": 15158332,    # Red
            "success": 3066993   # Green
        }

        payload = {
            "embeds": [{
                "title": "Trade Copier Notification",
                "description": message,
                "color": colors.get(level, 3447003)
            }]
        }

        try:
            response = requests.post(
                self.discord_webhook,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

    def _send_telegram(self, message: str):
        """Send notification to Telegram."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return

        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": f"🤖 Trade Copier\n\n{message}",
            "parse_mode": "HTML"
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    def trade_opened(self, symbol: str, direction: str, volume: float, slave_name: str):
        """Notify about opened trade."""
        emoji = "🟢" if direction.upper() == "BUY" else "🔴"
        message = f"{emoji} **Trade Opened**\n"
        message += f"Symbol: {symbol}\n"
        message += f"Direction: {direction}\n"
        message += f"Volume: {volume}\n"
        message += f"Account: {slave_name}"
        self.send(message, "success")

    def trade_closed(self, symbol: str, profit: float, slave_name: str):
        """Notify about closed trade."""
        emoji = "💰" if profit >= 0 else "💸"
        level = "success" if profit >= 0 else "warning"
        message = f"{emoji} **Trade Closed**\n"
        message += f"Symbol: {symbol}\n"
        message += f"Profit: {profit:+.2f}\n"
        message += f"Account: {slave_name}"
        self.send(message, level)

    def error(self, error_message: str):
        """Notify about an error."""
        message = f"⚠️ **Error**\n{error_message}"
        self.send(message, "error")
