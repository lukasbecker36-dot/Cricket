"""Minimal Telegram bot client for sending alerts and (optionally) reading commands."""
from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)


class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api = f"https://api.telegram.org/bot{bot_token}"
        self._last_update_id = 0
        self._session = requests.Session()

    def send(self, text: str, parse_mode: str = "Markdown", silent: bool = False) -> None:
        url = f"{self.api}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text[:4000],  # Telegram limit
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        try:
            r = self._session.post(url, json=payload, timeout=15)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning("telegram send failed: %s", e)

    def get_updates(self, timeout_s: int = 0) -> list[dict]:
        """Long-poll for messages addressed to the bot.

        Returns list of update objects since the last seen update_id.
        """
        url = f"{self.api}/getUpdates"
        params = {"offset": self._last_update_id + 1, "timeout": timeout_s}
        try:
            r = self._session.get(url, params=params, timeout=timeout_s + 10)
            r.raise_for_status()
            j = r.json()
        except requests.RequestException as e:
            logger.warning("telegram getUpdates failed: %s", e)
            return []
        updates = j.get("result", [])
        for u in updates:
            self._last_update_id = max(self._last_update_id, u.get("update_id", 0))
        # Only return messages from our chat
        return [u for u in updates
                if str(u.get("message", {}).get("chat", {}).get("id", "")) == str(self.chat_id)]
