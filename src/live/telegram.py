"""Telegram client: send messages, long-poll for incoming, download attached photos."""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)


class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api = f"https://api.telegram.org/bot{bot_token}"
        self.file_api = f"https://api.telegram.org/file/bot{bot_token}"
        self._last_update_id = 0
        self._session = requests.Session()

    def send(self, text: str, parse_mode: str = "Markdown", silent: bool = False) -> None:
        url = f"{self.api}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text[:4000],
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        try:
            r = self._session.post(url, json=payload, timeout=15)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning("telegram send failed: %s", e)

    def send_typing(self) -> None:
        """Hint that we're processing — shows 'typing...' indicator."""
        try:
            self._session.post(
                f"{self.api}/sendChatAction",
                json={"chat_id": self.chat_id, "action": "typing"}, timeout=10,
            )
        except requests.RequestException:
            pass

    def get_updates(self, timeout_s: int = 25) -> list[dict]:
        """Long-poll for new messages addressed to the bot."""
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
        return [u for u in updates
                if str(u.get("message", {}).get("chat", {}).get("id", "")) == str(self.chat_id)]

    def download_photo(self, file_id: str) -> tuple[bytes, str] | None:
        """Resolve file_id -> path -> bytes. Returns (bytes, media_type) or None."""
        try:
            r = self._session.get(f"{self.api}/getFile", params={"file_id": file_id}, timeout=15)
            r.raise_for_status()
            file_path = r.json()["result"]["file_path"]
            r2 = self._session.get(f"{self.file_api}/{file_path}", timeout=30)
            r2.raise_for_status()
            mt = "image/jpeg"
            if file_path.lower().endswith(".png"):
                mt = "image/png"
            elif file_path.lower().endswith(".webp"):
                mt = "image/webp"
            return r2.content, mt
        except (requests.RequestException, KeyError) as e:
            logger.warning("telegram download_photo failed: %s", e)
            return None


def extract_text_and_photo(update: dict) -> tuple[str, str | None]:
    """Return (text, largest_photo_file_id_or_None) from a Telegram update."""
    msg = update.get("message", {}) or {}
    text = msg.get("text") or msg.get("caption") or ""
    photo_id: str | None = None
    if isinstance(msg.get("photo"), list) and msg["photo"]:
        photos = sorted(msg["photo"], key=lambda p: p.get("file_size", 0))
        photo_id = photos[-1]["file_id"]
    elif (doc := msg.get("document")):
        # Treat image documents as photos too (e.g. uploaded files)
        mime = doc.get("mime_type", "")
        if mime.startswith("image/"):
            photo_id = doc.get("file_id")
    return text.strip(), photo_id
