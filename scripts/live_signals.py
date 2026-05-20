"""Entry point for the chat-driven live signal helper.

Receives Betfair screenshots via Telegram, runs vision extraction (Anthropic),
applies the saved full_innings model, and replies with lay recommendations.
No Betfair API access required.

Requires env vars (see deploy/live.env.example):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ANTHROPIC_API_KEY
"""
from __future__ import annotations

import os

from src.live.chat_runner import ChatRunner
from src.live.config import LiveConfig
from src.logging_setup import configure_logging


def main() -> int:
    configure_logging("INFO")
    cfg = LiveConfig.from_env()
    return ChatRunner(cfg, anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")).run()


if __name__ == "__main__":
    raise SystemExit(main())
