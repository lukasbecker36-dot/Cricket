"""Run the match bot: auto par-line projections around real fixtures + trade log.

Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, CRICKET_API_KEY.

Loop: handle incoming Telegram messages continuously; poll watched matches'
state on a slow cadence (default 5 min) to respect the API daily quota.
Add a match with the Telegram command:  watch <cricket_data_match_id>
(find IDs via the Cricket Data currentMatches endpoint / the bot's /fixtures).
"""
from __future__ import annotations

import logging
import os
import time

from src.live.cricket_data import CricketDataClient
from src.live.match_bot import MatchBot
from src.live.telegram import TelegramClient, extract_text_and_photo
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)
POLL_INTERVAL_S = 300  # 5 min — frugal on the free-tier daily quota


def main() -> int:
    configure_logging()
    tg = TelegramClient(os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"])
    api = CricketDataClient()  # reads CRICKET_API_KEY
    bot = MatchBot(tg, api)
    tg.send("🏏 match bot online. Commands: watch <id> | trade ... | inn1 <score> | status | help")

    last_poll = 0.0
    while True:
        # 1. drain incoming messages (long-poll up to 25s)
        try:
            for update in tg.get_updates(timeout_s=25):
                text, _photo = extract_text_and_photo(update)
                if text:
                    try:
                        bot.handle_message(text)
                    except Exception as e:
                        logger.exception("handle_message failed")
                        tg.send(f"⚠️ error handling that: {e}")
        except Exception as e:
            logger.warning("get_updates failed: %s", e)
            time.sleep(5)

        # 2. poll match state on a slow cadence
        now = time.time()
        if now - last_poll >= POLL_INTERVAL_S:
            try:
                bot.poll()
            except Exception as e:
                logger.warning("poll failed: %s", e)
            last_poll = now


if __name__ == "__main__":
    raise SystemExit(main())
