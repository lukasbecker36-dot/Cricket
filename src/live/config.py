"""Runtime configuration loaded from environment variables.

Required env vars (set via systemd EnvironmentFile=/etc/cricket/live.env):
  BETFAIR_APP_KEY
  BETFAIR_USERNAME
  BETFAIR_PASSWORD
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LiveConfig:
    betfair_app_key: str
    betfair_username: str
    betfair_password: str
    telegram_bot_token: str
    telegram_chat_id: str
    model_dir: Path = Path("models")
    poll_interval_s: int = 60
    pre_match_window_min: int = 15   # alert when match starts within this window
    post_match_grace_min: int = 5    # stop alerting this long after start
    edge_threshold: float = -0.03
    implied_min: float = 0.10
    implied_max: float = 0.90
    log_dir: Path = Path("data/live/signals")
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> "LiveConfig":
        def need(k: str) -> str:
            v = os.environ.get(k)
            if not v:
                raise RuntimeError(f"missing env var {k}")
            return v

        return cls(
            betfair_app_key=need("BETFAIR_APP_KEY"),
            betfair_username=need("BETFAIR_USERNAME"),
            betfair_password=need("BETFAIR_PASSWORD"),
            telegram_bot_token=need("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=need("TELEGRAM_CHAT_ID"),
            dry_run=os.environ.get("LIVE_DRY_RUN", "0") == "1",
        )
