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
    telegram_bot_token: str
    telegram_chat_id: str
    model_dir: Path = Path("models")
    edge_threshold: float = -0.03
    implied_min: float = 0.10
    implied_max: float = 0.90
    log_dir: Path = Path("data/live/signals")

    @classmethod
    def from_env(cls) -> "LiveConfig":
        def need(k: str) -> str:
            v = os.environ.get(k)
            if not v:
                raise RuntimeError(f"missing env var {k}")
            return v

        return cls(
            telegram_bot_token=need("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=need("TELEGRAM_CHAT_ID"),
        )
