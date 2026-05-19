"""Entry point for the live signal runner.

Usage:
  LIVE_DRY_RUN=1 python -m scripts.live_signals    # test, no telegram
  python -m scripts.live_signals                   # real
"""
from __future__ import annotations

from src.live.config import LiveConfig
from src.live.runner import Runner
from src.logging_setup import configure_logging


def main() -> int:
    configure_logging("INFO")
    cfg = LiveConfig.from_env()
    return Runner(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
