"""Main polling loop: every poll_interval_s, scan for live opportunities and alert."""
from __future__ import annotations

import json
import logging
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .betfair_client import BetfairClient, now_utc_iso
from .config import LiveConfig
from .signals import FullInningsModel, Signal, evaluate_market, format_signal
from .telegram import TelegramClient

logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self.betfair = BetfairClient(cfg.betfair_app_key, cfg.betfair_username, cfg.betfair_password)
        self.telegram = TelegramClient(cfg.telegram_bot_token, cfg.telegram_chat_id)
        self.model = FullInningsModel(cfg.model_dir)
        self.alerted_keys: set[str] = set()
        self.last_poll_at: datetime | None = None
        self.signals_sent_today: int = 0
        self.day_marker: str = ""
        self.paused: bool = False
        self.stop_requested: bool = False
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    def _on_signal(self, signum, frame):
        logger.info("received signal %s; will stop after current iteration", signum)
        self.stop_requested = True

    def _reset_daily_counters(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.day_marker:
            self.day_marker = today
            self.signals_sent_today = 0
            # Don't reset alerted_keys -- keeps de-dup across day boundary
            # The set will be cleaned of old keys periodically
            self._gc_alerted_keys()

    def _gc_alerted_keys(self) -> None:
        # Keep set size manageable; we rely on it not growing unboundedly
        if len(self.alerted_keys) > 5000:
            self.alerted_keys.clear()
            logger.info("cleared alerted-keys de-dup set (capacity)")

    def _save_signal(self, sig: Signal) -> None:
        date = sig.detected_at_utc[:10]
        path = self.cfg.log_dir / f"{date}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(sig.__dict__) + "\n")

    def _handle_commands(self) -> None:
        for upd in self.telegram.get_updates(timeout_s=0):
            msg = (upd.get("message", {}) or {}).get("text", "").strip().lower()
            if not msg:
                continue
            if msg in ("/status", "status"):
                last = self.last_poll_at.isoformat() if self.last_poll_at else "never"
                self.telegram.send(
                    f"*Status*\n"
                    f"running: {not self.stop_requested}\n"
                    f"paused: {self.paused}\n"
                    f"last_poll: {last}\n"
                    f"signals today: {self.signals_sent_today}\n"
                    f"de-dup cache size: {len(self.alerted_keys)}"
                )
            elif msg in ("/pause", "pause"):
                self.paused = True
                self.telegram.send("Paused. Use /resume to resume.")
            elif msg in ("/resume", "resume"):
                self.paused = False
                self.telegram.send("Resumed.")
            elif msg in ("/clearcache", "clearcache"):
                self.alerted_keys.clear()
                self.telegram.send("Cleared de-dup cache.")
            else:
                self.telegram.send(
                    "Commands:\n"
                    "/status - current state\n"
                    "/pause - stop alerts\n"
                    "/resume - resume alerts\n"
                    "/clearcache - clear de-dup cache (re-alert known signals)"
                )

    def _scan_once(self) -> int:
        now = datetime.now(timezone.utc)
        from_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_iso = (now + timedelta(minutes=self.cfg.pre_match_window_min + 240)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            catalogue = self.betfair.list_market_catalogue(
                from_iso=from_iso, to_iso=to_iso,
                market_types=["INNINGS_RUNS"],
            )
        except Exception as e:
            logger.exception("listMarketCatalogue failed")
            return 0
        if not catalogue:
            return 0
        # Filter to markets starting within the alert window
        soon = []
        for m in catalogue:
            start = m.get("marketStartTime", "")
            try:
                start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            except ValueError:
                try:
                    start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            delta_min = (start_dt - now).total_seconds() / 60.0
            if -self.cfg.post_match_grace_min <= delta_min <= self.cfg.pre_match_window_min:
                if "Innings Runs" in m.get("marketName", "") and "Over" not in m.get("marketName", ""):
                    soon.append(m)
        if not soon:
            return 0
        ids = [m["marketId"] for m in soon]
        try:
            books = self.betfair.list_market_book(ids)
        except Exception:
            logger.exception("listMarketBook failed")
            return 0
        book_by_id = {b["marketId"]: b for b in books}
        n_new = 0
        season_default = datetime.now(timezone.utc).year
        for cat in soon:
            book = book_by_id.get(cat["marketId"])
            if not book:
                continue
            sigs = evaluate_market(model=self.model, market_catalogue=cat,
                                   market_book=book, season=season_default)
            for s in sigs:
                if s.key() in self.alerted_keys:
                    continue
                self.alerted_keys.add(s.key())
                self._save_signal(s)
                if self.cfg.dry_run:
                    logger.info("(dry-run) %s", s)
                elif not self.paused:
                    self.telegram.send(format_signal(s))
                    self.signals_sent_today += 1
                n_new += 1
        return n_new

    def run(self) -> int:
        self.telegram.send(
            f"Cricket live signal runner *started*\n"
            f"poll interval: {self.cfg.poll_interval_s}s  /  "
            f"window: {self.cfg.pre_match_window_min} min pre-start  /  "
            f"dry_run: {self.cfg.dry_run}"
        )
        while not self.stop_requested:
            try:
                self._reset_daily_counters()
                self._handle_commands()
                n = self._scan_once()
                self.last_poll_at = datetime.now(timezone.utc)
                if n > 0:
                    logger.info("scan: %d new signals", n)
                time.sleep(self.cfg.poll_interval_s)
            except KeyboardInterrupt:
                break
            except Exception:
                logger.exception("scan loop iteration failed; sleeping then continuing")
                time.sleep(self.cfg.poll_interval_s)
        self.telegram.send("Cricket live signal runner *stopping*.")
        return 0
