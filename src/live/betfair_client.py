"""Minimal Betfair Exchange API client for the live signal layer.

Uses interactive login (username/password). Session lasts ~4 hours; we refresh
after 3 hours to be safe. For production-grade automation, switch to the
non-interactive cert login (see Betfair docs).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

LOGIN_URL = "https://identitysso.betfair.com/api/login"
EXCHANGE_BASE = "https://api.betfair.com/exchange/betting/rest/v1.0"
CRICKET_EVENT_TYPE_ID = "4"
INTERESTING_MARKET_TYPES = ["MATCH_ODDS", "INNINGS_RUNS"]


class BetfairClient:
    def __init__(self, app_key: str, username: str, password: str):
        self.app_key = app_key
        self.username = username
        self.password = password
        self.session_token: str | None = None
        self.session_expires_at: float = 0.0
        self._session = requests.Session()

    def _login(self) -> None:
        headers = {
            "X-Application": self.app_key,
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"username": self.username, "password": self.password}
        r = self._session.post(LOGIN_URL, headers=headers, data=data, timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get("status") != "SUCCESS":
            raise RuntimeError(f"betfair login failed: {j}")
        self.session_token = j["token"]
        self.session_expires_at = time.time() + 3 * 60 * 60  # refresh after 3h
        logger.info("logged in to Betfair; session refreshed")

    def _ensure_session(self) -> None:
        if not self.session_token or time.time() > self.session_expires_at:
            self._login()

    def _request(self, endpoint: str, params: dict) -> dict | list:
        self._ensure_session()
        url = f"{EXCHANGE_BASE}/{endpoint}/"
        headers = {
            "X-Application": self.app_key,
            "X-Authentication": self.session_token or "",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        r = self._session.post(url, headers=headers, json=params, timeout=30)
        if r.status_code == 401:
            # session expired earlier than expected
            logger.warning("got 401; refreshing session and retrying")
            self.session_token = None
            self._ensure_session()
            headers["X-Authentication"] = self.session_token or ""
            r = self._session.post(url, headers=headers, json=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def list_market_catalogue(
        self,
        from_iso: str,
        to_iso: str,
        market_types: list[str] | None = None,
        max_results: int = 200,
    ) -> list[dict]:
        params = {
            "filter": {
                "eventTypeIds": [CRICKET_EVENT_TYPE_ID],
                "marketTypeCodes": market_types or INTERESTING_MARKET_TYPES,
                "marketStartTime": {"from": from_iso, "to": to_iso},
            },
            "maxResults": max_results,
            "marketProjection": [
                "EVENT", "RUNNER_DESCRIPTION", "MARKET_START_TIME",
                "MARKET_DESCRIPTION", "COMPETITION",
            ],
        }
        result = self._request("listMarketCatalogue", params)
        return result if isinstance(result, list) else []

    def list_market_book(self, market_ids: list[str]) -> list[dict]:
        if not market_ids:
            return []
        # API supports up to 25 market IDs per call
        out = []
        for i in range(0, len(market_ids), 25):
            chunk = market_ids[i : i + 25]
            params = {
                "marketIds": chunk,
                "priceProjection": {
                    "priceData": ["EX_BEST_OFFERS", "EX_TRADED"],
                    "virtualise": True,
                },
            }
            r = self._request("listMarketBook", params)
            if isinstance(r, list):
                out.extend(r)
        return out


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
