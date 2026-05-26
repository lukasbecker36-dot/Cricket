"""Apply the saved full_innings model to live Betfair markets and emit signals.

Single match -> two markets ('1st Innings Runs', '2nd Innings Runs'), each a
multi-runner ladder. For each runner trading in [implied_min, implied_max]:
  1. Compute the feature row
  2. Score with the model
  3. If model_p - market_implied < edge_threshold -> emit signal (lay)

Signals are dataclasses; the runner is responsible for de-duping (so we
don't spam Telegram with the same trade repeatedly).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Signal:
    detected_at_utc: str
    market_id: str
    event_id: str
    event_name: str
    market_name: str       # '1st Innings Runs' / '2nd Innings Runs'
    innings: int           # 1 or 2
    runner_id: int
    runner_name: str       # 'X Runs or more'
    threshold_X: int
    market_implied: float
    market_lay_price: float
    model_p: float
    edge: float            # model_p - market_implied (negative = lay opportunity)
    suggested_action: str  # 'LAY'
    league_hint: str | None

    def key(self) -> str:
        return f"{self.market_id}:{self.runner_id}"


class FullInningsModel:
    """Generic phase model loader. Defaults to the full_innings files but
    accepts a label prefix to load phase_6 / phase_10 / phase_15 etc."""

    def __init__(self, model_dir: Path, prefix: str = "full_innings"):
        self.prefix = prefix
        with open(model_dir / f"{prefix}_meta.json") as f:
            self.meta = json.load(f)
        self.features: list[str] = self.meta["features"]
        self.edge_threshold: float = float(self.meta["edge_threshold"])
        self.implied_min: float = float(self.meta.get("implied_min", 0.10))
        self.implied_max: float = float(self.meta.get("implied_max", 0.90))
        self.leagues: list[str] = list(self.meta["leagues"])
        self.default_par: float = float(self.meta["default_par"])
        self.target_balls: int = int(self.meta.get("target_balls", 120))
        self.booster = lgb.Booster(model_file=str(model_dir / f"{prefix}_gbm.lgb"))
        # File names differ between the original (bat_pp/bowl_pp/venue_par) and
        # the phase models (bat_prior/bowl_prior/venue_par). Try both.
        for fname in (f"{prefix}_bat_pp.json", f"{prefix}_bat_prior.json"):
            p = model_dir / fname
            if p.exists():
                self.bat_pp = json.loads(p.read_text())
                break
        for fname in (f"{prefix}_bowl_pp.json", f"{prefix}_bowl_prior.json"):
            p = model_dir / fname
            if p.exists():
                self.bowl_pp = json.loads(p.read_text())
                break
        self.venue_par = json.loads((model_dir / f"{prefix}_venue_par.json").read_text())
        # Optional: trend-aware companion. {season: league_avg_phase_total_prior_season}
        trend_path = model_dir / f"{prefix}_league_trend.json"
        self.league_trend: dict = json.loads(trend_path.read_text()) if trend_path.exists() else {}
        # Optional: weather feature medians for imputation (stored in meta)
        self.weather_medians: dict = self.meta.get("weather_medians", {})
        # Optional: anomaly-weather companion. Per-venue climatological normals
        # used to convert raw conditions into deviations-from-normal so 'hot'
        # means the same in England and India. {venue: {var: mean}} + global fallback.
        climo_path = model_dir / f"{prefix}_venue_climo.json"
        if climo_path.exists():
            blob = json.loads(climo_path.read_text())
            self.venue_climo: dict = blob.get("by_venue", {})
            self.global_climo: dict = blob.get("global", {})
        else:
            self.venue_climo, self.global_climo = {}, {}

    def _lookup(self, table: dict, team: str, season: int) -> float:
        from src.ingestion.teams import canonical_team
        return float(table.get(f"{canonical_team(team)}|{season}", self.default_par))

    def predict_p(self, *, threshold_X: int, implied_open: float,
                  batting_team: str, bowling_team: str, venue: str,
                  season: int, innings: int, league: str | None,
                  target: float | None = None,
                  weather: dict | None = None) -> float:
        bat_prior = self._lookup(self.bat_pp, batting_team, season)
        bowl_prior = self._lookup(self.bowl_pp, bowling_team, season)
        v_par = self._lookup(self.venue_par, venue, season)
        phase_par_from_target = (float(target) * self.target_balls / 120.0) if target is not None else 0.0
        # Trend feature: league's prior-season average phase total. Falls back to
        # default_par when this season isn't in the trend table.
        trend = float(self.league_trend.get(str(int(season)), self.default_par)) if self.league_trend else self.default_par
        row = {
            "threshold_X": float(threshold_X),
            "implied_open": float(implied_open),
            "bat_prior": bat_prior,
            "bowl_prior": bowl_prior,
            "venue_par": v_par,
            "x_minus_par": float(threshold_X) - v_par,
            "x_minus_bat": float(threshold_X) - bat_prior,
            "x_minus_bowl": float(threshold_X) - bowl_prior,
            "innings": int(innings),
            "target": float(target) if target is not None else 0.0,
            "phase_par_from_target": phase_par_from_target,
            "x_minus_target_par": float(threshold_X) - phase_par_from_target,
            "league_trend": trend,
            "x_minus_trend": float(threshold_X) - trend,
        }
        # Absolute weather features: use provided values, else median-impute.
        for f in ("temp_c", "humidity_pct", "wind_kph", "precip_mm", "cloud_pct"):
            if f in self.features:
                val = (weather or {}).get(f)
                if val is None and self.weather_medians:
                    val = self.weather_medians.get(f)
                row[f] = float(val) if val is not None else 0.0
        # Anomaly weather features: deviation from this venue's climatological
        # normal. Missing weather -> 0 anomaly (assume normal conditions).
        anom_map = {"temp_anom": "temp_c", "humid_anom": "humidity_pct",
                    "wind_anom": "wind_kph", "cloud_anom": "cloud_pct"}
        for anom_feat, raw_var in anom_map.items():
            if anom_feat in self.features:
                raw = (weather or {}).get(raw_var)
                if raw is None:
                    row[anom_feat] = 0.0
                else:
                    base = (self.venue_climo.get(venue, {}).get(raw_var)
                            if venue in self.venue_climo else None)
                    if base is None:
                        base = self.global_climo.get(raw_var, float(raw))
                    row[anom_feat] = float(raw) - float(base)
        # precip stays absolute even in anomaly mode (handled in the loop above)
        for L in self.leagues:
            row[f"is_{L}"] = 1.0 if league == L else 0.0
        # Build the row vector strictly from self.features so missing keys aren't sent.
        x = np.array([[row.get(f, 0.0) for f in self.features]], dtype=np.float32)
        return float(self.booster.predict(x)[0])


def load_models(model_dir: Path) -> dict[str, "FullInningsModel"]:
    """Load every available phase model. Returns dict keyed by short label."""
    out: dict[str, FullInningsModel] = {}
    for prefix, label in [
        ("full_innings", "full"),
        ("phase_6", "6"),
        ("phase_10", "10"),
        ("phase_15", "15"),
        ("phase_6_inn2", "6_inn2"),
        ("phase_10_inn2", "10_inn2"),
    ]:
        if (model_dir / f"{prefix}_meta.json").exists():
            try:
                out[label] = FullInningsModel(model_dir, prefix=prefix)
            except Exception as e:  # missing companion files etc.
                pass
    return out


def model_for_market_name(market_name: str, registry: dict[str, "FullInningsModel"]) -> "FullInningsModel | None":
    """Pick the right phase model for a given Betfair market name.

    Routes innings-2 6-over and 10-over markets to dedicated inn2 models
    when available. Innings-2 phase_15 and full innings have too much
    chase-end selection bias and stay on the inn1 models (which means in
    practice we'll often skip those trades)."""
    if not market_name:
        return registry.get("full")
    name = market_name.lower()
    is_inn2 = name.startswith("2nd") or name.startswith("second")
    if "6 over" in name or "6 overs" in name:
        if is_inn2:
            return registry.get("6_inn2") or registry.get("6") or registry.get("full")
        return registry.get("6") or registry.get("full")
    if "10 over" in name or "10 overs" in name:
        if is_inn2:
            return registry.get("10_inn2") or registry.get("10") or registry.get("full")
        return registry.get("10") or registry.get("full")
    if "15 over" in name or "15 overs" in name:
        return registry.get("15") or registry.get("full")
    return registry.get("full")


def parse_runner_threshold(name: str) -> int | None:
    m = re.match(r"^(\d+)\s+Runs?\s+or\s+more", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def detect_league_from_event(event_name: str, competition_name: str = "") -> str | None:
    """Best-effort league detection from event name or competition name."""
    haystack = f"{competition_name} {event_name}".lower()
    if "ipl" in haystack or "indian premier" in haystack:
        return "ipl"
    if "big bash" in haystack or "bbl" in haystack:
        return "bbl"
    if "pakistan super league" in haystack or "psl" in haystack:
        return "psl"
    if "caribbean premier league" in haystack or "cpl" in haystack:
        return "cpl"
    if "t20 blast" in haystack or "vitality blast" in haystack or "ntb" in haystack:
        return "ntb"
    # Fall back to team-name heuristics
    from src.ingestion.league_rosters import LEAGUE_TEAMS
    for league, teams in LEAGUE_TEAMS.items():
        if any(t in event_name for t in teams):
            return league
    return None


def derive_batting_team(event_name: str, innings: int, toss_winner: str | None, toss_decision: str | None) -> tuple[str, str]:
    """Approximate: split 'Team A v Team B' into the two teams; we DON'T always
    know who batted first pre-match. If we have toss info, use it; otherwise
    return (team1, team2) and let the model's symmetric features handle ambiguity.
    """
    parts = re.split(r"\s+v(?:s)?\s+", event_name, maxsplit=1)
    if len(parts) != 2:
        return ("", "")
    team1, team2 = parts[0].strip(), parts[1].strip()
    if toss_winner and toss_decision:
        if toss_decision.lower() == "bat":
            first_innings_batters = toss_winner
        else:
            first_innings_batters = team2 if toss_winner == team1 else team1
        second_innings_batters = team2 if first_innings_batters == team1 else team1
        if innings == 1:
            return (first_innings_batters, second_innings_batters)
        return (second_innings_batters, first_innings_batters)
    # Best guess without toss: assume team1 bats first
    if innings == 1:
        return (team1, team2)
    return (team2, team1)


def evaluate_market(
    *,
    model: FullInningsModel,
    market_catalogue: dict,
    market_book: dict,
    season: int,
) -> list[Signal]:
    market_id = market_book.get("marketId")
    if not market_id:
        return []
    market_name = market_catalogue.get("marketName", "")
    if "Innings Runs" not in market_name or "Over" in market_name:
        return []
    innings = 1 if market_name.lower().startswith("1st") else 2
    event = market_catalogue.get("event", {}) or {}
    event_name = event.get("name", "")
    event_id = str(event.get("id", ""))
    competition = (market_catalogue.get("competition") or {}).get("name", "")
    league = detect_league_from_event(event_name, competition)
    venue = event.get("venue", "")
    bat_team, bowl_team = derive_batting_team(event_name, innings, None, None)

    runners_meta = {int(r["selectionId"]): r["runnerName"]
                    for r in market_catalogue.get("runners", []) if "selectionId" in r}

    out: list[Signal] = []
    for runner in market_book.get("runners", []):
        rid = int(runner.get("selectionId", 0))
        name = runners_meta.get(rid, "")
        X = parse_runner_threshold(name)
        if X is None:
            continue
        ex = runner.get("ex", {}) or {}
        # We want to LAY -> we pay the best AVAILABLE TO LAY price
        lay = ex.get("availableToLay") or []
        if not lay:
            continue
        lay_price = float(lay[0]["price"])
        if lay_price <= 1.0:
            continue
        market_implied = 1.0 / lay_price
        if not (model.implied_min <= market_implied <= model.implied_max):
            continue
        model_p = model.predict_p(
            threshold_X=X, implied_open=market_implied,
            batting_team=bat_team, bowling_team=bowl_team, venue=venue,
            season=season, innings=innings, league=league,
        )
        edge = model_p - market_implied
        if edge >= model.edge_threshold:
            continue
        out.append(Signal(
            detected_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            market_id=str(market_id),
            event_id=event_id,
            event_name=event_name,
            market_name=market_name,
            innings=innings,
            runner_id=rid,
            runner_name=name,
            threshold_X=X,
            market_implied=market_implied,
            market_lay_price=lay_price,
            model_p=model_p,
            edge=edge,
            suggested_action="LAY",
            league_hint=league,
        ))
    return out


def format_signal(sig: Signal, stake_flat: float = 10.0) -> str:
    liability = stake_flat * (sig.market_lay_price - 1.0)
    return (
        f"*Cricket LAY signal*  ({sig.league_hint or 'unknown league'})\n"
        f"_{sig.event_name}_\n"
        f"`{sig.market_name}`  innings {sig.innings}\n"
        f"Runner: *{sig.runner_name}*\n"
        f"Lay at *{sig.market_lay_price:.2f}*  (implied {sig.market_implied:.1%})\n"
        f"Model says {sig.model_p:.1%}  -> edge {sig.edge:+.1%}\n"
        f"£{stake_flat:.0f} stake -> liability £{liability:.2f}  / win £{stake_flat*0.95:.2f}\n"
        f"`{sig.market_id} / {sig.runner_id}`"
    )
