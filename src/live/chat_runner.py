"""Chat-driven runner: receive Betfair screenshots via Telegram, extract data
with Anthropic vision, run the saved model, recommend lay bets.

Replaces the Betfair-API polling runner. No Betfair credentials needed -- the
bot only talks to Telegram and Anthropic, both of which Hetzner can reach
without IP restrictions.
"""
from __future__ import annotations

import json
import logging
import signal as posix_signal
from datetime import datetime, timezone
from pathlib import Path

from .chat_state import ChatStateStore, Session
from .config import LiveConfig
from .signals import FullInningsModel, Signal, format_signal, load_models, model_for_market_name
from .telegram import TelegramClient, extract_text_and_photo
from .vision import MarketExtraction, RunnerExtraction, TextIntent, VisionClient, to_jsonable

logger = logging.getLogger(__name__)


# Treat current calendar year as default season for prior lookups
def default_season() -> int:
    return datetime.now(timezone.utc).year


def infer_league_from_teams(teams: list[str]) -> str | None:
    if not teams:
        return None
    try:
        from src.ingestion.league_rosters import LEAGUE_TEAMS
        for league, roster in LEAGUE_TEAMS.items():
            if any(t in roster for t in teams):
                return league
    except ImportError:
        pass
    return None


def evaluate_with_model(
    model: FullInningsModel,
    extraction: MarketExtraction,
    session: Session,
) -> tuple[list[Signal], list[dict]]:
    """Run model on each extracted runner. Returns (signals_above_threshold,
    full_per_runner_breakdown)."""
    innings = extraction.innings or session.last_innings or 1
    teams = extraction.teams or session.teams or ["", ""]
    if len(teams) < 2:
        teams = teams + [""] * (2 - len(teams))
    venue = extraction.venue or session.venue or ""
    season = session.season or default_season()
    league = session.league or infer_league_from_teams(teams)

    # Innings 1: teams[0] bats (by convention -- user can override). Innings 2: teams[1].
    if innings == 1:
        bat, bowl = teams[0], teams[1]
    else:
        bat, bowl = teams[1], teams[0]

    signals: list[Signal] = []
    breakdown: list[dict] = []
    # Spread costs and execution friction: assume actual lay price is at least
    # SPREAD_HAIRCUT worse than the visible LTP/lay price. Sensitivity analysis
    # on the 2026 OOS data shows +30%% ROI holds at 3-5%% haircut; we use 3%%
    # as the working default. Reject markets whose visible spread (back/lay
    # gap) is already wider than that.
    SPREAD_HAIRCUT = 0.03
    MAX_VISIBLE_SPREAD = 0.10  # if visible lay > 10%% above back, skip the market
    for r in extraction.runners:
        # Reject runners with a wide visible spread (illiquid market)
        if r.back_price is not None and r.lay_price is not None and r.back_price > 1.0:
            visible_spread = (r.lay_price - r.back_price) / r.back_price
            if visible_spread > MAX_VISIBLE_SPREAD:
                breakdown.append({
                    "X": r.threshold_X, "lay": r.lay_price,
                    "skip_reason": f"spread {visible_spread*100:.0f}%% too wide",
                })
                continue

        # Lay price is what we'd pay to take the LAY side
        lay_price = r.lay_price
        if lay_price is None or lay_price <= 1.0:
            if r.back_price is not None and r.back_price > 1.0:
                # No visible lay; estimate one tick above back, then apply haircut
                lay_price = r.back_price * (1 + SPREAD_HAIRCUT)
            else:
                breakdown.append({"X": r.threshold_X, "lay": None, "skip_reason": "no lay price"})
                continue
        # Apply spread haircut to model the realistic execution price
        exec_lay_price = lay_price * (1 + SPREAD_HAIRCUT)
        market_implied = 1.0 / exec_lay_price
        if not (model.implied_min <= market_implied <= model.implied_max):
            breakdown.append({
                "X": r.threshold_X, "lay": lay_price, "implied": market_implied,
                "skip_reason": f"out of {model.implied_min}-{model.implied_max} band",
            })
            continue
        model_p = model.predict_p(
            threshold_X=r.threshold_X, implied_open=market_implied,
            batting_team=bat, bowling_team=bowl, venue=venue,
            season=season, innings=innings, league=league,
        )
        edge = model_p - market_implied
        breakdown.append({
            "X": r.threshold_X, "lay": lay_price, "exec_lay": round(exec_lay_price, 3),
            "implied": round(market_implied, 3),
            "model_p": round(model_p, 3), "edge_pp": round(edge * 100, 1),
        })
        if edge < model.edge_threshold:
            signals.append(Signal(
                detected_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                market_id="screenshot", event_id="",
                event_name=f"{teams[0]} v {teams[1]}",
                market_name=extraction.market_name or f"Innings {innings} Runs",
                innings=innings, runner_id=r.threshold_X,
                runner_name=f"{r.threshold_X} Runs or more",
                threshold_X=r.threshold_X,
                market_implied=market_implied, market_lay_price=exec_lay_price,
                model_p=model_p, edge=edge,
                suggested_action="LAY",
                league_hint=league,
            ))
    return signals, breakdown


# Standard exchange decimal odds for Innings Runs Line markets.
# Confirmed on the live Betfair Exchange app: line markets quote at 2.0
# (effective 50/50 implied probability). 5%% commission on winnings.
# Sensitivity in evaluate_line_combined.py: at 2.0 vs 1.92, backtest ROI
# is ~5pp higher (better payout per win, slightly bigger edge per signal).
LINE_DEFAULT_ODDS = 2.00
LINE_EDGE_THRESHOLD = 0.05   # 5pp edge above 1/odds required for a signal
LINE_IMPLIED_AT_ODDS = 1.0 / LINE_DEFAULT_ODDS  # ~0.521


def evaluate_line_market_with_model(
    model: FullInningsModel,
    extraction: MarketExtraction,
    session: Session,
) -> tuple[list[Signal], list[dict]]:
    """For Line markets: model P(actual >= line). Compare to 1/decimal_odds.
    If model_p - implied > threshold -> back OVER; if < -threshold -> back UNDER."""
    innings = extraction.innings or session.last_innings or 1
    teams = extraction.teams or session.teams or ["", ""]
    if len(teams) < 2:
        teams = teams + [""] * (2 - len(teams))
    venue = extraction.venue or session.venue or ""
    season = session.season or default_season()
    league = session.league or infer_league_from_teams(teams)
    bat, bowl = (teams[0], teams[1]) if innings == 1 else (teams[1], teams[0])

    # Identify under/over runners; if only one was extracted, infer the line range
    under_runner = next((r for r in extraction.runners if r.side == "under"), None)
    over_runner  = next((r for r in extraction.runners if r.side == "over"),  None)
    # If side wasn't set, fall back to threshold values: lower is under, higher is over
    if under_runner is None or over_runner is None:
        sorted_runners = sorted(extraction.runners, key=lambda r: r.threshold_X)
        if len(sorted_runners) >= 2:
            under_runner = under_runner or sorted_runners[0]
            over_runner  = over_runner  or sorted_runners[-1]

    signals: list[Signal] = []
    breakdown: list[dict] = []
    for runner, label in [(over_runner, "over"), (under_runner, "under")]:
        if runner is None:
            continue
        line_X = int(runner.threshold_X)
        odds = (runner.back_price if label == "over" else runner.lay_price) or LINE_DEFAULT_ODDS
        implied = 1.0 / odds
        # Model gives P(actual >= X). For 'over X.5' line, threshold is X+1.
        # For 'under X.5' line, prob_under = 1 - P(actual >= X+1) using same model call
        # at threshold X+1 (since under X.5 means actual <= X).
        # Simplification: use the line value as the X feature (model is robust to off-by-1).
        model_p_over_at_line = model.predict_p(
            threshold_X=line_X, implied_open=LINE_IMPLIED_AT_ODDS,
            batting_team=bat, bowling_team=bowl, venue=venue,
            season=season, innings=innings, league=league,
        )
        if label == "over":
            side_p = model_p_over_at_line          # we back over: need P(actual >= line)
        else:
            side_p = 1.0 - model_p_over_at_line    # we back under: need P(actual < line)
        edge = side_p - implied
        breakdown.append({
            "side": label, "line_X": line_X, "odds": round(odds, 3),
            "implied": round(implied, 3),
            "model_p": round(side_p, 3),
            "edge_pp": round(edge * 100, 1),
        })
        if edge > LINE_EDGE_THRESHOLD:
            signals.append(Signal(
                detected_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                market_id="screenshot", event_id="",
                event_name=f"{teams[0]} v {teams[1]}",
                market_name=extraction.market_name or f"Innings {innings} Runs Line",
                innings=innings, runner_id=line_X,
                runner_name=f"{label.capitalize()} {line_X}",
                threshold_X=line_X,
                market_implied=implied, market_lay_price=odds,
                model_p=side_p, edge=edge,
                suggested_action=f"BACK {label.upper()}",
                league_hint=league,
            ))
    return signals, breakdown


def format_breakdown(breakdown: list[dict]) -> str:
    """Compact line per runner for confirmation display. Handles both ladder
    and line market breakdown rows."""
    lines = []
    for b in breakdown:
        if "side" in b:
            # Line market row
            edge = b["edge_pp"]
            tag = f" ← BACK {b['side'].upper()}" if edge > 5 else ""
            lines.append(
                f"  {b['side']:>5} {b['line_X']:>4}   odds {b['odds']:.2f}  "
                f"(implied {b['implied']*100:.0f}%) model {b['model_p']*100:.0f}%  "
                f"edge {edge:+.1f}pp{tag}"
            )
        elif "skip_reason" in b:
            lines.append(f"  {b['X']:>4} or more   lay {b.get('lay','?')}  - skipped ({b['skip_reason']})")
        else:
            edge = b["edge_pp"]
            tag = " ← LAY" if edge < -3 else ""
            lines.append(f"  {b['X']:>4} or more   lay {b['lay']:.2f}  (implied {b['implied']*100:.0f}%) "
                         f"model {b['model_p']*100:.0f}%  edge {edge:+.1f}pp{tag}")
    return "\n".join(lines)


class ChatRunner:
    def __init__(self, cfg: LiveConfig, anthropic_api_key: str | None = None):
        self.cfg = cfg
        self.telegram = TelegramClient(cfg.telegram_bot_token, cfg.telegram_chat_id)
        self.models = load_models(cfg.model_dir)
        if "full" not in self.models:
            raise RuntimeError(f"No full_innings model found in {cfg.model_dir}")
        # Backward compatibility: legacy code refers to self.model
        self.model = self.models["full"]
        logger.info("loaded models: %s", sorted(self.models.keys()))
        self.vision = VisionClient(api_key=anthropic_api_key)
        self.state = ChatStateStore(cfg.log_dir.parent / "chat_state.json")
        self.stop_requested = False
        posix_signal.signal(posix_signal.SIGTERM, self._on_signal)
        posix_signal.signal(posix_signal.SIGINT, self._on_signal)

    def _on_signal(self, signum, frame):
        logger.info("got signal %s, stopping", signum)
        self.stop_requested = True

    # ----- handlers -----
    def handle_photo(self, photo_id: str, caption: str) -> None:
        self.telegram.send_typing()
        dl = self.telegram.download_photo(photo_id)
        if dl is None:
            self.telegram.send("Couldn't download that image — try again?")
            return
        img_bytes, media_type = dl
        s = self.state.session
        prior_context = {
            "teams": s.teams, "league": s.league, "venue": s.venue,
            "season": s.season, "last_innings": s.last_innings,
        }
        try:
            extraction = self.vision.extract_market(
                img_bytes, media_type, user_caption=caption, prior_context=prior_context,
            )
        except Exception as e:
            logger.exception("vision call failed")
            self.telegram.send(f"Vision API error: `{e}`")
            return

        # Update session from anything new
        updates = {}
        if extraction.teams and len(extraction.teams) >= 2:
            updates["teams"] = extraction.teams[:2]
            league = infer_league_from_teams(extraction.teams[:2])
            if league:
                updates["league"] = league
        if extraction.innings is not None:
            updates["last_innings"] = extraction.innings
        if extraction.venue:
            updates["venue"] = extraction.venue
        if updates:
            self.state.update_session(**updates)

        # Pick the phase-appropriate model based on the extracted market name
        chosen_model = model_for_market_name(extraction.market_name or "", self.models) or self.model
        if extraction.market_kind == "line":
            signals, breakdown = evaluate_line_market_with_model(
                chosen_model, extraction, self.state.session
            )
        else:
            signals, breakdown = evaluate_with_model(
                chosen_model, extraction, self.state.session
            )

        # Stash pending state in case user wants to confirm or override
        self.state.set_pending({
            "kind": "screenshot",
            "extraction": to_jsonable(extraction),
            "breakdown": breakdown,
            "signals": [to_jsonable(s) for s in signals],
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self.state.append_history({
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": "screenshot",
            "caption": caption,
            "teams": extraction.teams,
            "innings": extraction.innings,
            "n_signals": len(signals),
        })

        # Compose reply
        sess = self.state.session
        teams_str = " v ".join(sess.teams) if sess.teams else "(teams unknown)"
        venue_str = f" @ {sess.venue}" if sess.venue else ""
        header = (
            f"*Read it:* {teams_str}{venue_str}, innings {extraction.innings or sess.last_innings or '?'}"
            f"  (vision confidence: {extraction.confidence})\n"
        )
        if extraction.notes:
            header += f"_Notes:_ {extraction.notes}\n"

        body = format_breakdown(breakdown) if breakdown else "  (no runners extracted)"
        if signals:
            recs = "\n\n*Signals:*\n"
            for s in signals:
                recs += "\n" + format_signal(s, stake_flat=10.0) + "\n"
        else:
            recs = "\n\n_No model edge < -3pp on this market._"

        footer = "\nReply *yes* / *no* to confirm or correct, or send another screenshot."
        self.telegram.send(header + "```\n" + body + "\n```" + recs + footer)

    def handle_text(self, text: str) -> None:
        s = self.state.session
        prior_context = {
            "teams": s.teams, "league": s.league, "venue": s.venue,
            "season": s.season, "last_innings": s.last_innings,
        }
        pending = self.state.pending
        intent: TextIntent
        try:
            intent = self.vision.parse_text(text, prior_context=prior_context, pending_extraction=pending)
        except Exception as e:
            logger.exception("text parse failed")
            self.telegram.send(f"Couldn't parse: `{e}`")
            return

        if intent.kind == "command":
            self._do_command(intent.payload.get("name", "help"))
            return
        if intent.kind == "context_update":
            self._apply_context_update(intent.payload)
            return
        if intent.kind == "confirm":
            if pending:
                self.telegram.send("Confirmed. Place your lay(s) on Betfair.")
                self.state.set_pending(None)
            else:
                self.telegram.send("Nothing pending to confirm — send a screenshot first.")
            return
        if intent.kind == "reject":
            self.state.set_pending(None)
            self.telegram.send("OK, dropped that one. Send a fresh screenshot when ready.")
            return
        self.telegram.send(
            "Not sure what you meant. Try:\n"
            "- send a screenshot of an Innings Runs market\n"
            "- say things like 'this is at Wankhede' or 'now innings 2'\n"
            "- 'status', 'reset', 'help'"
        )

    def _do_command(self, name: str) -> None:
        if name == "status":
            s = self.state.session
            self.telegram.send(
                "*Status*\n"
                f"current match: {' v '.join(s.teams) if s.teams else '-'}\n"
                f"league: {s.league or '-'}   venue: {s.venue or '-'}\n"
                f"innings: {s.last_innings or '-'}   season: {s.season or default_season()}\n"
                f"pending: {'yes' if self.state.pending else 'no'}"
            )
        elif name == "reset":
            self.state.reset_session()
            self.telegram.send("Session cleared.")
        elif name == "clear":
            self.state.set_pending(None)
            self.telegram.send("Cleared pending extraction.")
        elif name == "help":
            self.telegram.send(
                "*How to use*\n"
                "Send a screenshot of a Betfair *Innings Runs* market (any innings) "
                "with a quick caption like _'MI v KKR'_ or _'innings 2 now'_. "
                "I'll extract the ladder and tell you any lay opportunities the model spots.\n\n"
                "You can update context any time:\n"
                "  - 'venue is Wankhede'\n"
                "  - 'now innings 2'\n"
                "  - 'reset' to clear the current match\n"
                "  - 'status' to see what I know"
            )
        else:
            self.telegram.send(f"Unknown command: `{name}`")

    def _apply_context_update(self, payload: dict) -> None:
        updates = {}
        if payload.get("teams"):
            updates["teams"] = list(payload["teams"])[:2]
            updates["league"] = infer_league_from_teams(updates["teams"])
        if payload.get("innings") is not None:
            updates["last_innings"] = int(payload["innings"])
        if payload.get("venue"):
            updates["venue"] = str(payload["venue"])
        if updates:
            self.state.update_session(**updates)
            self.telegram.send(f"Updated: {', '.join(f'{k}={v}' for k, v in updates.items())}")
        else:
            self.telegram.send("Got it (nothing concrete to update).")

    # ----- main loop -----
    def run(self) -> int:
        self.telegram.send(
            "Cricket signal helper *online*.\n"
            "Send screenshots of Betfair Innings Runs markets. Type 'help' for tips."
        )
        while not self.stop_requested:
            try:
                updates = self.telegram.get_updates(timeout_s=25)
            except Exception:
                logger.exception("get_updates loop error")
                continue
            for upd in updates:
                text, photo_id = extract_text_and_photo(upd)
                try:
                    if photo_id:
                        self.handle_photo(photo_id, caption=text)
                    elif text:
                        self.handle_text(text)
                except Exception as e:
                    logger.exception("update handling failed")
                    self.telegram.send(f"Oops: `{e}`")
        self.telegram.send("Cricket signal helper *offline*.")
        return 0
