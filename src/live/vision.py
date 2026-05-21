"""Anthropic vision extractor for Betfair screenshots + conversational user text.

The vision call takes an image (Betfair market screenshot) plus the user's
caption AND any prior conversation context, and returns structured data:
match info + runner ladder.

The text-intent call takes a user text message + context and figures out
what the user means (confirmation, correction, context update, command).

Uses claude-sonnet-4-5 for accuracy on prices.
"""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import asdict, dataclass

from anthropic import Anthropic

logger = logging.getLogger(__name__)

VISION_MODEL = "claude-sonnet-4-5"
TEXT_MODEL = "claude-haiku-4-5"


@dataclass
class RunnerExtraction:
    threshold_X: int               # for ladder runners: the X in "X or more"
                                   # for line runners: the line value, rounded
    back_price: float | None
    lay_price: float | None
    available_back_size: float | None = None
    available_lay_size: float | None = None
    side: str | None = None        # 'under' / 'over' for line markets, None for ladders


@dataclass
class MarketExtraction:
    teams: list[str]               # ["Mumbai Indians", "Kolkata Knight Riders"]
    innings: int | None            # 1 or 2 (None if not determinable)
    market_name: str | None        # "1st Innings Runs" / "2nd Innings Runs" / etc.
    market_kind: str               # 'ladder' (multi-runner X-or-more) or 'line' (single line, Under/Over)
    venue: str | None
    runners: list[RunnerExtraction]
    confidence: str                # 'high' / 'medium' / 'low'
    notes: str                     # what the model wasn't sure about


@dataclass
class TextIntent:
    kind: str                      # 'confirm' / 'reject' / 'context_update' / 'command' / 'unknown'
    payload: dict                  # arbitrary structured update


class VisionClient:
    def __init__(self, api_key: str | None = None):
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def extract_market(
        self,
        image_bytes: bytes,
        image_media_type: str,
        user_caption: str,
        prior_context: dict | None = None,
    ) -> MarketExtraction:
        """Run vision extraction. prior_context: {teams, league, venue, season,
        last_innings} from session memory (may be partial or empty)."""
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        context_str = json.dumps(prior_context, indent=2) if prior_context else "(no prior context)"
        prompt = f"""You are analysing a Betfair Exchange screenshot for cricket betting.

User caption: "{user_caption}"

Prior session context (may be relevant for resolving ambiguity):
{context_str}

Cricket markets on Betfair come in two main formats for innings totals:

  LADDER format: many rows of "X Runs or more" each with their own back/lay
    decimal odds. Example market names: "1st Innings Runs", "2nd Innings
    6 Overs Total". Treat as market_kind = "ladder".

  LINE format: a single row labelled "Total Runs" with an Under value and
    an Over value as line thresholds (e.g. "Under 59.5 / Over 60.5") and
    £ liquidity amounts. Decimal odds are usually NOT shown (Betfair
    convention: ~1.92 each side). Example market names: "1st Innings
    Runs Line", "1st Innings 6 Overs Line". Treat as market_kind = "line".

Please extract structured data:

1. Match info:
   - teams: list of two team names exactly as Betfair displays them
   - innings: 1 or 2 (which innings the market is about). Infer from market name
     if visible, or the user caption, or prior context. Null if unclear.
   - market_name: full market name as shown. Null if not visible.
   - market_kind: "ladder" or "line" per the formats above.
   - venue: if visible or in caption, else null.

2. Runners:
   For a LADDER market, one runner per "X Runs or more" row:
     - threshold_X: integer (the X value)
     - back_price: decimal odds on BACK side
     - lay_price: decimal odds on LAY side
     - available_back_size, available_lay_size: GBP liquidity if visible
     - side: null

   For a LINE market, TWO runners (one for each side):
     - {{"threshold_X": <under line as int>, "back_price": null, "lay_price": null,
        "available_back_size": <under £ if visible>, "available_lay_size": null,
        "side": "under"}}
     - {{"threshold_X": <over line as int>, "back_price": null, "lay_price": null,
        "available_back_size": null, "available_lay_size": <over £ if visible>,
        "side": "over"}}
     Important: for LINE markets, the visible "59.5" / "60.5" values are line
     THRESHOLDS not decimal odds. Decimal odds are typically ~1.92 and usually
     not shown. Leave back_price / lay_price null if odds aren't explicitly
     displayed elsewhere on the screen.

3. Confidence: 'high' / 'medium' / 'low' overall.
4. Notes: 1-2 lines describing anything you weren't sure about.

Respond with a single JSON object, no other text. Schema:
{{
  "teams": ["str", "str"],
  "innings": 1 | 2 | null,
  "market_name": "str" | null,
  "market_kind": "ladder" | "line",
  "venue": "str" | null,
  "runners": [
    {{"threshold_X": int, "back_price": float|null, "lay_price": float|null,
      "available_back_size": float|null, "available_lay_size": float|null,
      "side": "under" | "over" | null}}
  ],
  "confidence": "high|medium|low",
  "notes": "str"
}}"""

        response = self.client.messages.create(
            model=VISION_MODEL,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": image_media_type, "data": b64,
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw = response.content[0].text.strip()
        # Strip optional code fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if raw.startswith("json\n"):
                raw = raw[5:]
        data = json.loads(raw)
        # Tolerate runners with the new 'side' field or without it
        runners = []
        for r in data.get("runners", []):
            r = dict(r)
            r.setdefault("side", None)
            runners.append(RunnerExtraction(**r))
        market_kind = data.get("market_kind") or (
            "line" if any(r.side in ("under", "over") for r in runners) else "ladder"
        )
        return MarketExtraction(
            teams=list(data.get("teams") or []),
            innings=data.get("innings"),
            market_name=data.get("market_name"),
            market_kind=str(market_kind),
            venue=data.get("venue"),
            runners=runners,
            confidence=str(data.get("confidence", "low")),
            notes=str(data.get("notes", "")),
        )

    def parse_text(
        self,
        user_message: str,
        prior_context: dict | None = None,
        pending_extraction: dict | None = None,
    ) -> TextIntent:
        """Use the LLM to figure out what the user means in conversational text."""
        ctx_str = json.dumps(prior_context, indent=2) if prior_context else "(none)"
        pending_str = json.dumps(pending_extraction, indent=2) if pending_extraction else "(none)"
        prompt = f"""You are interpreting a user's chat message in a cricket-betting helper bot.

User message: "{user_message}"

Session context:
{ctx_str}

Pending extraction (waiting for confirmation):
{pending_str}

The user can be:
- Confirming a pending extraction (e.g. 'yes', 'looks good', '✓', 'ok', 'go')
- Rejecting it (e.g. 'no', 'wrong', '✗', 'cancel', 'forget it')
- Updating context (e.g. 'this is innings 2', 'venue is Wankhede', 'now KKR are batting')
- Issuing a command (e.g. 'status', 'reset', 'help', 'clear')
- Asking a question or saying something unrelated

Return JSON, no other text:
{{
  "kind": "confirm" | "reject" | "context_update" | "command" | "unknown",
  "payload": {{...}}
}}

For "context_update", payload may include any of: teams (list), innings (int),
venue (str), match_starting_soon (bool). Only include fields explicitly given.

For "command", payload = {{"name": "status"|"reset"|"help"|"clear"|...}}.

For "confirm"/"reject"/"unknown", payload can be empty {{}}."""
        response = self.client.messages.create(
            model=TEXT_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if raw.startswith("json\n"):
                raw = raw[5:]
        data = json.loads(raw)
        return TextIntent(kind=str(data.get("kind", "unknown")),
                          payload=dict(data.get("payload") or {}))


def to_jsonable(obj):
    """Convert dataclass instances to dicts for JSON storage."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj
