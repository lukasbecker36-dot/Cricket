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
    threshold_X: int
    back_price: float | None
    lay_price: float | None
    available_back_size: float | None = None
    available_lay_size: float | None = None


@dataclass
class MarketExtraction:
    teams: list[str]               # ["Mumbai Indians", "Kolkata Knight Riders"]
    innings: int | None            # 1 or 2 (None if not determinable)
    market_name: str | None        # "1st Innings Runs" / "2nd Innings Runs" / etc.
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

Please extract structured data:

1. Match info:
   - teams: list of two team names exactly as Betfair displays them
   - innings: 1 or 2 (which innings the market is about). Infer from market name
     if visible, or the user caption, or prior context. Null if unclear.
   - market_name: full market name as shown (e.g. "1st Innings Runs",
     "2nd Innings 6 Overs Total"). Null if not visible.
   - venue: if visible or in caption, else null.

2. Runners (the "X Runs or more" ladder, or "Under/Over" pair):
   For EACH visible runner row, extract:
   - threshold_X: integer (the X value in "X Runs or more"). If the runner is
     an "Under" line, use the under threshold; if "Over", use over threshold.
   - back_price: decimal odds shown on the BACK side (usually pink/blue cells
     on the left). Float. Null if not visible.
   - lay_price: decimal odds shown on the LAY side (usually pink/blue cells
     on the right). Float. Null if not visible.
   - available_back_size, available_lay_size: GBP amounts shown under the
     prices (e.g. "£746", "£932"). Float, null if not visible.

3. Confidence: 'high' / 'medium' / 'low' overall.
4. Notes: 1-2 lines describing anything you weren't sure about.

Respond with a single JSON object, no other text. Schema:
{{
  "teams": ["str", "str"],
  "innings": 1 | 2 | null,
  "market_name": "str" | null,
  "venue": "str" | null,
  "runners": [
    {{"threshold_X": int, "back_price": float|null, "lay_price": float|null,
      "available_back_size": float|null, "available_lay_size": float|null}}
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
        runners = [RunnerExtraction(**r) for r in data.get("runners", [])]
        return MarketExtraction(
            teams=list(data.get("teams") or []),
            innings=data.get("innings"),
            market_name=data.get("market_name"),
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
