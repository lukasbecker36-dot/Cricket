"""v1 signal logger. Produces signals, logs them, never places a trade.

Per CLAUDE.md: live trading is v2, after >=1 month of paper trading.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    timestamp: str
    match_id: str
    ball_index: int
    model_p: float
    market_p: float
    edge: float
    suggested_side: str  # 'back' or 'lay'


def emit_signal(
    match_id: str,
    ball_index: int,
    model_p: float,
    market_p: float,
    edge_threshold: float,
    log_dir: Path | None = None,
) -> Signal | None:
    edge = model_p - market_p
    if abs(edge) < edge_threshold:
        return None
    sig = Signal(
        timestamp=datetime.utcnow().isoformat(),
        match_id=match_id,
        ball_index=ball_index,
        model_p=model_p,
        market_p=market_p,
        edge=edge,
        suggested_side="back" if edge > 0 else "lay",
    )
    logger.info("signal: %s", asdict(sig))
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"{match_id}.jsonl", "a") as f:
            f.write(json.dumps(asdict(sig)) + "\n")
    return sig
