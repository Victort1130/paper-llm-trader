"""Decision type shared by every strategy, and parsing of LLM output."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

Action = Literal["BUY", "SELL", "HOLD"]


class Decision(BaseModel):
    ticker: str
    action: Action
    target_weight: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = ""


class _RawDecision(BaseModel):
    ticker: str
    action: Action
    target_weight: float
    reason: str


class _RawDecisionSet(BaseModel):
    decisions: list[_RawDecision]


def output_schema(max_weight: float) -> dict:
    """JSON schema sent as output_config.format."""
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                        "target_weight": {
                            "type": "number",
                            "description": f"0 to {max_weight}",
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["ticker", "action", "target_weight", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def hold_all(tickers: list[str], reason: str) -> list[Decision]:
    return [Decision(ticker=t, action="HOLD", reason=reason) for t in tickers]


def parse_decisions(
    text: str, alias_to_ticker: dict[str, str], tickers: list[str]
) -> tuple[list[Decision], list[str]]:
    """Parse model text into one Decision per ticker.

    Anything unparseable becomes HOLD, never a trade. Returns the decisions
    and a list of problems found (empty when the output was clean).
    """
    problems: list[str] = []
    try:
        raw = _RawDecisionSet.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        return hold_all(tickers, "parse_error"), [f"parse_error: {type(e).__name__}"]

    by_ticker: dict[str, Decision] = {}
    for d in raw.decisions:
        real = alias_to_ticker.get(d.ticker)
        if real is None:
            problems.append(f"unknown ticker {d.ticker!r}")
            continue
        if real in by_ticker:
            problems.append(f"duplicate decision for {d.ticker!r}")
            continue
        w = min(max(d.target_weight, 0.0), 1.0)
        if w != d.target_weight:
            problems.append(f"target_weight {d.target_weight} clipped for {d.ticker!r}")
        by_ticker[real] = Decision(ticker=real, action=d.action, target_weight=w, reason=d.reason)

    out = []
    for t in tickers:
        if t in by_ticker:
            out.append(by_ticker[t])
        else:
            problems.append(f"missing decision for {t}")
            out.append(Decision(ticker=t, action="HOLD", reason="missing_from_output"))
    return out, problems
