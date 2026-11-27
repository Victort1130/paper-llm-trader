"""Prompt assembly from a point-in-time view.

With anonymize=True the prompt contains no ticker symbols, no calendar dates
and no absolute price levels, which limits how much the model can lean on
memorised market history.
"""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass
from pathlib import Path

from papertrader.data.indicators import compute_indicators
from papertrader.data.pit import PointInTimeView

_SPLIT_SYSTEM = "=== SYSTEM ==="
_SPLIT_USER = "=== USER ==="


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: string.Template
    user: string.Template
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "PromptTemplate":
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        if _SPLIT_SYSTEM not in raw or _SPLIT_USER not in raw:
            raise ValueError(f"{p} must contain {_SPLIT_SYSTEM} and {_SPLIT_USER}")
        system, user = raw.split(_SPLIT_SYSTEM, 1)[1].split(_SPLIT_USER, 1)
        return cls(
            name=p.stem,
            system=string.Template(system.strip()),
            user=string.Template(user.strip()),
            sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True)
class BuiltPrompt:
    system: str
    user: str
    alias_to_ticker: dict[str, str]
    market: dict  # the structured data that was rendered, for logging/tests


def aliases(tickers: list[str], anonymize: bool) -> dict[str, str]:
    """ticker -> alias. Order follows the config list, so it is stable."""
    if not anonymize:
        return {t: t for t in tickers}
    letters = string.ascii_uppercase
    out = {}
    for i, t in enumerate(tickers):
        label = letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]
        out[t] = f"Stock_{label}"
    return out


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def build_prompt(
    template: PromptTemplate,
    view: PointInTimeView,
    tickers: list[str],
    weights: dict[str, float],
    portfolio_return: float,
    cash_weight: float,
    *,
    lookback: int,
    max_weight: float,
    max_trades: int,
    max_dd: float,
    anonymize: bool,
    halted: bool,
) -> BuiltPrompt:
    t2a = aliases(tickers, anonymize)
    market = {}
    for t in tickers:
        hist = view.history(t)
        if hist.empty:
            continue
        closes = hist["close"].dropna().tail(lookback)
        base = float(closes.iloc[0])
        ind = compute_indicators(hist)
        market[t2a[t]] = {
            "closes_rebased": [round(float(c) / base * 100, 2) for c in closes],
            "indicators": {k: _r(v) for k, v in ind.items()},
            "current_weight": round(weights.get(t, 0.0), 4),
        }
    portfolio = {
        "return_since_start": round(portfolio_return, 4),
        "cash_weight": round(cash_weight, 4),
        "buying_halted_by_drawdown_rule": halted,
    }
    if not anonymize:
        portfolio["as_of"] = str(view.as_of.date())

    fields = {
        "max_weight": max_weight,
        "max_weight_pct": round(max_weight * 100, 2),
        "max_trades": max_trades,
        "max_dd_pct": round(max_dd * 100, 2),
        "lookback": lookback,
        "portfolio_json": json.dumps(portfolio, sort_keys=True),
        "market_json": json.dumps(market, sort_keys=True),
        "stock_list": ", ".join(t2a[t] for t in tickers if t2a[t] in market),
    }
    return BuiltPrompt(
        system=template.system.substitute(fields),
        user=template.user.substitute(fields),
        alias_to_ticker={a: t for t, a in t2a.items()},
        market=market,
    )
