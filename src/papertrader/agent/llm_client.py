"""LLM clients.

`AnthropicClient` calls the Claude Messages API. `FakeLLM` is a deterministic
stand-in used by tests and offline demos; it never touches the network.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from papertrader.config import LLMCfg


@dataclass
class LLMResponse:
    text: str
    model_requested: str
    model_served: str
    stop_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    latency_s: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False


class LLMClient(Protocol):
    def complete(self, system: str, user: str, schema: dict) -> LLMResponse: ...


class AnthropicClient:
    """Claude via the official `anthropic` SDK. Reads ANTHROPIC_API_KEY."""

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, cfg: LLMCfg):
        import anthropic  # optional dependency

        if cfg.temperature is not None:
            raise ValueError(
                "Current Claude models and SDK do not accept temperature; set llm.temperature to null."
            )
        self.cfg = cfg
        self.client = anthropic.Anthropic()

    def _params(self, schema: dict) -> dict[str, Any]:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.cfg.effort:
            output_config["effort"] = self.cfg.effort
        params: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "output_config": output_config,
        }
        return params

    def complete(self, system: str, user: str, schema: dict) -> LLMResponse:
        params = self._params(schema)
        t0 = time.monotonic()
        if self.cfg.fallbacks:
            resp = self.client.beta.messages.create(
                **params,
                system=system,
                messages=[{"role": "user", "content": user}],
                betas=[self.FALLBACK_BETA],
                fallbacks=self.cfg.fallbacks,
            )
        else:
            resp = self.client.messages.create(
                **params, system=system, messages=[{"role": "user", "content": user}]
            )
        latency = time.monotonic() - t0
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
        iterations = usage.get("iterations") or []
        fallback_used = any(
            (it.get("type") if isinstance(it, dict) else getattr(it, "type", None))
            == "fallback_message"
            for it in iterations
        )
        return LLMResponse(
            text=text,
            model_requested=self.cfg.model,
            model_served=resp.model,
            stop_reason=resp.stop_reason,
            usage=usage,
            request_id=getattr(resp, "_request_id", None),
            latency_s=round(latency, 3),
            params={**params, "fallbacks": self.cfg.fallbacks},
            fallback_used=fallback_used,
        )


_MARKET_RE = re.compile(r"<market_data>\s*(.*?)\s*</market_data>", re.S)
_PORTFOLIO_RE = re.compile(r"<portfolio>\s*(.*?)\s*</portfolio>", re.S)


class FakeLLM:
    """Rule-based stand-in that reads only the prompt text.

    Momentum rule: BUY to `weight` when price is above its 20-day average and
    RSI < 70; SELL to 0 when below; otherwise HOLD. With noise > 0 it flips
    some actions at random (seeded), which imitates run-to-run variation.
    """

    def __init__(self, weight: float = 0.15, noise: float = 0.0, seed: int = 0):
        self.weight = weight
        self.noise = noise
        self.rng = random.Random(seed)
        self.calls = 0

    def complete(self, system: str, user: str, schema: dict) -> LLMResponse:
        self.calls += 1
        market = json.loads(_MARKET_RE.search(user).group(1))
        portfolio = json.loads(_PORTFOLIO_RE.search(user).group(1))
        decisions = []
        for alias in sorted(market):
            ind = market[alias]["indicators"]
            above = ind.get("close_vs_sma20")
            rsi = ind.get("rsi_14")
            if above is None or rsi is None:
                action, w, why = "HOLD", 0.0, "not enough history"
            elif above > 0 and rsi < 70 and not portfolio["buying_halted_by_drawdown_rule"]:
                action, w, why = "BUY", self.weight, "price above 20-day average, RSI below 70"
            elif above < 0:
                action, w, why = "SELL", 0.0, "price below 20-day average"
            else:
                action, w, why = "HOLD", 0.0, "no clear signal"
            if self.noise and self.rng.random() < self.noise:
                action = self.rng.choice(["BUY", "SELL", "HOLD"])
                w = self.weight if action == "BUY" else 0.0
                why += " (noise)"
            decisions.append({"ticker": alias, "action": action, "target_weight": w, "reason": why})
        return LLMResponse(
            text=json.dumps({"decisions": decisions}),
            model_requested="fake-llm",
            model_served="fake-llm",
            stop_reason="end_turn",
            usage={"input_tokens": len(system + user) // 4, "output_tokens": 0},
            params={"weight": self.weight, "noise": self.noise},
        )
