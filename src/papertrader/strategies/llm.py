"""LLM strategy: prompt -> model -> parsed decisions, with every call logged."""

from __future__ import annotations

import traceback

from papertrader.agent.llm_client import LLMClient
from papertrader.agent.prompt import PromptTemplate, build_prompt
from papertrader.agent.schema import Decision, hold_all, output_schema, parse_decisions
from papertrader.config import Config
from papertrader.data.pit import PointInTimeView
from papertrader.reporting.logger import JsonlLogger
from papertrader.strategies.base import PortfolioState


class LLMStrategy:
    name = "llm"

    def __init__(self, cfg: Config, client: LLMClient, call_log: JsonlLogger | None = None):
        self.cfg = cfg
        self.client = client
        self.template = PromptTemplate.load(cfg.llm.prompt_file)
        self.schema = output_schema(cfg.risk.max_position_weight)
        self.call_log = call_log

    def decide(self, view: PointInTimeView, state: PortfolioState) -> list[Decision]:
        tickers = self.cfg.data.tickers
        prompt = build_prompt(
            self.template,
            view,
            tickers,
            state.weights,
            state.return_since_start,
            state.cash_weight,
            lookback=self.cfg.strategy.lookback_days,
            max_weight=self.cfg.risk.max_position_weight,
            max_trades=self.cfg.risk.max_trades_per_day,
            max_dd=self.cfg.risk.max_drawdown,
            anonymize=self.cfg.llm.anonymize,
            halted=state.buying_halted,
        )
        record = {
            "date": str(view.as_of.date()),
            "prompt_template": self.template.name,
            "prompt_sha256": self.template.sha256,
            "system": prompt.system,
            "user": prompt.user,
            "alias_to_ticker": prompt.alias_to_ticker,
        }
        try:
            resp = self.client.complete(prompt.system, prompt.user, self.schema)
        except Exception as e:  # API down, auth error, etc.: never trade on failure
            record.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            self._log(record)
            return hold_all(tickers, "llm_error")

        decisions, problems = parse_decisions(resp.text, prompt.alias_to_ticker, tickers)
        if resp.stop_reason == "refusal":
            problems.append("refusal")
            decisions = hold_all(tickers, "refusal")
        record.update(
            model_requested=resp.model_requested,
            model_served=resp.model_served,
            fallback_used=resp.fallback_used,
            params=resp.params,
            stop_reason=resp.stop_reason,
            request_id=resp.request_id,
            latency_s=resp.latency_s,
            usage=resp.usage,
            raw_output=resp.text,
            parsed=[d.model_dump() for d in decisions],
            problems=problems,
        )
        self._log(record)
        return decisions

    def _log(self, record: dict) -> None:
        if self.call_log is not None:
            self.call_log.write(record)
