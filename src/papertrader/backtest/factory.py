"""Builds strategies from config and runs them N times."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from papertrader.agent.llm_client import FakeLLM
from papertrader.backtest.loop import RunResult, run_backtest
from papertrader.config import Config
from papertrader.data.pit import MarketData
from papertrader.reporting.logger import JsonlLogger
from papertrader.strategies.base import Strategy
from papertrader.strategies.llm import LLMStrategy
from papertrader.strategies.ma_cross import MACrossStrategy


def make_strategy(cfg: Config, kind: str, run_dir: Path, seed: int = 0, fake_noise: float = 0.0) -> Strategy:
    call_log = JsonlLogger(run_dir / "llm_calls.jsonl")
    if kind == "llm":
        from papertrader.agent.llm_client import AnthropicClient

        return LLMStrategy(cfg, AnthropicClient(cfg.llm), call_log)
    if kind == "fake_llm":
        s = LLMStrategy(cfg, FakeLLM(weight=min(0.15, cfg.risk.max_position_weight), noise=fake_noise, seed=seed), call_log)
        s.name = "fake_llm"
        return s
    if kind == "ma_cross":
        sc = cfg.strategy
        return MACrossStrategy(cfg.data.tickers, sc.ma_fast, sc.ma_slow, sc.ma_target_weight)
    raise ValueError(f"unknown strategy {kind}")


def new_experiment_dir(cfg: Config, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path(cfg.run.runs_dir) / f"{stamp}_{label}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def run_many(
    cfg: Config, market: MarketData, kind: str, n: int, exp_dir: Path, fake_noise: float = 0.0
) -> list[RunResult]:
    """Same config, n independent runs (fresh client each time, no caching)."""
    results = []
    for k in range(n):
        run_dir = exp_dir / f"{kind}_run{k + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        strat = make_strategy(cfg, kind, run_dir, seed=k, fake_noise=fake_noise)
        results.append(run_backtest(cfg, market, strat, run_dir))
    return results
