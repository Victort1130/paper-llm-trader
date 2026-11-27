"""Typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RunCfg(BaseModel):
    initial_cash: float = 100_000
    start: str = "2024-01-01"
    end: str = "2025-12-31"
    warmup_days: int = 120
    runs_dir: str = "runs"


class DataCfg(BaseModel):
    source: Literal["yahoo", "csv", "synthetic"] = "yahoo"
    dir: str = "data/raw"
    tickers: list[str]
    benchmark: str = "QQQ"


class StrategyCfg(BaseModel):
    kind: Literal["llm", "ma_cross", "fake_llm"] = "llm"
    lookback_days: int = 20
    ma_fast: int = 20
    ma_slow: int = 50
    ma_target_weight: float = 0.10


class LLMCfg(BaseModel):
    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-opus-5-5"
    temperature: float | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "medium"
    max_tokens: int = 16_000
    prompt_file: str = "configs/prompts/decision_v1.txt"
    anonymize: bool = True
    fallbacks: str | None = "default"


class RiskCfg(BaseModel):
    max_position_weight: float = Field(0.20, gt=0, le=1)
    max_trades_per_day: int = Field(5, ge=0)
    max_drawdown: float = Field(0.15, gt=0, lt=1)


class CostCfg(BaseModel):
    commission_rate: float = Field(0.001, ge=0)
    min_commission: float = Field(0.0, ge=0)
    slippage_bps: float = Field(5.0, ge=0)
    min_order_weight: float = Field(0.01, ge=0, lt=1)


class EvalCfg(BaseModel):
    risk_free_rate: float = 0.0
    consistency_runs: int = 5
    forward_days: int = 5


class Config(BaseModel):
    run: RunCfg = RunCfg()
    data: DataCfg
    strategy: StrategyCfg = StrategyCfg()
    llm: LLMCfg = LLMCfg()
    risk: RiskCfg = RiskCfg()
    costs: CostCfg = CostCfg()
    evaluation: EvalCfg = EvalCfg()


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))


def dump_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False, allow_unicode=True)
