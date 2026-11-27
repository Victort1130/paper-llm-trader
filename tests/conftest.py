from pathlib import Path

import pytest

from papertrader.config import Config, load_config
from papertrader.data.pit import MarketData
from papertrader.data.sources import synthetic_bars

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = load_config(ROOT / "configs/default.yaml")
    c.data.tickers = TICKERS
    c.data.benchmark = "BENCH"
    c.data.source = "synthetic"
    c.run.start = "2024-03-01"
    c.run.end = "2024-06-28"
    c.run.runs_dir = str(tmp_path / "runs")
    c.llm.prompt_file = str(ROOT / "configs/prompts/decision_v1.txt")
    return c


@pytest.fixture
def bars():
    return synthetic_bars(TICKERS + ["BENCH"], "2023-10-01", "2024-07-31", seed=7)


@pytest.fixture
def market(bars) -> MarketData:
    return MarketData(bars)
