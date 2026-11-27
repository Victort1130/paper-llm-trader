"""Proof that no decision can use information from after its decision date.

1. A view never contains rows after as_of, and asking for them raises.
2. Overwriting every price after day t leaves day t's indicators, prompt text
   and decisions byte-for-byte unchanged.
3. Computing from data truncated at t equals computing from full data viewed at t.
4. A decision made at the close of t fills at the open of the next trading day.
5. Across a whole backtest, no strategy call ever saw a date after its own day.
"""

import json

import numpy as np
import pandas as pd
import pytest

from papertrader.agent.llm_client import FakeLLM
from papertrader.agent.prompt import PromptTemplate, build_prompt
from papertrader.agent.schema import Decision, parse_decisions
from papertrader.backtest.loop import run_backtest
from papertrader.data.indicators import compute_indicators
from papertrader.data.pit import LookaheadError, MarketData, PointInTimeView
from papertrader.reporting.logger import read_jsonl
from papertrader.strategies.base import PortfolioState

from conftest import TICKERS

STATE = PortfolioState(weights={}, return_since_start=0.0, cash_weight=1.0, buying_halted=False)


def _prompt(cfg, view):
    return build_prompt(
        PromptTemplate.load(cfg.llm.prompt_file), view, TICKERS, {}, 0.0, 1.0,
        lookback=cfg.strategy.lookback_days, max_weight=0.2, max_trades=5, max_dd=0.15,
        anonymize=True, halted=False,
    )


def _decide(cfg, view):
    p = _prompt(cfg, view)
    resp = FakeLLM().complete(p.system, p.user, {})
    return p, parse_decisions(resp.text, p.alias_to_ticker, TICKERS)[0]


def test_view_contains_nothing_after_as_of(market):
    days = market.bars["AAA"].index
    rng = np.random.default_rng(0)
    for t in rng.choice(days[60:-5], size=25, replace=False):
        v = market.view(t)
        for tk in v.tickers:
            assert v.history(tk).index.max() <= t
        with pytest.raises(LookaheadError):
            v.history("AAA", end=t + pd.Timedelta(days=1))


def test_tampering_with_the_future_changes_nothing(cfg, bars):
    t = pd.Timestamp("2024-04-15")
    honest = MarketData(bars)
    tampered_bars = {}
    rng = np.random.default_rng(1)
    for tk, df in bars.items():
        df = df.copy()
        future = df.index > t
        df.loc[future, ["open", "high", "low", "close"]] = rng.uniform(1, 1000, (future.sum(), 4))
        tampered_bars[tk] = df
    tampered = MarketData(tampered_bars)
    assert not honest.bars["AAA"].equals(tampered.bars["AAA"])  # the future really differs

    for tk in TICKERS:
        assert compute_indicators(honest.view(t).history(tk)) == compute_indicators(tampered.view(t).history(tk))
    p1, d1 = _decide(cfg, honest.view(t))
    p2, d2 = _decide(cfg, tampered.view(t))
    assert p1.system == p2.system
    assert p1.user == p2.user
    assert d1 == d2


def test_truncated_data_equals_full_data_viewed_at_t(cfg, bars):
    for t in pd.to_datetime(["2024-03-28", "2024-05-02", "2024-06-20"]):
        truncated = MarketData({k: v.loc[:t] for k, v in bars.items()})
        full = MarketData(bars)
        p1, d1 = _decide(cfg, truncated.view(t))
        p2, d2 = _decide(cfg, full.view(t))
        assert p1.user == p2.user and d1 == d2


def test_decision_fills_at_next_open(cfg, market, tmp_path):
    class BuyOnce:
        name = "buy_once"

        def __init__(self):
            self.done = False

        def decide(self, view, state):
            if self.done:
                return []
            self.done = True
            return [Decision(ticker="AAA", action="BUY", target_weight=0.1)]

    run_backtest(cfg, market, BuyOnce(), tmp_path / "r")
    fills = read_jsonl(tmp_path / "r/fills.jsonl")
    assert len(fills) == 1
    f = fills[0]
    days = market.trading_days("BENCH", cfg.run.start, cfg.run.end)
    assert f["decision_date"] == str(days[0].date())
    assert f["fill_date"] == str(days[1].date())
    assert f["open_price"] == pytest.approx(market.price_on("AAA", days[1], "open"))
    assert f["fill_price"] == pytest.approx(f["open_price"] * (1 + cfg.costs.slippage_bps / 10_000))


def test_backtest_never_shows_a_strategy_the_future(cfg, market, tmp_path):
    seen = []

    class Spy:
        name = "spy"

        def decide(self, view: PointInTimeView, state):
            latest = max(view.history(tk).index.max() for tk in view.tickers)
            seen.append((view.as_of, latest))
            assert "BENCH" not in view.tickers  # only the configured universe
            return []

    run_backtest(cfg, market, Spy(), tmp_path / "spy")
    assert seen
    assert all(latest <= as_of for as_of, latest in seen)


def test_prompt_hides_identity_when_anonymized(cfg, market):
    p = _prompt(cfg, market.view(pd.Timestamp("2024-04-15")))
    for tk in TICKERS:
        assert tk not in p.user
    assert "2024" not in p.user
    data = json.loads(p.user.split("<market_data>")[1].split("</market_data>")[0])
    for alias, d in data.items():
        assert d["closes_rebased"][0] == 100.0
