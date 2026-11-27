import math

import numpy as np
import pandas as pd
import pytest

from papertrader.evaluation import metrics


def _eq(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_max_drawdown():
    assert metrics.max_drawdown(_eq([100, 120, 60, 90, 130])) == pytest.approx(-0.5)
    assert metrics.max_drawdown(_eq([100, 101, 102])) == 0.0


def test_sharpe_matches_formula():
    r = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001])
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert metrics.sharpe(r) == pytest.approx(expected)
    assert metrics.sharpe(pd.Series([0.01, 0.01])) is None  # zero variance


def test_summary_of_constant_growth():
    eq = _eq(100 * 1.001 ** np.arange(253))
    s = metrics.summarize(eq)
    assert s["total_return"] == pytest.approx(1.001 ** 252 - 1)
    assert s["cagr"] == pytest.approx(1.001 ** 252 - 1)
    assert s["ann_volatility"] == pytest.approx(0.0, abs=1e-10)
    assert s["max_drawdown"] == 0.0


def test_differential_t_stat_is_sharpe_times_sqrt_n():
    rng = np.random.default_rng(3)
    a = _eq(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 300)))
    b = _eq(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300)))
    d = metrics.differential(a, b)
    diff = (a.pct_change() - b.pct_change()).dropna()
    per = diff.mean() / diff.std(ddof=1)
    assert d["t_stat"] == pytest.approx(per * math.sqrt(len(diff)))
    assert d["sharpe_annual"] == pytest.approx(per * math.sqrt(252))
    same = metrics.differential(a, a)
    assert same["t_stat"] is None


def _fill(date, side, sh, px, comm=0.0):
    return {"fill_date": date, "ticker": "A", "side": side, "shares": sh, "fill_price": px,
            "commission": comm, "slippage_cost": 0.0}


def test_round_trips_fifo_and_win_rate():
    fills = [
        _fill("2024-01-02", "BUY", 10, 100),
        _fill("2024-01-03", "BUY", 10, 110),
        _fill("2024-01-04", "SELL", 15, 105),  # 10 @100 + 5 @110 -> pnl 1575 - 1550 = 25
        _fill("2024-01-05", "SELL", 5, 100),   # 5 @110 -> pnl -50
    ]
    trades = metrics.round_trips(fills)
    assert [round(t["pnl"], 6) for t in trades] == [25.0, -50.0]
    eq = _eq([1000, 1000, 1000, 1000, 1000])
    stats = metrics.trade_stats(fills, eq)
    assert stats["win_rate"] == 0.5
    assert stats["n_closed_trades"] == 2


def test_costs_and_turnover():
    fills = [_fill("2024-01-02", "BUY", 10, 100, comm=1.0)]
    fills[0]["slippage_cost"] = 0.5
    eq = _eq([2000, 2000, 2000])
    stats = metrics.trade_stats(fills, eq)
    assert stats["total_costs"] == pytest.approx(1.5)
    assert stats["avg_daily_turnover"] == pytest.approx((1000 / 2000) / 2)
