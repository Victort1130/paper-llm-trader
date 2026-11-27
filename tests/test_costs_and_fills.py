import pytest

from papertrader.execution.costs import CostModel
from papertrader.execution.portfolio import Portfolio
from papertrader.execution.simulator import execute
from papertrader.risk.rules import Order

COSTS = CostModel(commission_rate=0.001, min_commission=1.0, slippage_bps=10)


def test_slippage_direction_and_commission():
    assert COSTS.fill_price("BUY", 100) == pytest.approx(100.1)
    assert COSTS.fill_price("SELL", 100) == pytest.approx(99.9)
    assert COSTS.commission(10_000) == pytest.approx(10.0)
    assert COSTS.commission(100) == 1.0  # minimum applies


def test_round_trip_loses_exactly_the_costs():
    pf = Portfolio(cash=10_000)
    execute([Order("d0", "A", "BUY", 50, 100)], pf, {"A": 100.0}, "d1", COSTS)
    execute([Order("d1", "A", "SELL", 50, 100)], pf, {"A": 100.0}, "d2", COSTS)
    buy_cost = 50 * 100.1 + max(50 * 100.1 * 0.001, 1)
    sell_proceeds = 50 * 99.9 - max(50 * 99.9 * 0.001, 1)
    assert pf.cash == pytest.approx(10_000 - buy_cost + sell_proceeds)
    assert pf.positions == {}


def test_gap_up_trims_buy_to_available_cash():
    pf = Portfolio(cash=1_000)
    fills, _ = execute([Order("d0", "A", "BUY", 10, 95)], pf, {"A": 120.0}, "d1", COSTS)
    assert fills[0].shares == 8
    assert pf.cash >= 0


def test_sells_fill_before_buys_and_missing_price_is_unfilled():
    pf = Portfolio(cash=0, positions={"A": 10})
    orders = [Order("d0", "B", "BUY", 5, 100), Order("d0", "A", "SELL", 10, 100), Order("d0", "C", "BUY", 1, 10)]
    fills, unfilled = execute(orders, pf, {"A": 100.0, "B": 100.0}, "d1", COSTS)
    assert [f.side for f in fills] == ["SELL", "BUY"]
    assert unfilled[0]["ticker"] == "C"


def test_portfolio_refuses_negative_cash():
    pf = Portfolio(cash=10)
    with pytest.raises(AssertionError):
        pf.apply("A", "BUY", 1, 100, 0)
