import json

from papertrader.agent.schema import parse_decisions
from papertrader.backtest.loop import decisions_to_orders
from papertrader.agent.schema import Decision

ALIASES = {"Stock_A": "AAA", "Stock_B": "BBB"}
TICKERS = ["AAA", "BBB"]


def test_bad_json_means_hold_everything():
    decisions, problems = parse_decisions("not json {", ALIASES, TICKERS)
    assert all(d.action == "HOLD" for d in decisions)
    assert problems and problems[0].startswith("parse_error")


def test_aliases_are_mapped_back_and_missing_become_hold():
    text = json.dumps({"decisions": [
        {"ticker": "Stock_B", "action": "BUY", "target_weight": 0.1, "reason": "up"},
        {"ticker": "Stock_Z", "action": "BUY", "target_weight": 0.1, "reason": "?"},
    ]})
    decisions, problems = parse_decisions(text, ALIASES, TICKERS)
    assert decisions[0].ticker == "AAA" and decisions[0].action == "HOLD"
    assert decisions[1].ticker == "BBB" and decisions[1].action == "BUY"
    assert any("Stock_Z" in p for p in problems)
    assert any("missing" in p for p in problems)


def test_out_of_range_weight_is_clipped():
    text = json.dumps({"decisions": [{"ticker": "Stock_A", "action": "BUY", "target_weight": 3, "reason": ""}]})
    decisions, problems = parse_decisions(text, ALIASES, ["AAA"])
    assert decisions[0].target_weight == 1.0
    assert any("clipped" in p for p in problems)


def test_target_weights_become_share_orders():
    ds = [
        Decision(ticker="AAA", action="BUY", target_weight=0.1),
        Decision(ticker="BBB", action="SELL", target_weight=0.0),
        Decision(ticker="CCC", action="BUY", target_weight=0.05),  # already above target
    ]
    orders, skipped = decisions_to_orders(
        ds, "d", {"BBB": 30, "CCC": 100}, {"AAA": 50.0, "BBB": 10.0, "CCC": 100.0}, 100_000
    )
    assert [(o.ticker, o.side, o.shares) for o in orders] == [("AAA", "BUY", 200), ("BBB", "SELL", 30)]
    assert skipped[0]["ticker"] == "CCC"


def test_tiny_orders_are_skipped():
    ds = [Decision(ticker="AAA", action="BUY", target_weight=0.1)]
    orders, skipped = decisions_to_orders(ds, "d", {"AAA": 195}, {"AAA": 50.0}, 100_000, min_order_weight=0.01)
    assert orders == [] and "below" in skipped[0]["why"]
