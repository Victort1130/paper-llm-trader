from papertrader.config import CostCfg, RiskCfg
from papertrader.risk.engine import RiskEngine
from papertrader.risk.rules import Order, RiskContext


def ctx(**kw):
    base = dict(equity=100_000, cash=100_000, positions={}, prices={"A": 100.0, "B": 50.0}, peak_equity=100_000)
    base.update(kw)
    return RiskContext(**base)


def buy(t="A", sh=100, px=100.0):
    return Order("2024-01-02", t, "BUY", sh, px)


def sell(t="A", sh=100, px=100.0):
    return Order("2024-01-02", t, "SELL", sh, px)


ENGINE = RiskEngine(RiskCfg(max_position_weight=0.2, max_trades_per_day=2, max_drawdown=0.15), CostCfg(commission_rate=0, slippage_bps=0))


def test_position_cap_trims_to_20_percent():
    approved, audit = ENGINE.review([buy(sh=500)], ctx())
    assert approved[0].shares == 200  # 20% of 100k at $100
    assert audit[0]["status"] == "trimmed"


def test_position_cap_counts_existing_holding():
    approved, audit = ENGINE.review([buy(sh=100)], ctx(positions={"A": 150}, cash=85_000))
    assert approved[0].shares == 50
    approved, audit = ENGINE.review([buy(sh=10)], ctx(positions={"A": 200}, cash=80_000))
    assert approved == [] and audit[0]["status"] == "rejected"


def test_no_leverage_limits_buys_to_cash():
    approved, _ = ENGINE.review([buy(sh=150)], ctx(cash=10_000))
    assert approved[0].shares == 100


def test_cash_is_shared_across_buys_on_the_same_day():
    approved, audit = ENGINE.review([buy("A", 100), buy("B", 200)], ctx(cash=15_000))
    assert sum(o.shares * o.ref_price for o in approved) <= 15_000
    assert [a["status"] for a in audit] == ["approved", "trimmed"]


def test_no_short_selling():
    approved, audit = ENGINE.review([sell(sh=10)], ctx())
    assert approved == [] and "short" in audit[0]["notes"][0]
    approved, _ = ENGINE.review([sell(sh=50)], ctx(positions={"A": 20}))
    assert approved[0].shares == 20


def test_daily_trade_limit_prefers_sells():
    orders = [buy("A", 10), buy("B", 10), sell("A", 5)]
    approved, audit = ENGINE.review(orders, ctx(positions={"A": 5}))
    assert len(approved) == 2
    assert approved[0].side == "SELL"
    assert any("daily trade limit" in n for a in audit for n in a["notes"])


def test_kill_switch_blocks_buys_but_allows_sells():
    c = ctx(equity=84_000, peak_equity=100_000, positions={"A": 10})
    approved, audit = ENGINE.review([buy("B", 10), sell("A", 10)], c)
    assert [o.side for o in approved] == ["SELL"]
    assert c.halted
    # once halted it stays halted even after equity recovers
    c2 = ctx(equity=99_000, peak_equity=100_000, halted=True)
    assert ENGINE.review([buy("B", 1)], c2)[0] == []


def test_costs_reduce_affordable_shares():
    eng = RiskEngine(RiskCfg(), CostCfg(commission_rate=0.01, slippage_bps=0))
    approved, _ = eng.review([buy(sh=100)], ctx(cash=10_000))
    assert approved[0].shares == 99
