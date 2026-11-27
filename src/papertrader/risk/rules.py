"""Risk rules: plain functions, no LLM involved.

Each rule takes an order and the current context and returns the order
unchanged, a smaller order, or None (rejected) plus a note saying why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class Order:
    date: str
    ticker: str
    side: Side
    shares: int
    ref_price: float  # last close at decision time
    target_weight: float = 0.0
    reason: str = ""


@dataclass
class RiskContext:
    equity: float
    cash: float
    positions: dict[str, int]
    prices: dict[str, float]
    peak_equity: float
    halted: bool = False
    orders_today: int = 0
    committed_cash: float = 0.0  # estimated cost of BUYs already approved today
    est_cost_rate: float = 0.0  # commission + slippage, used to price BUYs conservatively
    notes: list[str] = field(default_factory=list)

    @property
    def drawdown(self) -> float:
        return 0.0 if self.peak_equity <= 0 else self.equity / self.peak_equity - 1.0


@dataclass(frozen=True)
class RuleResult:
    order: Order | None
    note: str | None = None


def drawdown_kill_switch(order: Order, ctx: RiskContext, max_drawdown: float) -> RuleResult:
    """Once drawdown from peak reaches max_drawdown, block all BUYs for good."""
    if not ctx.halted and ctx.drawdown <= -max_drawdown:
        ctx.halted = True
    if ctx.halted and order.side == "BUY":
        return RuleResult(None, f"kill switch: drawdown {ctx.drawdown:.2%} hit limit {max_drawdown:.0%}")
    return RuleResult(order)


def max_trades_per_day(order: Order, ctx: RiskContext, limit: int) -> RuleResult:
    if ctx.orders_today >= limit:
        return RuleResult(None, f"daily trade limit {limit} reached")
    return RuleResult(order)


def no_short(order: Order, ctx: RiskContext) -> RuleResult:
    if order.side != "SELL":
        return RuleResult(order)
    held = ctx.positions.get(order.ticker, 0)
    if held <= 0:
        return RuleResult(None, "no position to sell (short selling not allowed)")
    if order.shares > held:
        return RuleResult(replace(order, shares=held), f"sell trimmed {order.shares} -> {held} (no short)")
    return RuleResult(order)


def no_leverage(order: Order, ctx: RiskContext) -> RuleResult:
    """A BUY may only use cash on hand minus BUYs already approved today."""
    if order.side != "BUY":
        return RuleResult(order)
    unit_cost = order.ref_price * (1 + ctx.est_cost_rate)
    available = ctx.cash - ctx.committed_cash
    affordable = math.floor(max(available, 0.0) / unit_cost) if unit_cost > 0 else 0
    if affordable <= 0:
        return RuleResult(None, "insufficient cash (no leverage)")
    if order.shares > affordable:
        return RuleResult(
            replace(order, shares=affordable), f"buy trimmed {order.shares} -> {affordable} (no leverage)"
        )
    return RuleResult(order)


def max_position_weight(order: Order, ctx: RiskContext, limit: float) -> RuleResult:
    """Post-trade position value at the reference price must be <= limit * equity."""
    if order.side != "BUY":
        return RuleResult(order)
    held = ctx.positions.get(order.ticker, 0)
    cap_shares = math.floor(limit * ctx.equity / order.ref_price) - held
    if cap_shares <= 0:
        return RuleResult(None, f"position already at {limit:.0%} cap")
    if order.shares > cap_shares:
        return RuleResult(
            replace(order, shares=cap_shares), f"buy trimmed {order.shares} -> {cap_shares} ({limit:.0%} cap)"
        )
    return RuleResult(order)
