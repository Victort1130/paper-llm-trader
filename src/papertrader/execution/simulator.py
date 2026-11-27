"""Simulated fills. This is the only place an "order" turns into a trade,
and it only changes an in-memory Portfolio. Nothing is sent anywhere."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from papertrader.execution.costs import CostModel
from papertrader.execution.portfolio import Portfolio
from papertrader.risk.rules import Order


@dataclass(frozen=True)
class Fill:
    decision_date: str
    fill_date: str
    ticker: str
    side: str
    shares: int
    open_price: float
    fill_price: float
    commission: float
    slippage_cost: float

    def to_dict(self) -> dict:
        return asdict(self)


def execute(
    orders: list[Order],
    portfolio: Portfolio,
    open_prices: dict[str, float],
    fill_date: str,
    costs: CostModel,
) -> tuple[list[Fill], list[dict]]:
    """Fill orders at `fill_date` open plus slippage. SELLs fill before BUYs.

    A BUY that no longer fits in cash (the open gapped up) is trimmed. An order
    for a ticker with no open price that day is not filled.
    """
    fills: list[Fill] = []
    unfilled: list[dict] = []
    for o in sorted(orders, key=lambda o: o.side != "SELL"):
        px = open_prices.get(o.ticker)
        if px is None or not math.isfinite(px) or px <= 0:
            unfilled.append({**asdict(o), "why": "no open price on fill date"})
            continue
        fp = costs.fill_price(o.side, px)
        shares = o.shares
        if o.side == "SELL":
            shares = min(shares, portfolio.positions.get(o.ticker, 0))
        else:
            while shares > 0 and shares * fp + costs.commission(shares * fp) > portfolio.cash:
                shares -= 1
        if shares <= 0:
            unfilled.append({**asdict(o), "why": "nothing to fill after cash/position check"})
            continue
        commission = costs.commission(shares * fp)
        portfolio.apply(o.ticker, o.side, shares, fp, commission)
        fills.append(
            Fill(
                decision_date=o.date,
                fill_date=fill_date,
                ticker=o.ticker,
                side=o.side,
                shares=shares,
                open_price=px,
                fill_price=fp,
                commission=commission,
                slippage_cost=abs(fp - px) * shares,
            )
        )
    return fills, unfilled
