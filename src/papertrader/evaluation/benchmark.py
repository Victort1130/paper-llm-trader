"""Buy-and-hold benchmarks with the same timing and costs as the agent.

Both are bought at the open of the second trading day, the first open at
which the agent's own day-one decision could fill.
"""

from __future__ import annotations

import math

import pandas as pd

from papertrader.data.pit import MarketData
from papertrader.execution.costs import CostModel


def buy_and_hold(
    market: MarketData,
    weights: dict[str, float],
    days: list[pd.Timestamp],
    cash: float,
    costs: CostModel,
) -> tuple[pd.Series, float]:
    """Returns (daily close equity, total costs paid)."""
    buy_day = days[1]
    shares: dict[str, int] = {}
    paid = 0.0
    left = cash
    for t, w in weights.items():
        px = market.price_on(t, buy_day, "open")
        if px is None:
            continue
        fp = costs.fill_price("BUY", px)
        n = math.floor(w * cash / (fp * (1 + costs.commission_rate)))
        if n <= 0:
            continue
        comm = costs.commission(n * fp)
        left -= n * fp + comm
        paid += comm + (fp - px) * n
        shares[t] = n
    values = {}
    for d in days:
        if d < buy_day:
            values[d] = cash
            continue
        mv = 0.0
        for t, n in shares.items():
            p = market.last_close_on_or_before(t, d)
            mv += n * (p or 0.0)
        values[d] = left + mv
    return pd.Series(values, name="equity"), paid


def qqq_buy_and_hold(market, benchmark, days, cash, costs):
    return buy_and_hold(market, {benchmark: 1.0}, days, cash, costs)


def equal_weight_buy_and_hold(market, tickers, days, cash, costs):
    """Same stocks as the agent, equal weights, never rebalanced. Shares the
    agent's survivorship bias, so beating QQQ alone is not evidence of skill."""
    return buy_and_hold(market, {t: 1.0 / len(tickers) for t in tickers}, days, cash, costs)
