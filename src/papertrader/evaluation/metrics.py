"""Performance metrics from a daily equity curve and the fill log."""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    """Most negative equity / running peak - 1 (e.g. -0.25 for a 25% drawdown)."""
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float | None:
    """Annualized Sharpe ratio of daily returns."""
    if len(returns) < 2:
        return None
    ex = returns - rf_annual / TRADING_DAYS
    sd = ex.std(ddof=1)
    if sd == 0 or not math.isfinite(sd):
        return None
    return float(ex.mean() / sd * math.sqrt(TRADING_DAYS))


def differential(agent: pd.Series, bench: pd.Series) -> dict:
    """Sharpe and t-stat of D = agent daily return - benchmark daily return.

    t = per-period Sharpe of D * sqrt(N) (Sharpe 1994). |t| above about 2 is
    the usual bar before calling an edge statistically distinguishable from 0.
    """
    ra, rb = daily_returns(agent), daily_returns(bench)
    d = (ra - rb).dropna()
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return {"n_days": n, "mean_daily": None, "sharpe_annual": None, "t_stat": None}
    per_period = d.mean() / d.std(ddof=1)
    return {
        "n_days": n,
        "mean_daily": float(d.mean()),
        "sharpe_annual": float(per_period * math.sqrt(TRADING_DAYS)),
        "t_stat": float(per_period * math.sqrt(n)),
    }


def round_trips(fills: list[dict]) -> list[dict]:
    """FIFO-match sells against earlier buys. Each sell fill closes one or more
    lots; its P&L includes both sides' commission and slippage."""
    lots: dict[str, deque] = defaultdict(deque)
    trades = []
    for f in sorted(fills, key=lambda x: (x["fill_date"], x["side"] != "SELL")):
        sh = f["shares"]
        if f["side"] == "BUY":
            unit_cost = (f["fill_price"] * sh + f["commission"]) / sh
            lots[f["ticker"]].append([sh, unit_cost, f["fill_date"]])
            continue
        unit_proceeds = (f["fill_price"] * sh - f["commission"]) / sh
        remaining, cost, opened = sh, 0.0, None
        while remaining > 0 and lots[f["ticker"]]:
            lot = lots[f["ticker"]][0]
            take = min(remaining, lot[0])
            cost += take * lot[1]
            opened = opened or lot[2]
            lot[0] -= take
            remaining -= take
            if lot[0] == 0:
                lots[f["ticker"]].popleft()
        matched = sh - remaining
        if matched:
            pnl = matched * unit_proceeds - cost
            trades.append(
                {"ticker": f["ticker"], "opened": opened, "closed": f["fill_date"],
                 "shares": matched, "pnl": pnl, "return": pnl / cost if cost else 0.0}
            )
    return trades


def trade_stats(fills: list[dict], equity: pd.Series) -> dict:
    trades = round_trips(fills)
    wins = [t for t in trades if t["pnl"] > 0]
    commissions = sum(f["commission"] for f in fills)
    slippage = sum(f["slippage_cost"] for f in fills)
    notional = defaultdict(float)
    for f in fills:
        notional[f["fill_date"]] += f["shares"] * f["fill_price"]
    eq_by_day = {str(pd.Timestamp(d).date()): v for d, v in equity.items()}
    daily_turnover = [notional[d] / eq_by_day[d] for d in notional if eq_by_day.get(d)]
    n_days = max(len(equity) - 1, 1)
    return {
        "n_fills": len(fills),
        "n_closed_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else None,
        "avg_trade_return": float(np.mean([t["return"] for t in trades])) if trades else None,
        "total_commission": commissions,
        "total_slippage": slippage,
        "total_costs": commissions + slippage,
        "avg_daily_turnover": sum(daily_turnover) / n_days,
        "annual_turnover": sum(daily_turnover) / n_days * TRADING_DAYS,
    }


def summarize(equity: pd.Series, rf_annual: float = 0.0) -> dict:
    r = daily_returns(equity)
    years = max(len(r), 1) / TRADING_DAYS
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    return {
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0,
        "ann_volatility": float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(r) > 1 else None,
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe(r, rf_annual),
    }
