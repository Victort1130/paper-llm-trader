"""Daily backtest loop.

Timeline for each trading day t:
  1. fill orders decided at t-1 at the open of t (plus slippage and commission)
  2. mark the portfolio to the close of t, update peak equity and drawdown
  3. the strategy sees a view as of the close of t and returns decisions
  4. decisions become orders, the risk engine trims or rejects them
  5. approved orders wait for the open of t+1
So a decision can never trade on the price it was made from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from papertrader.agent.schema import Decision
from papertrader.config import Config, dump_config
from papertrader.data.pit import MarketData
from papertrader.execution.costs import CostModel
from papertrader.execution.portfolio import Portfolio
from papertrader.execution.simulator import execute
from papertrader.reporting.logger import JsonlLogger
from papertrader.risk.engine import RiskEngine
from papertrader.risk.rules import Order, RiskContext
from papertrader.strategies.base import PortfolioState, Strategy


@dataclass
class RunResult:
    run_dir: Path
    equity: pd.DataFrame  # index date; columns equity, cash, drawdown, halted
    strategy: str


def decisions_to_orders(
    decisions: list[Decision],
    date: str,
    positions: dict[str, int],
    prices: dict[str, float],
    equity: float,
    min_order_weight: float = 0.0,
) -> tuple[list[Order], list[dict]]:
    """Turn target weights into share orders at today's close.

    BUY must raise the position and SELL must lower it; anything else is
    treated as HOLD and noted.
    """
    orders, skipped = [], []
    for d in decisions:
        if d.action == "HOLD":
            continue
        price = prices.get(d.ticker)
        if price is None or price <= 0:
            skipped.append({"date": date, "ticker": d.ticker, "why": "no price today"})
            continue
        held = positions.get(d.ticker, 0)
        target = math.floor(d.target_weight * equity / price)
        delta = target - held
        if delta != 0 and abs(delta) * price < min_order_weight * equity and target != 0:
            skipped.append(
                {"date": date, "ticker": d.ticker, "why": f"order below {min_order_weight:.0%} of equity"}
            )
            continue
        if d.action == "BUY" and delta > 0:
            orders.append(Order(date, d.ticker, "BUY", delta, price, d.target_weight, d.reason))
        elif d.action == "SELL" and delta < 0:
            orders.append(Order(date, d.ticker, "SELL", -delta, price, d.target_weight, d.reason))
        else:
            skipped.append(
                {"date": date, "ticker": d.ticker, "why": f"{d.action} to {d.target_weight} needs no trade"}
            )
    return orders, skipped


def run_backtest(cfg: Config, market: MarketData, strategy: Strategy, run_dir: str | Path) -> RunResult:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    (run_dir / "strategy.txt").write_text(strategy.name + "\n", encoding="utf-8")
    log_dec = JsonlLogger(run_dir / "decisions.jsonl")
    log_ord = JsonlLogger(run_dir / "orders.jsonl")
    log_fill = JsonlLogger(run_dir / "fills.jsonl")

    tickers = cfg.data.tickers
    costs = CostModel.from_cfg(cfg.costs)
    risk = RiskEngine(cfg.risk, cfg.costs)
    pf = Portfolio(cash=cfg.run.initial_cash)
    days = market.trading_days(cfg.data.benchmark, cfg.run.start, cfg.run.end)
    if len(days) < 2:
        raise ValueError("need at least two trading days")

    pending: list[Order] = []
    peak = cfg.run.initial_cash
    halted = False
    rows = []
    for i, t in enumerate(days):
        ds = str(t.date())
        # 1. fills at today's open
        if pending:
            fills, unfilled = execute(pending, pf, market.prices_on(tickers, t, "open"), ds, costs)
            for f in fills:
                log_fill.write(f.to_dict())
            for u in unfilled:
                log_ord.write({**u, "status": "unfilled"})
            pending = []

        # 2. mark to close
        closes = {}
        for tk in tickers:
            p = market.last_close_on_or_before(tk, t)
            if p is not None:
                closes[tk] = p
        equity = pf.equity(closes)
        peak = max(peak, equity)
        dd = equity / peak - 1.0
        if dd <= -cfg.risk.max_drawdown:
            halted = True
        rows.append({"date": t, "equity": equity, "cash": pf.cash, "drawdown": dd, "halted": halted})

        if i == len(days) - 1:
            break  # no next open to fill a decision made today

        # 3. decide from a point-in-time view
        view = market.view(t, tickers)
        weights = pf.weights(closes)
        state = PortfolioState(
            weights=weights,
            return_since_start=equity / cfg.run.initial_cash - 1.0,
            cash_weight=pf.cash / equity if equity > 0 else 1.0,
            buying_halted=halted,
        )
        decisions = strategy.decide(view, state)
        for d in decisions:
            log_dec.write({"date": ds, **d.model_dump(), "weight_before": round(weights.get(d.ticker, 0.0), 6)})

        # 4. orders and risk review (prices = closes that exist today)
        today = {tk: p for tk, p in closes.items() if view.has_bar_today(tk)}
        orders, skipped = decisions_to_orders(
            decisions, ds, pf.positions, today, equity, cfg.costs.min_order_weight
        )
        for s in skipped:
            log_ord.write({**s, "status": "no_order"})
        ctx = RiskContext(
            equity=equity, cash=pf.cash, positions=dict(pf.positions), prices=today,
            peak_equity=peak, halted=halted,
        )
        approved, audit = risk.review(orders, ctx)
        halted = ctx.halted
        for a in audit:
            log_ord.write(a)
        pending = approved

    eq = pd.DataFrame(rows).set_index("date")
    eq.to_csv(run_dir / "equity.csv")
    return RunResult(run_dir=run_dir, equity=eq, strategy=strategy.name)
