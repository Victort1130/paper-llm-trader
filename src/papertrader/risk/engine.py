"""Applies the risk rules in a fixed order and records every outcome."""

from __future__ import annotations

from dataclasses import asdict

from papertrader.config import CostCfg, RiskCfg
from papertrader.risk import rules
from papertrader.risk.rules import Order, RiskContext


class RiskEngine:
    def __init__(self, cfg: RiskCfg, costs: CostCfg):
        self.cfg = cfg
        self.est_cost_rate = costs.commission_rate + costs.slippage_bps / 10_000

    def review(self, orders: list[Order], ctx: RiskContext) -> tuple[list[Order], list[dict]]:
        """Return approved orders and an audit record for every proposed order.

        SELLs are reviewed first so they take daily trade slots before BUYs.
        Cash from SELLs is not counted toward BUYs on the same day.
        """
        ctx.est_cost_rate = self.est_cost_rate
        approved: list[Order] = []
        audit: list[dict] = []
        for order in sorted(orders, key=lambda o: (o.side != "SELL", o.ticker)):
            current: Order | None = order
            notes: list[str] = []
            for rule in (
                lambda o: rules.drawdown_kill_switch(o, ctx, self.cfg.max_drawdown),
                lambda o: rules.max_trades_per_day(o, ctx, self.cfg.max_trades_per_day),
                lambda o: rules.no_short(o, ctx),
                lambda o: rules.max_position_weight(o, ctx, self.cfg.max_position_weight),
                lambda o: rules.no_leverage(o, ctx),
            ):
                res = rule(current)
                if res.note:
                    notes.append(res.note)
                current = res.order
                if current is None:
                    break
            if current is not None and current.shares <= 0:
                current, notes = None, notes + ["zero shares after checks"]
            status = "rejected" if current is None else ("trimmed" if current != order else "approved")
            audit.append(
                {
                    **asdict(order),
                    "status": status,
                    "approved_shares": 0 if current is None else current.shares,
                    "notes": notes,
                }
            )
            if current is not None:
                approved.append(current)
                ctx.orders_today += 1
                if current.side == "BUY":
                    ctx.committed_cash += current.shares * current.ref_price * (1 + self.est_cost_rate)
        return approved, audit
