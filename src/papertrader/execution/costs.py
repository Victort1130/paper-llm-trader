"""Commission and slippage, all set from config."""

from __future__ import annotations

from dataclasses import dataclass

from papertrader.config import CostCfg


@dataclass(frozen=True)
class CostModel:
    commission_rate: float = 0.001
    min_commission: float = 0.0
    slippage_bps: float = 5.0

    @classmethod
    def from_cfg(cls, cfg: CostCfg) -> "CostModel":
        return cls(cfg.commission_rate, cfg.min_commission, cfg.slippage_bps)

    def fill_price(self, side: str, open_price: float) -> float:
        """Buys pay above the open, sells receive below it."""
        s = self.slippage_bps / 10_000
        return open_price * (1 + s) if side == "BUY" else open_price * (1 - s)

    def commission(self, notional: float) -> float:
        if notional <= 0:
            return 0.0
        return max(notional * self.commission_rate, self.min_commission)
