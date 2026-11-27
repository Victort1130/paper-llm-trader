"""Cash and positions of the simulated account."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(sh * prices[t] for t, sh in self.positions.items() if sh)

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {}
        return {t: sh * prices[t] / eq for t, sh in self.positions.items() if sh}

    def apply(self, ticker: str, side: str, shares: int, price: float, commission: float) -> None:
        if side == "BUY":
            self.cash -= shares * price + commission
            self.positions[ticker] = self.positions.get(ticker, 0) + shares
        else:
            self.cash += shares * price - commission
            self.positions[ticker] = self.positions.get(ticker, 0) - shares
            if self.positions[ticker] == 0:
                del self.positions[ticker]
        if self.cash < -1e-6:
            raise AssertionError(f"cash went negative: {self.cash}")
        if self.positions.get(ticker, 0) < 0:
            raise AssertionError(f"short position in {ticker}")
