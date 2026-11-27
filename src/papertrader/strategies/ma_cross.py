"""Control strategy: 20/50-day moving average crossover on the same data."""

from __future__ import annotations

from papertrader.agent.schema import Decision
from papertrader.data.indicators import sma
from papertrader.data.pit import PointInTimeView
from papertrader.strategies.base import PortfolioState


class MACrossStrategy:
    name = "ma_cross"

    def __init__(self, tickers: list[str], fast: int = 20, slow: int = 50, weight: float = 0.10):
        self.tickers = tickers
        self.fast, self.slow, self.weight = fast, slow, weight

    def decide(self, view: PointInTimeView, state: PortfolioState) -> list[Decision]:
        out = []
        for t in self.tickers:
            closes = view.history(t)["close"].dropna()
            f, s = sma(closes, self.fast), sma(closes, self.slow)
            held = state.weights.get(t, 0.0) > 0
            if f is None or s is None:
                out.append(Decision(ticker=t, action="HOLD", reason="not enough history"))
            elif f > s and not held:
                out.append(
                    Decision(
                        ticker=t, action="BUY", target_weight=self.weight,
                        reason=f"SMA{self.fast} above SMA{self.slow}",
                    )
                )
            elif f < s and held:
                out.append(
                    Decision(
                        ticker=t, action="SELL", target_weight=0.0,
                        reason=f"SMA{self.fast} below SMA{self.slow}",
                    )
                )
            else:
                out.append(Decision(ticker=t, action="HOLD", reason="no crossover change"))
        return out
