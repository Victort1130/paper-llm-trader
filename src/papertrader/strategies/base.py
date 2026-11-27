"""Strategy interface. A strategy sees a PointInTimeView and nothing else."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from papertrader.agent.schema import Decision
from papertrader.data.pit import PointInTimeView


@dataclass(frozen=True)
class PortfolioState:
    weights: dict[str, float]
    return_since_start: float
    cash_weight: float
    buying_halted: bool


class Strategy(Protocol):
    name: str

    def decide(self, view: PointInTimeView, state: PortfolioState) -> list[Decision]: ...
