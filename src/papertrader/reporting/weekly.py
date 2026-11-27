"""Weekly decision review.

Looks back at each week's decisions with hindsight (the next N days' returns).
This is for human review only; nothing here is fed back to the strategy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from papertrader.data.pit import MarketData
from papertrader.reporting.logger import read_jsonl

TEMPLATES = Path(__file__).parent / "templates"


def _forward_return(market: MarketData, ticker: str, date: pd.Timestamp, days: int) -> float | None:
    s = market.bars[ticker]["close"].dropna()
    after = s.loc[date:]
    if len(after) <= days:
        return None
    return float(after.iloc[days] / after.iloc[0] - 1.0)


def _score(action: str, fwd: float | None) -> float | None:
    """Positive when hindsight agrees with the action."""
    if fwd is None:
        return None
    return fwd if action == "BUY" else -fwd if action == "SELL" else None


def build_weekly_reports(
    run_dir: str | Path,
    market: MarketData,
    benches: dict[str, pd.Series],
    forward_days: int = 5,
) -> list[Path]:
    run_dir = Path(run_dir)
    eq = pd.read_csv(run_dir / "equity.csv", index_col="date", parse_dates=True)["equity"]
    decisions = pd.DataFrame(read_jsonl(run_dir / "decisions.jsonl"))
    orders = pd.DataFrame(read_jsonl(run_dir / "orders.jsonl"))
    fills = pd.DataFrame(read_jsonl(run_dir / "fills.jsonl"))
    out_dir = run_dir / "weekly"
    out_dir.mkdir(exist_ok=True)
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    tpl = env.get_template("weekly.md.j2")

    week_of = eq.index.to_series().dt.strftime("%G-W%V")
    paths = []
    for week, dates in eq.index.to_series().groupby(week_of):
        first, last = dates.iloc[0], dates.iloc[-1]
        prev = eq.loc[:first].iloc[-2] if len(eq.loc[:first]) > 1 else eq.loc[first]

        def wk_ret(series: pd.Series) -> float:
            p = series.loc[:first]
            start = p.iloc[-2] if len(p) > 1 else p.iloc[-1]
            return float(series.loc[last] / start - 1.0)

        wd = decisions[decisions["date"].between(str(first.date()), str(last.date()))] if len(decisions) else decisions
        actions = []
        for _, r in wd[wd["action"] != "HOLD"].iterrows() if len(wd) else []:
            fwd = _forward_return(market, r["ticker"], pd.Timestamp(r["date"]), forward_days)
            actions.append({**r.to_dict(), "fwd": fwd, "score": _score(r["action"], fwd)})
        scored = [a for a in actions if a["score"] is not None]
        scored.sort(key=lambda a: a["score"], reverse=True)

        wo = orders[orders["date"].between(str(first.date()), str(last.date()))] if len(orders) else orders
        blocked = wo[wo["status"].isin(["rejected", "trimmed"])].to_dict("records") if len(wo) and "status" in wo else []
        wf = fills[fills["fill_date"].between(str(first.date()), str(last.date()))].to_dict("records") if len(fills) else []

        text = tpl.render(
            week=week,
            first=first.date(),
            last=last.date(),
            equity_start=float(prev),
            equity_end=float(eq.loc[last]),
            ret=float(eq.loc[last] / prev - 1.0),
            bench=[(name, wk_ret(s)) for name, s in benches.items()],
            actions=actions,
            fills=wf,
            blocked=blocked,
            best=scored[:3],
            worst=scored[::-1][:3] if len(scored) > 3 else [],
            n_hold=int((wd["action"] == "HOLD").sum()) if len(wd) else 0,
            forward_days=forward_days,
        )
        p = out_dir / f"{week}.md"
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths
