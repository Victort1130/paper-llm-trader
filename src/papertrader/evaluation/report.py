"""Experiment summary: strategies vs QQQ and equal-weight buy-and-hold."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from papertrader.config import Config
from papertrader.data.pit import MarketData
from papertrader.evaluation import benchmark, consistency, metrics
from papertrader.execution.costs import CostModel
from papertrader.reporting.logger import read_jsonl

KEYS = ["total_return", "cagr", "ann_volatility", "max_drawdown", "sharpe", "win_rate",
        "annual_turnover", "total_costs"]


def _fmt(v, pct=True):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{v:.2%}" if pct else f"{v:,.2f}"


def benchmarks(cfg: Config, market: MarketData) -> dict[str, tuple[pd.Series, float]]:
    days = market.trading_days(cfg.data.benchmark, cfg.run.start, cfg.run.end)
    costs = CostModel.from_cfg(cfg.costs)
    cash = cfg.run.initial_cash
    return {
        f"{cfg.data.benchmark} buy-and-hold": benchmark.qqq_buy_and_hold(
            market, cfg.data.benchmark, days, cash, costs),
        f"Equal-weight buy-and-hold (same {len(cfg.data.tickers)})": benchmark.equal_weight_buy_and_hold(
            market, cfg.data.tickers, days, cash, costs),
    }


def run_metrics(run_dir: Path, rf: float, benches: dict[str, pd.Series]) -> dict:
    eq = pd.read_csv(run_dir / "equity.csv", index_col="date", parse_dates=True)["equity"]
    fills = read_jsonl(run_dir / "fills.jsonl")
    out = {**metrics.summarize(eq, rf), **metrics.trade_stats(fills, eq)}
    out["differential"] = {name: metrics.differential(eq, b) for name, b in benches.items()}
    return out


def evaluate_experiment(cfg: Config, market: MarketData, exp_dir: Path, run_dirs: dict[str, list[Path]]) -> dict:
    rf = cfg.evaluation.risk_free_rate
    benches = benchmarks(cfg, market)
    bench_eq = {k: v[0] for k, v in benches.items()}
    summary: dict = {"benchmarks": {}, "strategies": {}}
    for name, (eq, paid) in benches.items():
        summary["benchmarks"][name] = {**metrics.summarize(eq, rf), "total_costs": paid}

    for kind, dirs in run_dirs.items():
        per_run = [run_metrics(Path(d), rf, bench_eq) for d in dirs]
        entry = {"runs": per_run, "spread": consistency.metric_spread(per_run, KEYS)}
        if len(dirs) > 1:
            entry["agreement"] = consistency.decision_agreement([Path(d) for d in dirs])
        stats = consistency.llm_call_stats([Path(d) for d in dirs])
        if stats:
            entry["llm_calls"] = stats
        summary["strategies"][kind] = entry

    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (exp_dir / "summary.md").write_text(render_markdown(cfg, summary), encoding="utf-8")
    return summary


def render_markdown(cfg: Config, s: dict) -> str:
    lines = [
        "# Experiment summary",
        "",
        f"Period {cfg.run.start} to {cfg.run.end}. Initial cash {cfg.run.initial_cash:,.0f}. "
        f"Costs: commission {cfg.costs.commission_rate:.2%}, slippage {cfg.costs.slippage_bps} bps.",
        "",
        "Paper trading only. The stock list is today's large QQQ holdings, so every result "
        "here carries survivorship bias; compare against the equal-weight benchmark, which "
        "shares it.",
        "",
        "| Strategy | Total return | CAGR | Volatility | Max drawdown | Sharpe | Win rate | Annual turnover | Costs paid |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in s["benchmarks"].items():
        lines.append(
            f"| {name} | {_fmt(m['total_return'])} | {_fmt(m['cagr'])} | {_fmt(m['ann_volatility'])} | "
            f"{_fmt(m['max_drawdown'])} | {_fmt(m['sharpe'], False)} | n/a | n/a | {_fmt(m['total_costs'], False)} |"
        )
    for kind, e in s["strategies"].items():
        sp = e["spread"]
        n = len(e["runs"])

        def cell(k, pct=True):
            if k not in sp:
                return "n/a"
            v = sp[k]
            if n == 1:
                return _fmt(v["mean"], pct)
            return f"{_fmt(v['mean'], pct)} ± {_fmt(v['std'], pct)}"

        label = f"{kind} (mean ± sd of {n} runs)" if n > 1 else kind
        lines.append(
            f"| {label} | {cell('total_return')} | {cell('cagr')} | {cell('ann_volatility')} | "
            f"{cell('max_drawdown')} | {cell('sharpe', False)} | {cell('win_rate')} | "
            f"{cell('annual_turnover')} | {cell('total_costs', False)} |"
        )

    lines += ["", "## Edge over each benchmark", "",
              "D = strategy daily return minus benchmark daily return. t = per-day Sharpe of D × √N. "
              "|t| < 2 means the difference is not distinguishable from luck.", "",
              "| Strategy | Run | vs | Annualized Sharpe of D | t-stat |", "|---|---|---|---|---|"]
    for kind, e in s["strategies"].items():
        for i, r in enumerate(e["runs"], 1):
            for bname, d in r["differential"].items():
                lines.append(f"| {kind} | {i} | {bname} | {_fmt(d['sharpe_annual'], False)} | {_fmt(d['t_stat'], False)} |")

    lines += ["", "## Consistency across runs", ""]
    for kind, e in s["strategies"].items():
        a = e.get("agreement")
        if a:
            lines.append(
                f"- **{kind}**: {a['n_runs']} runs. All runs chose the same action on "
                f"{_fmt(a['all_agree'])} of stock-days; average pairwise agreement {_fmt(a['pairwise'])}."
            )
        c = e.get("llm_calls")
        if c:
            lines.append(
                f"  LLM calls {c['n_calls']}, errors {c['n_errors']}, outputs with problems "
                f"{c['n_with_problems']}, answered by fallback model {c['n_fallback_served']}, "
                f"models {', '.join(c['models_served'])}, tokens in/out {c['input_tokens']:,}/{c['output_tokens']:,}."
            )
    return "\n".join(lines) + "\n"
