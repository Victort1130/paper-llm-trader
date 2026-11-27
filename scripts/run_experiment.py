"""Run strategies, compare with QQQ and equal-weight buy-and-hold, write
summary.md and weekly reviews.

Offline demo on synthetic prices (no data download, no API calls):
    python scripts/run_experiment.py --synthetic --strategies fake_llm ma_cross --runs 5 --fake-noise 0.05

Real experiment (needs data/raw from fetch_data.py and ANTHROPIC_API_KEY):
    python scripts/run_experiment.py --strategies llm ma_cross --runs 5 --yes
"""

from __future__ import annotations

import argparse

from papertrader.backtest.factory import new_experiment_dir, run_many
from papertrader.config import load_config
from papertrader.data.loading import load_market
from papertrader.evaluation.report import benchmarks, evaluate_experiment
from papertrader.reporting.weekly import build_weekly_reports

DETERMINISTIC = {"ma_cross"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--strategies", nargs="+", default=["llm", "ma_cross"])
    ap.add_argument("--runs", type=int, default=None, help="runs per non-deterministic strategy")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--synthetic", action="store_true", help="use generated prices (offline demo)")
    ap.add_argument("--fake-noise", type=float, default=0.0, help="random action flips for fake_llm")
    ap.add_argument("--yes", action="store_true", help="confirm paid LLM API calls")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.start:
        cfg.run.start = args.start
    if args.end:
        cfg.run.end = args.end
    if args.synthetic:
        cfg.data.source = "synthetic"
    n = args.runs or cfg.evaluation.consistency_runs

    market = load_market(cfg)
    days = market.trading_days(cfg.data.benchmark, cfg.run.start, cfg.run.end)
    if "llm" in args.strategies:
        calls = (len(days) - 1) * n
        print(f"The llm strategy will make about {calls} API calls to {cfg.llm.model} "
              f"({len(days) - 1} trading days x {n} runs), billed to your API key.")
        if not args.yes:
            raise SystemExit("Re-run with --yes to confirm, or shorten the period with --start/--end.")

    exp_dir = new_experiment_dir(cfg, "-".join(args.strategies))
    run_dirs = {}
    for kind in args.strategies:
        k = 1 if kind in DETERMINISTIC else n
        print(f"running {kind} x{k} ...")
        results = run_many(cfg, market, kind, k, exp_dir, fake_noise=args.fake_noise)
        run_dirs[kind] = [r.run_dir for r in results]

    evaluate_experiment(cfg, market, exp_dir, run_dirs)
    bench_eq = {name: eq for name, (eq, _) in benchmarks(cfg, market).items()}
    for dirs in run_dirs.values():
        for d in dirs:
            build_weekly_reports(d, market, bench_eq, cfg.evaluation.forward_days)
    print(f"done: {exp_dir}/summary.md")


if __name__ == "__main__":
    main()
