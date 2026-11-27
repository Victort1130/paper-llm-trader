"""Rebuild weekly reviews for one finished run.

    python scripts/make_weekly_report.py runs/<experiment>/llm_run1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from papertrader.config import load_config
from papertrader.data.loading import load_market
from papertrader.evaluation.report import benchmarks
from papertrader.reporting.weekly import build_weekly_reports


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    market = load_market(cfg)
    bench_eq = {name: eq for name, (eq, _) in benchmarks(cfg, market).items()}
    paths = build_weekly_reports(run_dir, market, bench_eq, cfg.evaluation.forward_days)
    print(f"wrote {len(paths)} weekly reports to {run_dir / 'weekly'}")


if __name__ == "__main__":
    main()
