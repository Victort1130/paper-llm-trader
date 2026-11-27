"""Full offline experiment: fake LLM and MA control on synthetic prices."""

import pandas as pd

from papertrader.backtest.factory import run_many
from papertrader.evaluation.consistency import decision_agreement
from papertrader.evaluation.report import benchmarks, evaluate_experiment
from papertrader.reporting.logger import read_jsonl
from papertrader.reporting.weekly import build_weekly_reports


def test_experiment_runs_and_respects_rules(cfg, market, tmp_path):
    exp = tmp_path / "exp"
    fake = run_many(cfg, market, "fake_llm", 2, exp, fake_noise=0.1)
    ma = run_many(cfg, market, "ma_cross", 1, exp)
    dirs = {"fake_llm": [r.run_dir for r in fake], "ma_cross": [r.run_dir for r in ma]}

    for r in fake + ma:
        eq = r.equity
        assert (eq["cash"] >= -1e-6).all()
        for o in read_jsonl(r.run_dir / "orders.jsonl"):
            if o.get("status") in ("approved", "trimmed") and o["side"] == "BUY":
                assert o["approved_shares"] > 0
        orders = pd.DataFrame(read_jsonl(r.run_dir / "orders.jsonl"))
        if "status" in orders:
            per_day = orders[orders["status"].isin(["approved", "trimmed"])].groupby("date").size()
            assert per_day.max() <= cfg.risk.max_trades_per_day

    calls = read_jsonl(fake[0].run_dir / "llm_calls.jsonl")
    assert calls and {"system", "user", "raw_output", "parsed", "prompt_sha256", "model_served"} <= set(calls[0])

    summary = evaluate_experiment(cfg, market, exp, dirs)
    assert set(summary["benchmarks"]) == {"BENCH buy-and-hold", "Equal-weight buy-and-hold (same 4)"}
    assert (exp / "summary.md").read_text().count("|") > 20
    agree = summary["strategies"]["fake_llm"]["agreement"]
    assert 0 < agree["all_agree"] < 1  # noise makes runs differ

    bench_eq = {k: v[0] for k, v in benchmarks(cfg, market).items()}
    paths = build_weekly_reports(fake[0].run_dir, market, bench_eq)
    assert paths and "Weekly review" in paths[0].read_text()


def test_identical_runs_agree_fully(cfg, market, tmp_path):
    runs = run_many(cfg, market, "fake_llm", 2, tmp_path / "exp")
    assert decision_agreement([r.run_dir for r in runs])["all_agree"] == 1.0
