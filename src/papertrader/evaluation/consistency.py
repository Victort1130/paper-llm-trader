"""Compare N runs of the same configuration."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from papertrader.reporting.logger import read_jsonl


def decision_agreement(run_dirs: list[Path]) -> dict:
    """How often the runs chose the same action for the same stock and day.

    all_agree: share of (date, ticker) cells where every run matched.
    pairwise: average share of matching cells over all pairs of runs.
    """
    tables = []
    for d in run_dirs:
        rows = read_jsonl(Path(d) / "decisions.jsonl")
        s = pd.Series({(r["date"], r["ticker"]): r["action"] for r in rows})
        tables.append(s)
    if len(tables) < 2:
        return {"n_runs": len(tables), "all_agree": None, "pairwise": None, "n_cells": 0}
    df = pd.concat(tables, axis=1).dropna()
    all_agree = float((df.nunique(axis=1) == 1).mean()) if len(df) else None
    pair = [float((df[a] == df[b]).mean()) for a, b in combinations(df.columns, 2)]
    return {
        "n_runs": len(tables),
        "n_cells": int(len(df)),
        "all_agree": all_agree,
        "pairwise": sum(pair) / len(pair) if pair else None,
    }


def metric_spread(per_run: list[dict], keys: list[str]) -> dict:
    df = pd.DataFrame(per_run)
    out = {}
    for k in keys:
        if k not in df or df[k].isna().all():
            continue
        col = df[k].astype(float)
        out[k] = {"mean": col.mean(), "std": col.std(ddof=1), "min": col.min(), "max": col.max()}
    return out


def llm_call_stats(run_dirs: list[Path]) -> dict:
    calls = [c for d in run_dirs for c in read_jsonl(Path(d) / "llm_calls.jsonl")]
    if not calls:
        return {}
    return {
        "n_calls": len(calls),
        "n_errors": sum(1 for c in calls if c.get("error")),
        "n_with_problems": sum(1 for c in calls if c.get("problems")),
        "n_fallback_served": sum(1 for c in calls if c.get("fallback_used")),
        "models_served": sorted({c.get("model_served") for c in calls if c.get("model_served")}),
        "input_tokens": sum((c.get("usage") or {}).get("input_tokens", 0) or 0 for c in calls),
        "output_tokens": sum((c.get("usage") or {}).get("output_tokens", 0) or 0 for c in calls),
    }
