"""Download daily bars once into data/raw/ and write data/raw/manifest.json.

This is the only script that downloads anything. Backtests read the files.

    pip install -e ".[data]"
    python scripts/fetch_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from papertrader.config import load_config
from papertrader.data.loading import history_start
from papertrader.data.sources import YahooSource
from papertrader.data.store import save_bars, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    src = YahooSource()
    start = history_start(cfg)
    end = str((pd.Timestamp(cfg.run.end) + pd.Timedelta(days=10)).date())  # room for forward reviews
    names = list(dict.fromkeys(cfg.data.tickers + [cfg.data.benchmark]))
    for t in names:
        df = src.fetch(t, start, end)
        if df.empty:
            raise SystemExit(f"no data returned for {t}")
        save_bars(cfg.data.dir, t, df)
        print(f"{t}: {len(df)} rows {df.index[0].date()} .. {df.index[-1].date()}")
    m = write_manifest(
        cfg.data.dir,
        src.name,
        {"tickers": names, "start": start, "end": end, "adjusted": True,
         "url": "https://finance.yahoo.com"},
    )
    print(f"manifest: {m}")


if __name__ == "__main__":
    main()
