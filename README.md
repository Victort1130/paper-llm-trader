# paper-llm-trader

A small, testable research harness that asks one question:

> Given the same daily prices and indicators, does an LLM make better trading
> decisions than a simple rule, after costs, and does it make the same
> decisions when run again?

**Paper money only.** There is no broker, exchange or account integration in
this repository, and a test fails if one is ever added. Every "order" is filled
by an in-memory simulator.

The architecture is loosely inspired by
[HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader), rewritten from scratch
as a much smaller codebase that focuses on backtest hygiene.

## What is compared

| Name | What it is | Why it is there |
|---|---|---|
| `llm` | Claude reads rebased prices plus a few indicators and returns BUY / SELL / HOLD, a target weight and a reason for each stock | The thing being tested |
| `ma_cross` | 20/50-day moving-average crossover on the same stocks, same rules and costs | Shows whether the LLM beats a rule anyone could write |
| Equal-weight buy-and-hold | The same 10 stocks, bought once | Shares the survivorship bias of the stock list, so it isolates the decisions from the stock picking |
| QQQ buy-and-hold | The index ETF, bought once | The market baseline |

The default stock list is today's ten largest QQQ holdings. Backtesting today's
winners over the past flatters every strategy on that list, which is why the
equal-weight benchmark matters more than QQQ.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                                   # no network; LLM client tests need the [llm] extra

# Offline demo: synthetic prices, a rule-based fake LLM, 5 runs each
python scripts/run_experiment.py --synthetic --strategies fake_llm ma_cross --runs 5 --fake-noise 0.05
```

Real data and the real LLM:

```bash
pip install -e ".[data,llm]"
python scripts/fetch_data.py                # Yahoo Finance daily bars -> data/raw + manifest.json
export ANTHROPIC_API_KEY=...                # your own key; calls are billed to it

# Try a short period first
python scripts/run_experiment.py --strategies llm ma_cross --runs 1 --start 2025-10-01 --end 2025-12-31 --yes
# Full experiment: 2024-2025, 5 LLM runs
python scripts/run_experiment.py --strategies llm ma_cross --runs 5 --yes
```

The script prints how many API calls it will make and stops unless `--yes` is
given. One call is made per trading day per run (about 500 per run for two
years). The prompt is about 1,300 input tokens.

## How a day works

```
day t                                             day t+1
───────────────────────────────────────────────── ─────────────────────────
fill orders from t-1 at today's open  ┐           fill today's orders at open
mark portfolio to close, update peak  │             (+ slippage, + commission)
PointInTimeView(as_of = t)            │
  └ only rows with date <= t exist    │
indicators from that window only      │
strategy -> decisions                 │
decisions -> share orders             │
risk engine trims / rejects (code) ───┘──────────►
```

## Look-ahead protection

1. **Data.** A strategy only ever receives a `PointInTimeView`, which is built
   by slicing every series to `date <= as_of`. Later rows do not exist inside
   the object. Asking for a later date raises `LookaheadError`.
2. **Timing.** A decision made from the close of day t fills at the open of
   the next trading day, never at the price it was made from.
3. **Memorisation.** An LLM may remember what these stocks did in 2024-2025.
   By default (`llm.anonymize: true`) the prompt contains no tickers
   (`Stock_A` ...), no dates, and prices rebased to 100, so the model cannot
   easily look up its own memory. This reduces the problem; it cannot remove
   it. A clean test needs a period after the model's training cutoff.

`tests/test_no_lookahead.py` proves points 1 and 2:

- views never contain rows after `as_of`, and asking for them raises
- overwriting every price after day t with random numbers leaves day t's
  indicators, prompt text and decisions byte-for-byte identical
- data truncated at t gives the same prompt and decisions as full data viewed at t
- a decision on day t fills at day t+1's open, with slippage applied
- across a whole backtest, a spy strategy never sees a date after its own day
- the anonymised prompt contains no ticker and no year

## Decision layer

- Model `claude-opus-5-5`, effort `medium`, `max_tokens` 16000, JSON output
  enforced with `output_config.format` and validated again with pydantic.
- **Temperature is not set.** Current Claude models reject sampling
  parameters with HTTP 400 and the current SDK has dropped the argument, so a
  fixed temperature is not possible. What is held fixed instead: the model id,
  effort, the prompt template (its SHA-256 is written on every log line) and
  the output schema. The 5-run consistency check measures what variation is left.
- Server-side refusal fallback is on (`fallbacks: default`). If it triggers, a
  different model answers; every call logs `model_served` and
  `fallback_used`, and the summary counts them. Set `llm.fallbacks: null` to
  turn it off.
- Every call writes one line to `llm_calls.jsonl`: full system and user
  prompt, parameters, raw output, parsed decisions, problems found, token
  usage, latency and request id.
- Unparseable output, a refusal or an API error means HOLD for every stock.
  The agent never trades on a failure.

## Risk rules (plain code, `src/papertrader/risk/`)

Applied in this order to every proposed order, each recorded in `orders.jsonl`
as approved, trimmed or rejected with the reason:

1. **Drawdown kill switch.** Once equity is 15% below its peak, no new buying
   for the rest of the run. Selling is still allowed.
2. **Daily trade limit.** At most 5 orders per day; sells are reviewed first.
3. **No short selling.** A sell cannot exceed the position held.
4. **20% position cap.** A buy is trimmed so the post-trade position is at most
   20% of equity at the decision price.
5. **No leverage.** Buys only use cash on hand, minus buys already approved
   that day, with costs included.

Known simplifications: cash from same-day sells is not used for buys (a
rebalance takes two days); a position that grows above 20% by price moves is
not trimmed; orders smaller than 1% of equity are skipped to avoid paying
commission for tiny top-ups; only whole shares.

## Costs

Commission 0.1% of notional (optional minimum), slippage 5 bps against you on
both sides. Both are config values in `configs/default.yaml`. Benchmarks pay
the same costs on their one purchase.

## Evaluation

`runs/<experiment>/summary.md` reports, for each strategy and benchmark: total
return, CAGR, annualized volatility, maximum drawdown, Sharpe ratio, win rate
of closed trades (FIFO, net of costs), annual turnover and total costs paid.

It also reports the **edge over each benchmark**: the daily differential
return D = strategy − benchmark, its annualized Sharpe and its t-statistic
(t = per-day Sharpe of D × √N). With two years of data, |t| below about 2
means the difference is not distinguishable from luck.

For strategies run several times it reports mean ± standard deviation of
every metric and **decision agreement**: the share of stock-days on which all
runs chose the same action, and the average pairwise agreement.

## Weekly review

`runs/<experiment>/<run>/weekly/YYYY-Www.md`: the week's return against both
benchmarks, every BUY and SELL with its stated reason and the stock's return
over the next 5 days (hindsight, clearly labelled, never shown to the
strategy), fills, orders blocked by the risk rules, and the best and worst
calls of the week.

See `docs/examples/` for a summary and a weekly review produced from
**synthetic** prices and the fake LLM. Those numbers mean nothing about real
markets; they only show the format.

## Layout

```
configs/default.yaml           every experiment setting
configs/prompts/decision_v1.txt  fixed prompt template
scripts/fetch_data.py          the only code that downloads data
scripts/run_experiment.py      runs strategies, writes summary and weekly reviews
scripts/make_weekly_report.py  rebuilds weekly reviews for one run
src/papertrader/
  data/        pit.py (PointInTimeView, MarketData), indicators.py, sources.py, store.py, loading.py
  agent/       prompt.py, llm_client.py (Claude, FakeLLM), schema.py
  strategies/  llm.py, ma_cross.py
  risk/        rules.py, engine.py
  execution/   costs.py, portfolio.py, simulator.py
  backtest/    loop.py, factory.py
  evaluation/  metrics.py, benchmark.py, consistency.py, report.py
  reporting/   logger.py, weekly.py, templates/weekly.md.j2
tests/         no-lookahead proofs, rules, costs, metrics, parsing, LLM request shape, end to end
docs/DESIGN.md the original design notes (in Chinese)
```

## Data source

Daily bars from Yahoo Finance (https://finance.yahoo.com) via the unofficial
`yfinance` package, adjusted for splits and dividends. `data/raw/manifest.json`
records the download time and a SHA-256 hash of every file, and a warning is
printed if a file changes afterwards. Yahoo data is for personal research;
check its terms before publishing results built on it.
