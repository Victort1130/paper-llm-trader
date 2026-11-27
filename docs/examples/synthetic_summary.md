# Experiment summary

Period 2024-01-01 to 2025-12-31. Initial cash 100,000. Costs: commission 0.10%, slippage 5.0 bps.

Paper trading only. The stock list is today's large QQQ holdings, so every result here carries survivorship bias; compare against the equal-weight benchmark, which shares it.

| Strategy | Total return | CAGR | Volatility | Max drawdown | Sharpe | Win rate | Annual turnover | Costs paid |
|---|---|---|---|---|---|---|---|---|
| QQQ buy-and-hold | 162.32% | 59.29% | 23.81% | -19.27% | 2.08 | n/a | n/a | 149.74 |
| Equal-weight buy-and-hold (same 10) | 43.69% | 19.12% | 7.60% | -6.20% | 2.34 | n/a | n/a | 149.08 |
| fake_llm (mean ± sd of 5 runs) | 12.07% ± 2.83% | 5.65% ± 1.29% | 8.00% ± 0.11% | -7.43% ± 0.74% | 0.73 ± 0.15 | 30.01% ± 0.93% | 4952.95% ± 93.77% | 16,115.34 ± 392.36 |
| ma_cross | 16.53% | 7.67% | 6.15% | -5.51% | 1.23 | 45.90% | 588.03% | 1,932.28 |

## Edge over each benchmark

D = strategy daily return minus benchmark daily return. t = per-day Sharpe of D × √N. |t| < 2 means the difference is not distinguishable from luck.

| Strategy | Run | vs | Annualized Sharpe of D | t-stat |
|---|---|---|---|---|
| fake_llm | 1 | QQQ buy-and-hold | -1.79 | -2.58 |
| fake_llm | 1 | Equal-weight buy-and-hold (same 10) | -2.22 | -3.20 |
| fake_llm | 2 | QQQ buy-and-hold | -1.74 | -2.50 |
| fake_llm | 2 | Equal-weight buy-and-hold (same 10) | -1.92 | -2.76 |
| fake_llm | 3 | QQQ buy-and-hold | -1.68 | -2.42 |
| fake_llm | 3 | Equal-weight buy-and-hold (same 10) | -1.83 | -2.63 |
| fake_llm | 4 | QQQ buy-and-hold | -1.77 | -2.54 |
| fake_llm | 4 | Equal-weight buy-and-hold (same 10) | -2.14 | -3.08 |
| fake_llm | 5 | QQQ buy-and-hold | -1.67 | -2.40 |
| fake_llm | 5 | Equal-weight buy-and-hold (same 10) | -1.74 | -2.51 |
| ma_cross | 1 | QQQ buy-and-hold | -1.70 | -2.45 |
| ma_cross | 1 | Equal-weight buy-and-hold (same 10) | -2.27 | -3.27 |

## Consistency across runs

- **fake_llm**: 5 runs. All runs chose the same action on 83.91% of stock-days; average pairwise agreement 93.32%.
  LLM calls 2610, errors 0, outputs with problems 0, answered by fallback model 0, models fake-llm, tokens in/out 3,478,080/0.
