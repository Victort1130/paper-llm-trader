# 模擬資金 LLM 交易代理：架構設計 v0.1

> 狀態：v0.1 設計，已於 2026-11-27 確認並實作。實作時的調整寫在 README.md（例如溫度無法設定、多了均線規則與等權重對照組、`strategies/` 資料夾、腳本合併為 `run_experiment.py`）。

## 0. 安全邊界（寫死在程式結構裡）

- 專案中**不存在任何券商、交易所的 API 模組**。「成交」只發生在 `execution/simulator.py` 的記憶體內帳本。
- 對外網路只有兩種：下載歷史資料（`scripts/fetch_data.py`，只在回測前手動執行一次）、呼叫 LLM API。回測迴圈本身只讀本地快取。
- 測試 `tests/test_no_live_trading.py` 會掃描相依套件與 import，若出現券商 SDK（如 alpaca、ib_insync、ccxt 等）直接失敗。
- 不做任何帳號註冊或登入動作。

## 1. 整體流程（每個交易日 t）

```
本地快取資料 ──► PointInTimeView(as_of=t 收盤)
                     │  只回傳 date ≤ t 的資料
                     ▼
               指標計算（只用該視窗）
                     ▼
         Prompt 組裝（固定模板 + 版本 hash）
                     ▼
         LLM（固定溫度）──► 解析成 Decision{動作, 比例, 理由}
                     │          完整輸入／輸出寫入 JSONL
                     ▼
        風控引擎（純程式規則，可拒絕／裁減訂單）
                     ▼
     成交模擬：t+1 開盤價 ± 滑價，扣手續費
                     ▼
            帳本更新、每日淨值、日誌
```

決策用 t 日收盤後的資訊，成交用 t+1 開盤價。這是防止前視偏差的第二道防線（第一道是資料視窗）。

## 2. 各層設計

### 2.1 資料層 `data/`
- **資料來源【假設】**：Yahoo Finance 日線（透過 `yfinance`），使用還原權值價格。備案為 Stooq CSV。下載時寫一份 `manifest.json`，記錄來源、下載時間、檔案 hash，讓每次回測可追溯。
- **標的池【假設】**：QQQ 前 10 大成分股 + QQQ 本身（例如 AAPL、MSFT、NVDA、AMZN、GOOGL、META、AVGO、TSLA、COST、NFLX）。注意：用「今天」的成分股回測過去會有存活者偏差，週報與評估會註明。
- **回測期間【假設】**：2024-01-01 至 2025-12-31。
- `PointInTimeView`：唯一允許策略讀資料的介面。`get_bars(ticker, as_of, lookback)` 只回傳 `date ≤ as_of` 的列；嘗試讀取更晚日期會丟出 `LookaheadError`。
- 指標【假設】：5／20 日報酬、20／50 日均線、14 日 RSI、20 日波動率。全部只在視窗內計算，不呼叫會用到全序列的函式。

**前視偏差測試（`tests/test_no_lookahead.py`）**
1. 視窗測試：任意 t，`PointInTimeView` 回傳的最後一列日期必須 ≤ t。
2. 竄改未來測試：把 t 之後的價格全部改成亂數或 NaN，t 當天的指標、prompt 文字、決策都必須和原本完全一樣（用假 LLM 驗證）。
3. 截斷等價測試：用「資料截斷到 t」跑出來的指標，必須等於用「完整資料 + as_of=t」跑出來的。
4. 成交時序測試：t 日的決策只能用 t+1 開盤價成交。

**一個 LLM 特有的風險**：LLM 的訓練資料可能已經「記得」歷史行情，這是程式無法擋的前視偏差。緩解選項：(a) 回測期間選在模型知識截止日之後；(b) prompt 中把股票代號匿名化（Stock_A）並把日期改成相對日數。【假設】預設開啟 (b)，並在報告中註明這個限制。

### 2.2 決策層 `agent/`
- **LLM 供應商【假設】**：Anthropic Claude API，模型名稱寫在設定檔。另有 `FakeLLM`（規則式、可重現）供測試與離線開發，測試完全不打 API。
- 固定溫度【假設】：`temperature = 0`。prompt 模板放在 `configs/prompts/decision_v1.txt`，執行時計算模板 hash 寫入日誌，模板一改版號就跟著變。
- 輸出格式：要求 JSON，`{ "ticker", "action": "BUY|SELL|HOLD", "target_weight": 0~0.2, "reason" }`，用 pydantic 驗證。解析失敗一律視為 HOLD 並記錄原文。
- 日誌：每次呼叫寫一行 JSONL，包含完整 prompt、模型名稱、參數、原始回應、解析結果、token 用量、耗時。

### 2.3 風控層 `risk/`（純程式，不經 LLM）
每條規則是獨立的純函式，輸入（訂單、帳本、日期）、輸出（通過／裁減／拒絕 + 原因）：
- `MaxPositionWeight`：單一標的 ≤ 20% 淨值，超過就裁減到上限。
- `NoLeverage`：買進金額不得超過現金，不允許放空。
- `MaxTradesPerDay`【假設】：每日最多 5 筆。
- `DrawdownKillSwitch`【假設 X = 15%】：淨值從高點回撤達 X% 後停止所有新買進，只允許賣出或持有，直到回測結束。
- 所有被拒絕或裁減的訂單都進日誌，週報會統計。

### 2.4 成交模擬 `execution/`
- 成交價 = t+1 開盤價 × (1 ± 滑價)，買進加、賣出減。
- 滑價【假設】：5 bps。手續費【假設】：成交金額 0.1%，可改成每股計價。
- 股數【假設】：只允許整數股。
- 初始資金【假設】：100,000 美元。

### 2.5 評估 `evaluation/`
- 和「第一天全部買進 QQQ 並持有」比較：總報酬、年化報酬、年化波動、最大回撤、夏普值（無風險利率【假設】設為 0，可設定）、勝率（以平倉交易計算）。
- **一致性**：同一設定跑 5 次。報告每項指標的平均、標準差、最小、最大，以及「每日決策一致率」（5 次中同一天做出同樣動作的比例）。即使溫度為 0，API 回應仍可能有細微差異，這正是要量的東西。

### 2.6 日誌與週報 `reporting/`
- 每次回測產生 `runs/<run_id>/`：`config.yaml` 快照、`decisions.jsonl`、`orders.jsonl`、`fills.jsonl`、`equity.csv`、`metrics.json`。
- 週報（Markdown）：本週淨值與 QQQ 對比、每筆交易與 LLM 理由、事後 5 日表現（理由是否應驗）、被風控擋下的訂單、最好與最差的決策各 3 筆。週報只用事後資料做「檢討」，不回饋給決策，避免汙染。

## 3. 檔案結構

```
paper-llm-trader/
├── README.md
├── pyproject.toml                 # Python 3.11，pandas、pydantic、pyyaml、jinja2、pytest
├── .gitignore                     # 忽略 data/、runs/、.env
├── configs/
│   ├── default.yaml               # 標的、期間、風控、成本、LLM 參數
│   └── prompts/
│       └── decision_v1.txt        # 固定 prompt 模板
├── data/                          # 不進版控
│   ├── raw/                       # 下載的原始 CSV
│   └── manifest.json              # 來源、下載時間、hash
├── src/papertrader/
│   ├── config.py                  # 讀取並驗證設定
│   ├── data/
│   │   ├── sources.py             # YahooSource、CSVSource
│   │   ├── store.py               # 本地快取讀寫與 manifest
│   │   ├── pit.py                 # PointInTimeView、LookaheadError
│   │   └── indicators.py          # 只吃視窗資料的指標
│   ├── agent/
│   │   ├── llm_client.py          # LLMClient 介面、AnthropicClient、FakeLLM
│   │   ├── prompt.py              # 組裝 prompt、匿名化、模板 hash
│   │   ├── schema.py              # Decision 資料結構與解析
│   │   └── decider.py             # 串起 prompt → LLM → Decision
│   ├── risk/
│   │   ├── rules.py               # 四條規則，各為純函式
│   │   └── engine.py              # 依序套用規則並記錄原因
│   ├── execution/
│   │   ├── costs.py               # 手續費、滑價模型
│   │   ├── portfolio.py           # 現金、持倉、淨值
│   │   └── simulator.py           # 模擬成交（唯一的「下單」處）
│   ├── backtest/
│   │   ├── calendar.py            # 交易日序列
│   │   ├── loop.py                # 每日主迴圈
│   │   └── multi_run.py           # 同設定跑 N 次
│   ├── evaluation/
│   │   ├── metrics.py             # 報酬、波動、MDD、夏普、勝率
│   │   ├── benchmark.py           # QQQ 買進持有
│   │   └── consistency.py         # 5 次結果比較
│   └── reporting/
│       ├── logger.py              # JSONL 寫入
│       ├── weekly.py              # 週報產生
│       └── templates/weekly.md.j2
├── scripts/
│   ├── fetch_data.py              # 唯一會下載資料的地方
│   ├── run_backtest.py
│   ├── run_consistency.py         # 跑 5 次並輸出比較
│   └── make_weekly_report.py
├── runs/                          # 不進版控，每次回測一個資料夾
└── tests/
    ├── fixtures/                  # 小型合成價格資料
    ├── test_no_lookahead.py       # 四個前視偏差測試
    ├── test_indicators.py
    ├── test_risk_rules.py
    ├── test_costs_and_fills.py
    ├── test_portfolio.py
    ├── test_metrics.py            # 用已知答案的合成序列驗證
    ├── test_decision_parsing.py   # 壞 JSON → HOLD
    └── test_no_live_trading.py    # 禁止券商 SDK
```

## 4. 待確認的假設

| # | 項目 | 我的預設 |
|---|------|----------|
| 1 | 資料來源 | Yahoo Finance（yfinance），還原權值日線 |
| 2 | 標的池 | QQQ 前 10 大成分股 |
| 3 | 回測期間 | 2024-01-01 ～ 2025-12-31 |
| 4 | LLM 供應商與模型 | Anthropic Claude（需要你自己的 API key，放 `.env`） |
| 5 | 溫度 | 0 |
| 6 | 總回撤停止門檻 X | 15% |
| 7 | 每日交易上限 | 5 筆 |
| 8 | 手續費／滑價 | 0.1%／5 bps |
| 9 | 初始資金、整數股 | 100,000 美元，整數股 |
| 10 | 股票代號匿名化 | 預設開啟 |
| 11 | 程式碼放哪裡 | 尚未決定：GitHub repo 或只放在專案檔案夾 |
