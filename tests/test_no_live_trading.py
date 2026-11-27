"""Guards the paper-only rule: no broker, exchange or trading SDK anywhere."""

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "alpaca", "alpaca_trade_api", "ib_insync", "ib_async", "ibapi", "ccxt", "robin_stocks",
    "tda", "schwab", "td_ameritrade", "tradier", "oandapyV20", "MetaTrader5", "binance",
    "kiteconnect", "webull", "tastytrade", "shioaji", "fubon_neo",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_trading_sdk_is_imported():
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    assert files
    for f in files:
        bad = _imports(f) & FORBIDDEN
        assert not bad, f"{f} imports {bad}"


def test_no_trading_sdk_in_dependencies():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = list(data["project"]["dependencies"])
    for extra in data["project"].get("optional-dependencies", {}).values():
        deps += extra
    names = {d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for d in deps}
    assert not names & {n.lower() for n in FORBIDDEN}


def test_only_known_network_clients():
    """Only the data download (yfinance) and the LLM client (anthropic) may reach the network."""
    network_modules = {"requests", "httpx", "urllib3", "aiohttp", "socket", "websocket", "websockets"}
    for f in (ROOT / "src").rglob("*.py"):
        assert not _imports(f) & network_modules, f"{f} opens network connections directly"
