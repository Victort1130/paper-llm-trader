"""AnthropicClient request shape, checked against the installed SDK signature.
No network: the SDK method is replaced by a stub."""

import inspect
from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")

from papertrader.agent.llm_client import AnthropicClient  # noqa: E402
from papertrader.agent.schema import output_schema  # noqa: E402
from papertrader.config import LLMCfg  # noqa: E402


class _Usage(SimpleNamespace):
    def model_dump(self):
        return dict(self.__dict__)


def _client(monkeypatch, **cfg):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    c = AnthropicClient(LLMCfg(**cfg))
    calls = {}
    real_beta = c.client.beta.messages.create
    real_plain = c.client.messages.create

    def stub(real, name):
        def f(**kwargs):
            inspect.signature(real).bind(**kwargs)  # raises if an argument is not accepted
            calls[name] = kwargs
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text='{"decisions": []}')],
                model=kwargs["model"],
                stop_reason="end_turn",
                usage=_Usage(input_tokens=10, output_tokens=5, iterations=None),
                _request_id="req_1",
            )
        return f

    monkeypatch.setattr(c.client.beta.messages, "create", stub(real_beta, "beta"))
    monkeypatch.setattr(c.client.messages, "create", stub(real_plain, "plain"))
    return c, calls


def test_request_with_fallbacks_uses_beta_endpoint(monkeypatch):
    c, calls = _client(monkeypatch)
    r = c.complete("sys", "user", output_schema(0.2))
    kw = calls["beta"]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["output_config"]["effort"] == "medium"
    assert kw["fallbacks"] == "default" and kw["betas"] == [AnthropicClient.FALLBACK_BETA]
    assert "temperature" not in kw
    assert r.text == '{"decisions": []}' and r.request_id == "req_1" and not r.fallback_used


def test_request_without_fallbacks(monkeypatch):
    c, calls = _client(monkeypatch, fallbacks=None)
    c.complete("sys", "user", output_schema(0.2))
    assert "plain" in calls and "fallbacks" not in calls["plain"]


def test_temperature_is_refused_up_front(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with pytest.raises(ValueError):
        AnthropicClient(LLMCfg(temperature=0))
