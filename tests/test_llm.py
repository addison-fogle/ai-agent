from types import SimpleNamespace

import pytest

from llm import LLMError, OpenAILLMClient


def test_stateless_api_requests(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="completed", output_text="hello")

    client = OpenAILLMClient("test-model")
    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    messages = [{"role": "user", "content": "Hello"}]
    assert client.generate(messages) == "hello"
    assert client.generate(messages) == "hello"
    assert calls == [{"model": "test-model", "input": messages, "store": False}] * 2


def test_missing_key_is_actionable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="Set OPENAI_API_KEY"):
        OpenAILLMClient("test").generate([])


def test_api_error_does_not_expose_remote_body(monkeypatch):
    from openai import OpenAIError
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")

    def fail(**kwargs):
        raise OpenAIError("sensitive-remote-body")

    client = OpenAILLMClient("test")
    client._client = SimpleNamespace(responses=SimpleNamespace(create=fail))
    with pytest.raises(LLMError) as result:
        client.generate([])
    assert "sensitive-remote-body" not in str(result.value)

