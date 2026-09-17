"""The only API-specific module. No conversation IDs or hidden model state."""

import os
import re
from typing import Protocol


class LLMError(RuntimeError):
    """An API error safe to display without dumping request data or credentials."""


class LLMClient(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


def redact_secrets(text: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)


class OpenAILLMClient:
    def __init__(self, model: str):
        # Lazy import lets inspection commands and the offline demo run without the SDK.
        self.model = model
        self._client = None

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise LLMError("Set OPENAI_API_KEY in your environment to chat.")
        try:
            from openai import OpenAI, OpenAIError
        except ImportError:
            raise LLMError("Install dependencies with: python -m pip install -r requirements.txt") from None
        try:
            if self._client is None:
                self._client = OpenAI(timeout=60.0, max_retries=1)
            result = self._client.responses.create(
                model=self.model, input=messages, store=False,
            )
        except OpenAIError as error:
            # Exception bodies can include remote request content. Do not print them.
            raise LLMError(f"OpenAI request failed ({type(error).__name__}); check credentials, model and connection.") from None
        if result.status != "completed" or not result.output_text.strip():
            raise LLMError("OpenAI returned an incomplete or empty response; no turn was saved.")
        return result.output_text

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

