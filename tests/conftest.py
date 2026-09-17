import json
from copy import deepcopy

import pytest


def extraction(**changes):
    payload = {
        "memories": [], "new_goals": [], "completed_goals": [],
        "self_summary": None, "recent_beliefs": None,
    }
    payload.update(changes)
    return json.dumps(payload)


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, messages):
        self.calls.append(deepcopy(messages))
        result = next(self.outputs)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def make_extraction():
    return extraction

