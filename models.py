"""Plain data records and strict validation at the untrusted JSON boundary."""

import json
from dataclasses import dataclass, field
from typing import Any

MEMORY_TYPES = {"user_fact", "episodic", "self", "goal", "decision"}


def exact_keys(value: Any, keys: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Unexpected or missing JSON fields")
    return value


def string(value: Any, *, limit: int = 2000, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError("Expected a bounded string")
    return value.strip()


def strings(value: Any, *, limit: int = 100) -> list[str]:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError("Expected a bounded string list")
    result: dict[str, str] = {}
    for item in value:
        item = string(item, limit=500)
        result.setdefault(item.casefold(), item)
    return list(result.values())


@dataclass(frozen=True)
class Identity:
    name: str = "Elias"
    personality: list[str] = field(default_factory=lambda: ["analytical", "curious", "calm"])
    core_goals: list[str] = field(default_factory=lambda: [
        "help the user solve problems", "maintain continuity across conversations"
    ])

    @classmethod
    def from_dict(cls, data: Any) -> "Identity":
        data = exact_keys(data, {"name", "personality", "core_goals"})
        return cls(string(data["name"], limit=100), strings(data["personality"]), strings(data["core_goals"]))


@dataclass(frozen=True)
class SelfState:
    self_summary: str = ""
    active_goals: list[str] = field(default_factory=list)
    completed_goals: list[str] = field(default_factory=list)
    recent_beliefs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "SelfState":
        data = exact_keys(data, {"self_summary", "active_goals", "completed_goals", "recent_beliefs"})
        state = cls(
            string(data["self_summary"], limit=4000, empty=True),
            strings(data["active_goals"]), strings(data["completed_goals"]),
            strings(data["recent_beliefs"], limit=20),
        )
        if {g.casefold() for g in state.active_goals} & {g.casefold() for g in state.completed_goals}:
            raise ValueError("A goal cannot be both active and completed")
        return state


@dataclass(frozen=True)
class NewMemory:
    memory_type: str
    content: str
    importance: int

    @classmethod
    def from_dict(cls, data: Any) -> "NewMemory":
        data = exact_keys(data, {"memory_type", "content", "importance"})
        kind = string(data["memory_type"], limit=20)
        if kind not in MEMORY_TYPES:
            raise ValueError("Unknown memory type")
        importance = data["importance"]
        if type(importance) is not int or not 1 <= importance <= 10:
            raise ValueError("Importance must be an integer from 1 to 10")
        return cls(kind, string(data["content"]), importance)


@dataclass(frozen=True)
class Memory:
    id: int
    created_at: str
    memory_type: str
    content: str
    importance: int
    source_turn_id: str


@dataclass(frozen=True)
class Extraction:
    memories: list[NewMemory]
    new_goals: list[str]
    completed_goals: list[str]
    self_summary: str | None
    recent_beliefs: list[str] | None

    @classmethod
    def parse(cls, raw: str) -> "Extraction":
        if len(raw) > 100_000:
            raise ValueError("Extraction is too large")
        data = exact_keys(json.loads(raw), {
            "memories", "new_goals", "completed_goals", "self_summary", "recent_beliefs"
        })
        if not isinstance(data["memories"], list) or len(data["memories"]) > 20:
            raise ValueError("Expected at most 20 memories")
        return cls(
            [NewMemory.from_dict(item) for item in data["memories"]],
            strings(data["new_goals"]), strings(data["completed_goals"]),
            None if data["self_summary"] is None else string(data["self_summary"], limit=4000, empty=True),
            None if data["recent_beliefs"] is None else strings(data["recent_beliefs"], limit=20),
        )

    def apply(self, old: SelfState) -> SelfState:
        known = {g.casefold() for g in old.active_goals + old.completed_goals + self.new_goals}
        if any(g.casefold() not in known for g in self.completed_goals):
            raise ValueError("Cannot complete an unknown goal")
        completed = strings(old.completed_goals + self.completed_goals)
        closed = {g.casefold() for g in completed}
        active = strings([g for g in old.active_goals + self.new_goals if g.casefold() not in closed])
        return SelfState(
            old.self_summary if self.self_summary is None else self.self_summary,
            active, completed,
            old.recent_beliefs if self.recent_beliefs is None else self.recent_beliefs,
        )

