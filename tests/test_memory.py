from datetime import datetime, timedelta, timezone

import pytest

from memory import MemoryStore
from models import NewMemory

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    memory = MemoryStore(tmp_path / "nested" / "memories.db")
    yield memory
    memory.close()


def test_initialization_and_insert(store):
    assert store.all() == []
    memory_id = store.add(NewMemory("user_fact", "The user is building a compiler.", 8), "turn-1", created_at=NOW)
    memory = store.all()[0]
    assert memory.id == memory_id
    assert memory.source_turn_id == "turn-1"
    assert memory.created_at == NOW.isoformat()
    assert memory.importance == 8


def test_keyword_overlap_outweighs_unrelated_importance(store):
    store.add(NewMemory("user_fact", "Enjoys baking bread.", 10), "1", created_at=NOW)
    store.add(NewMemory("episodic", "Debugged compiler lexer.", 2), "2", created_at=NOW)
    found = store.retrieve("compiler lexer", limit=1, now=NOW)
    assert found[0].content == "Debugged compiler lexer."
    assert store.retrieve("compiler", limit=0) == []


def test_importance_ordering_when_overlap_and_recency_match(store):
    store.add(NewMemory("goal", "Debug compiler.", 2), "1", created_at=NOW)
    store.add(NewMemory("decision", "Debug compiler.", 9), "2", created_at=NOW)
    assert [m.importance for m in store.retrieve("compiler", now=NOW)] == [9, 2]


def test_recency_and_deterministic_ties(store):
    store.add(NewMemory("goal", "compiler", 5), "1", created_at=NOW - timedelta(days=60))
    store.add(NewMemory("decision", "compiler", 5), "2", created_at=NOW)
    store.add(NewMemory("self", "compiler", 5), "3", created_at=NOW)
    expected = [3, 2, 1]
    assert [m.id for m in store.retrieve("compiler", now=NOW)] == expected
    assert [m.id for m in store.retrieve("compiler", now=NOW)] == expected
    assert len(store.retrieve("unmatched keyword", now=NOW)) == 3  # documented fallback


def test_duplicates_and_conservative_near_duplicates(store):
    original = NewMemory("user_fact", "The user is building a compiler in Rust.", 7)
    first = store.add(original, "1")
    assert first is not None
    assert store.add(original, "2") is None
    assert store.add(NewMemory("user_fact", "USER is building compiler  in Rust.", 9), "3") is None
    assert store.add(NewMemory("user_fact", "The user is not building a compiler in Rust.", 7), "4") is not None
    assert store.add(NewMemory("user_fact", "The user is building a compiler in Ruby.", 7), "5") is not None
    assert len(store.all()) == 3
    assert store.all()[0].source_turn_id == "1"  # duplicates cannot rewrite provenance


def test_dedup_preserves_punctuation_and_memory_type(store):
    for language in ("C", "C++", "C#"):
        store.add(NewMemory("user_fact", f"Uses {language}", 5), "1")
    store.add(NewMemory("decision", "Uses C", 5), "1")
    assert len(store.all()) == 4


@pytest.mark.parametrize("kind,importance", [("unknown", 5), ("goal", 0), ("goal", 11), ("goal", True)])
def test_invalid_memory_rejected(store, kind, importance):
    with pytest.raises(ValueError):
        store.add(NewMemory(kind, "hello", importance), "1")
    assert store.all() == []

