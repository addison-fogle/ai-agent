import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

import memory
from agent import Agent
from config import Config
from memory import MemoryStore, load_state
from models import NewMemory, SelfState


def test_completely_new_agent_retrieves_stored_fact(tmp_path, fake_llm, make_extraction):
    first = Agent(Config(tmp_path), fake_llm("I can help.", make_extraction(memories=[{
        "memory_type": "user_fact", "content": "The user is building a compiler in Rust.", "importance": 8,
    }])))
    first.respond("I'm building a compiler in Rust.")
    first.close()
    del first

    second = Agent(Config(tmp_path), fake_llm())
    try:
        assert second.recent == []
        assert second.state.self_summary == ""
        assert second.memory.retrieve("compiler")[0].content == "The user is building a compiler in Rust."
    finally:
        second.close()


def test_process_independent_persistence(tmp_path):
    script = Path(__file__).resolve().parents[1] / "demo.py"
    write = subprocess.run([sys.executable, str(script), "--worker", "write", "--storage", str(tmp_path)],
                           check=True, capture_output=True, text=True, timeout=20)
    read = subprocess.run([sys.executable, str(script), "--worker", "read", "--storage", str(tmp_path)],
                          check=True, capture_output=True, text=True, timeout=20)
    assert "Saved 1 memory" in write.stdout
    assert "The user is building a compiler in Rust." in read.stdout
    assert "Recent turns at startup: 0" in read.stdout


def test_interrupted_json_export_recovers_on_new_agent(tmp_path, fake_llm, monkeypatch):
    store = MemoryStore(tmp_path / "memories.db")
    state_path = tmp_path / "self_state.json"
    memory.atomic_json(state_path, asdict(SelfState()))
    next_state = SelfState("Discussed a compiler", ["debug"], [], [])

    def fail_export(*args):
        raise OSError("simulated disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(memory, "atomic_json", fail_export)
        with pytest.raises(OSError):
            store.commit_update([NewMemory("user_fact", "Uses Rust", 8)], next_state, "turn", state_path)
    assert load_state(state_path) == SelfState()  # previous JSON is still valid
    store.close()
    del store
    recovered = Agent(Config(tmp_path), fake_llm())
    try:
        assert recovered.state == next_state
        assert recovered.memory.all()[0].content == "Uses Rust"
        assert load_state(state_path) == next_state
        assert recovered.memory.connection.execute("SELECT count(*) FROM pending_state").fetchone()[0] == 0
    finally:
        recovered.close()


def test_database_batch_failure_rolls_back_all_memories(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    try:
        with pytest.raises(ValueError):
            store.commit_update([
                NewMemory("user_fact", "Valid fact", 5), NewMemory("bad_type", "Invalid", 5),
            ], SelfState(), "turn", tmp_path / "self_state.json")
        assert store.all() == []
        assert store.connection.execute("SELECT count(*) FROM pending_state").fetchone()[0] == 0
    finally:
        store.close()

