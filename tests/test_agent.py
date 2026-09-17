import json
from dataclasses import asdict

import pytest

from agent import Agent
from config import Config
from llm import LLMError
from memory import atomic_json, load_state
from models import Identity, NewMemory, SelfState


def test_identity_loading_and_no_rewrite(tmp_path, fake_llm):
    identity = Identity(name="Ada", personality=["calm"], core_goals=["help"])
    path = tmp_path / "identity.json"
    atomic_json(path, asdict(identity))
    before = path.read_bytes()
    agent = Agent(Config(tmp_path), fake_llm())
    try:
        assert agent.identity == identity
        assert path.read_bytes() == before
    finally:
        agent.close()


def test_missing_storage_files_and_state_roundtrip(tmp_path, fake_llm):
    root = tmp_path / "missing"
    agent = Agent(Config(root), fake_llm())
    agent.close()
    assert {p.name for p in root.iterdir()} == {"identity.json", "self_state.json", "memories.db"}
    state = SelfState("A prior instance helped with a lexer.", ["debug"], ["plan"], ["Lexer may be faulty."])
    atomic_json(root / "self_state.json", asdict(state))
    assert load_state(root / "self_state.json") == state


def test_state_updates_and_context_across_turns(tmp_path, fake_llm, make_extraction):
    first_update = make_extraction(
        memories=[{"memory_type": "user_fact", "content": "The user is building a compiler in Rust.", "importance": 8}],
        new_goals=["Debug the compiler"],
        self_summary="A prior instance discussed the user's Rust compiler.",
        recent_beliefs=["The compiler may need lexer work."],
    )
    second_update = make_extraction(completed_goals=["Debug the compiler"])
    llm = fake_llm("Let's debug it.", first_update, "The compiler is fixed.", second_update)
    agent = Agent(Config(tmp_path), llm)
    try:
        agent.respond("I'm building a compiler in Rust.")
        assert agent.state.active_goals == ["Debug the compiler"]
        agent.respond("The compiler is fixed now.")
        assert agent.state.active_goals == []
        assert agent.state.completed_goals == ["Debug the compiler"]
        assert agent.state.recent_beliefs == ["The compiler may need lexer work."]
        assert load_state(tmp_path / "self_state.json") == agent.state
        assert len(llm.calls) == 4
        context = json.dumps(llm.calls[2])
        for text in ("IDENTITY", "CURRENT SELF STATE", "RELEVANT MEMORIES", "RECENT CONVERSATION",
                     "USER MESSAGE", "BEHAVIORAL RULES", "Let's debug it.", "compiler in Rust"):
            assert text in context
        assert llm.calls[2] == agent.last_messages
        assert llm.calls[3] == agent.last_extraction_messages
        assert "Debug the compiler" in json.dumps(llm.calls[3])
    finally:
        agent.close()


@pytest.mark.parametrize("output", [
    "not JSON", "[]", "{}", "```json\n{}\n```",
    json.dumps({"memories": [{"memory_type": "self", "content": "hi", "importance": True}],
                "new_goals": [], "completed_goals": [], "self_summary": None, "recent_beliefs": None}),
    json.dumps({"memories": [], "new_goals": [], "completed_goals": ["invented goal"],
                "self_summary": None, "recent_beliefs": None}),
    json.dumps({"memories": [], "new_goals": "a string", "completed_goals": [],
                "self_summary": None, "recent_beliefs": None}),
    LLMError("secret-like remote error must not be logged"),
])
def test_bad_extraction_keeps_reply_and_preserves_storage(tmp_path, fake_llm, output, caplog):
    agent = Agent(Config(tmp_path), fake_llm("A valid reply.", output))
    before = (tmp_path / "self_state.json").read_bytes()
    try:
        assert agent.respond("hello") == "A valid reply."
        assert agent.memory.all() == []
        assert (tmp_path / "self_state.json").read_bytes() == before
        assert agent.last_warning is not None
        assert len(agent.recent) == 2
        assert "extraction skipped" in caplog.text
        assert "secret-like" not in caplog.text
    finally:
        agent.close()


def test_invalid_member_rejects_entire_batch(tmp_path, fake_llm, make_extraction):
    raw = make_extraction(memories=[
        {"memory_type": "goal", "content": "Valid", "importance": 7},
        {"memory_type": "goal", "content": "Invalid", "importance": 100},
    ], self_summary="Must not commit")
    agent = Agent(Config(tmp_path), fake_llm("reply", raw))
    try:
        agent.respond("hello")
        assert agent.memory.all() == []
        assert agent.state == SelfState()
    finally:
        agent.close()


def test_response_failure_does_not_extract_or_append(tmp_path, fake_llm):
    llm = fake_llm(LLMError("Unavailable"))
    agent = Agent(Config(tmp_path), llm)
    try:
        with pytest.raises(LLMError):
            agent.respond("hello")
        assert len(llm.calls) == 1
        assert agent.recent == []
        assert agent.memory.all() == []
    finally:
        agent.close()


@pytest.mark.parametrize("limit,expected", [(0, 0), (1, 2)])
def test_bounded_recent_turns(tmp_path, fake_llm, make_extraction, limit, expected):
    llm = fake_llm("one", make_extraction(), "two", make_extraction())
    agent = Agent(Config(tmp_path, recent_turn_limit=limit), llm)
    try:
        agent.respond("first")
        agent.respond("second")
        assert len(agent.recent) == expected
        assert len(agent.memory.all()) == 0  # no indiscriminate transcript storage
    finally:
        agent.close()


def test_debug_complete_but_redacts_credentials(tmp_path, fake_llm, make_extraction, monkeypatch):
    key = "sk-test-secret-123456789"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    log = []
    agent = Agent(Config(tmp_path), fake_llm("reply", make_extraction()), debug=True, debug_sink=log.append)
    try:
        agent.respond(f"My key is {key}")
        trace = "\n".join(log)
        for section in ("IDENTITY LOADED", "SELF-STATE LOADED", "RETRIEVED MEMORIES", "RESPONSE CONTEXT",
                        "EXTRACTION CONTEXT", "UPDATES PROPOSED", "UPDATES COMMITTED"):
            assert section in trace
        assert key not in trace
        assert key not in json.dumps(agent.last_messages)
        assert "[REDACTED]" in trace
    finally:
        agent.close()


@pytest.mark.parametrize("filename,bad_content", [("identity.json", "{broken"), ("self_state.json", '{"active_goals": 42}')])
def test_corrupt_existing_json_not_silently_reset(tmp_path, fake_llm, filename, bad_content):
    path = tmp_path / filename
    path.write_text(bad_content)
    with pytest.raises(ValueError):
        Agent(Config(tmp_path), fake_llm())
    assert path.read_text() == bad_content


def test_duplicate_extraction_records_only_one_memory(tmp_path, fake_llm, make_extraction):
    raw = make_extraction(memories=[{"memory_type": "user_fact", "content": "Uses Rust", "importance": 5}])
    agent = Agent(Config(tmp_path), fake_llm("one", raw, "two", raw))
    try:
        agent.respond("Uses Rust")
        agent.respond("Uses Rust")
        assert len(agent.memory.all()) == 1
    finally:
        agent.close()

