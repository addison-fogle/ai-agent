"""SQLite memories, deterministic keyword retrieval, and atomic JSON storage."""

import json
import os
import re
import sqlite3
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import Identity, Memory, NewMemory, SelfState

STOP_WORDS = set("a an the i you me my what was were is are am be to of in on and it this that".split())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold())) - STOP_WORDS


def duplicate_key(text: str) -> str:
    # Conservative near duplicates: case, whitespace, and articles only.
    # Preserve punctuation, negation and token order (e.g. C vs C++).
    return " ".join(re.findall(r"\w+|[^\w\s]", re.sub(r"\b(a|an|the)\b", "", text.casefold())))


def atomic_json(path: Path, data: Any) -> None:
    """Replace a JSON file only after its complete contents have reached disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_identity(path: Path) -> Identity:
    if not path.exists():
        atomic_json(path, asdict(Identity()))
    return Identity.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_state(path: Path) -> SelfState:
    if not path.exists():
        atomic_json(path, asdict(SelfState()))
    return SelfState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def retrieval_score(memory: Memory, query: str, now: datetime) -> float:
    """Jaccard overlap + importance + 30-day recency; ties use timestamp/id."""
    query_tokens, content_tokens = tokens(query), tokens(memory.content)
    union = query_tokens | content_tokens
    overlap = len(query_tokens & content_tokens) / len(union) if union else 0.0
    age_days = max(0.0, (now - datetime.fromisoformat(memory.created_at)).total_seconds() / 86400)
    return 3.0 * overlap + 0.3 * memory.importance / 10 + 0.2 / (1 + age_days / 30)


class MemoryStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        with self.connection:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    memory_type TEXT NOT NULL CHECK (memory_type IN
                        ('user_fact', 'episodic', 'self', 'goal', 'decision')),
                    content TEXT NOT NULL,
                    importance INTEGER NOT NULL CHECK (importance BETWEEN 1 AND 10),
                    source_turn_id TEXT NOT NULL,
                    duplicate_key TEXT NOT NULL,
                    UNIQUE(memory_type, duplicate_key)
                );
                CREATE TABLE IF NOT EXISTS pending_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL
                );
            """)

    def close(self) -> None:
        self.connection.close()

    def all(self) -> list[Memory]:
        rows = self.connection.execute("""
            SELECT id, created_at, memory_type, content, importance, source_turn_id
            FROM memories ORDER BY id
        """).fetchall()
        return [Memory(**dict(row)) for row in rows]

    def _insert(self, memory: NewMemory, source_turn_id: str, created_at: str) -> int | None:
        # Also validate callers that construct NewMemory directly, outside extraction.
        memory = NewMemory.from_dict(asdict(memory))
        cursor = self.connection.execute("""
            INSERT INTO memories
            (created_at, memory_type, content, importance, source_turn_id, duplicate_key)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_type, duplicate_key) DO NOTHING
        """, (created_at, memory.memory_type, memory.content, memory.importance,
              source_turn_id, duplicate_key(memory.content)))
        return cursor.lastrowid if cursor.rowcount else None

    def add(self, memory: NewMemory, source_turn_id: str, *, created_at: datetime | None = None) -> int | None:
        timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        with self.connection:
            return self._insert(memory, source_turn_id, timestamp)

    def retrieve(self, query: str, limit: int = 7, *, now: datetime | None = None) -> list[Memory]:
        """Replace this method to experiment with semantic retrieval later."""
        if limit < 0:
            raise ValueError("Memory limit must be non-negative")
        now = now or datetime.now(timezone.utc)
        return sorted(self.all(), key=lambda m: (
            retrieval_score(m, query, now), m.created_at, m.id
        ), reverse=True)[:limit]

    def recover_state(self, state_path: Path) -> None:
        """Finish a previously committed SQLite update whose JSON export was interrupted."""
        row = self.connection.execute("SELECT payload FROM pending_state WHERE id = 1").fetchone()
        if row:
            state = SelfState.from_dict(json.loads(row["payload"]))
            atomic_json(state_path, asdict(state))
            with self.connection:
                self.connection.execute("DELETE FROM pending_state WHERE id = 1")

    def commit_update(self, memories: list[NewMemory], state: SelfState,
                      source_turn_id: str, state_path: Path) -> list[int]:
        """One DB transaction, then replayable JSON export. Use one process per directory."""
        self.recover_state(state_path)
        SelfState.from_dict(asdict(state))
        timestamp = datetime.now(timezone.utc).isoformat()
        inserted = []
        with self.connection:
            for memory in memories:
                memory_id = self._insert(memory, source_turn_id, timestamp)
                if memory_id is not None:
                    inserted.append(memory_id)
            self.connection.execute("INSERT OR REPLACE INTO pending_state VALUES (1, ?)",
                                    (json.dumps(asdict(state)),))
        self.recover_state(state_path)
        return inserted

