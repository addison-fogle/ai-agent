"""Controller: load context -> stateless response -> validated durable updates."""

import json
import logging
import sqlite3
from dataclasses import asdict
from typing import Callable
from uuid import uuid4

from config import Config
from llm import LLMClient, LLMError, redact_secrets
from memory import MemoryStore, load_identity, load_state
from models import Extraction
from prompts import extraction_messages, response_messages

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    pass


class Agent:
    def __init__(self, config: Config, llm: LLMClient, *, debug: bool = False,
                 debug_sink: Callable[[str], None] = print):
        if config.recent_turn_limit < 0 or config.memory_limit < 0:
            raise ValueError("Context limits must be non-negative")
        self.config, self.llm = config, llm
        self.debug, self.debug_sink = debug, debug_sink
        self.recent: list[dict[str, str]] = []
        self.last_messages: list[dict[str, str]] = []
        self.last_extraction_messages: list[dict[str, str]] = []
        self.last_warning: str | None = None
        self.identity = load_identity(config.storage_dir / "identity.json")
        self.state_path = config.storage_dir / "self_state.json"
        self.memory = MemoryStore(config.storage_dir / "memories.db")
        try:
            self.memory.recover_state(self.state_path)
            self.state = load_state(self.state_path)
        except Exception:
            self.memory.close()
            raise
        self._trace("IDENTITY LOADED", asdict(self.identity))
        self._trace("SELF-STATE LOADED", asdict(self.state))

    def close(self) -> None:
        self.memory.close()

    def _trace(self, title: str, value: object) -> None:
        if self.debug:
            self.debug_sink(redact_secrets(f"[DEBUG] {title}\n{json.dumps(value, ensure_ascii=False, indent=2)}"))

    def respond(self, user: str) -> str:
        user = redact_secrets(user.strip())
        if not user:
            raise ValueError("Message cannot be empty")
        self.last_warning = None
        retrieved = self.memory.retrieve(user, self.config.memory_limit)
        self._trace("RETRIEVED MEMORIES", [asdict(m) for m in retrieved])
        self.last_messages = response_messages(self.identity, self.state, retrieved, self.recent, user)
        self._trace("RESPONSE CONTEXT SENT TO LLM", self.last_messages)
        response = redact_secrets(self.llm.generate(self.last_messages))
        if not response.strip():
            raise LLMError("LLM returned an empty response; no turn was saved.")
        self.last_extraction_messages = extraction_messages(self.state, retrieved, self.recent, user, response)
        self._trace("EXTRACTION CONTEXT SENT TO LLM", self.last_extraction_messages)
        try:
            raw = redact_secrets(self.llm.generate(self.last_extraction_messages))
            self._trace("MEMORY UPDATES PROPOSED (UNVALIDATED)", raw)
            update = Extraction.parse(raw)
            next_state = update.apply(self.state)
        except (ValueError, RecursionError, LLMError) as error:
            self.last_warning = f"Memory extraction skipped ({type(error).__name__}); persistent state unchanged."
            logger.warning(self.last_warning)
            self._trace("MEMORY UPDATES COMMITTED", {"skipped": True})
        else:
            try:
                inserted = self.memory.commit_update(update.memories, next_state, str(uuid4()), self.state_path)
            except (OSError, sqlite3.Error) as error:
                raise StorageError(
                    f"Persistence could not finish ({type(error).__name__}). Stop and restart after fixing storage; "
                    "a committed update will be recovered from SQLite."
                ) from None
            self.state = next_state
            self._trace("MEMORY UPDATES COMMITTED", {
                "inserted_ids": inserted, "duplicates_skipped": len(update.memories) - len(inserted),
                "self_state": asdict(self.state),
            })
        self.recent.extend([{"role": "user", "content": user}, {"role": "assistant", "content": response}])
        count = 2 * self.config.recent_turn_limit
        self.recent = self.recent[-count:] if count else []
        return response

