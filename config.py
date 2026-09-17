"""Small environment-based configuration; credentials never enter prompts."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    storage_dir: Path
    model: str = "gpt-5-mini"
    memory_limit: int = 7
    recent_turn_limit: int = 6

    @classmethod
    def from_env(cls, storage_dir: Path | None = None) -> "Config":
        default = Path(__file__).resolve().parent / "storage"
        return cls(
            storage_dir=(storage_dir or Path(os.getenv("AGENT_STORAGE_DIR", str(default)))).expanduser(),
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        )

