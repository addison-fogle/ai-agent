"""Offline demonstration: two child processes, one shared temporary directory.

The tiny scripted LLM demonstrates plumbing, not language understanding. On the
read run its reply is derived exclusively from the supplied memory context.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from agent import Agent
from config import Config


class DemoLLM:
    def generate(self, messages: list[dict[str, str]]) -> str:
        if messages[0]["content"].startswith("MEMORY EXTRACTION"):
            raw_user = messages[1]["content"].split("USER MESSAGE\n", 1)[1]
            user, _ = json.JSONDecoder().raw_decode(raw_user)
            memories = []
            if user.startswith("I'm building "):
                memories = [{
                    "memory_type": "user_fact", "content": "The user is building " + user[len("I'm building "):],
                    "importance": 8,
                }]
            return json.dumps({"memories": memories, "new_goals": [], "completed_goals": [],
                               "self_summary": None, "recent_beliefs": None})
        if messages[-1]["content"].startswith("USER MESSAGE\nI'm building "):
            return "I can help with that project."
        raw_memories = messages[1]["content"].split("RELEVANT MEMORIES\n", 1)[1]
        memories, _ = json.JSONDecoder().raw_decode(raw_memories)
        facts = [m["content"] for m in memories if m["memory_type"] == "user_fact"]
        return "Stored record: " + facts[0] if facts else "I have no stored record of your project."


def worker(mode: str, storage: Path) -> None:
    agent = Agent(Config(storage), DemoLLM())
    try:
        print(f"Process {os.getpid()} ({mode}). Recent turns at startup: {len(agent.recent)}")
        if mode == "write":
            print("Elias:", agent.respond("I'm building a compiler in Rust."))
            print(f"Saved {len(agent.memory.all())} memory to SQLite.")
        else:
            assert agent.recent == []
            assert agent.state.self_summary == ""
            print("Elias:", agent.respond("What project was I working on?"))
    finally:
        agent.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["write", "read"], help=argparse.SUPPRESS)
    parser.add_argument("--storage", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.storage is None:
            parser.error("--worker requires --storage")
        worker(args.worker, args.storage)
        return
    print("Offline persistence demonstration (scripted fake LLM; no API calls).", flush=True)
    with tempfile.TemporaryDirectory(prefix="persistent-agent-demo-") as directory:
        for mode in ("write", "read"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                            "--storage", directory], check=True, timeout=20)
    print("Both processes exited. Temporary demonstration storage removed.")


if __name__ == "__main__":
    main()
