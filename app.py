"""Terminal interface; inspection commands never require an API key."""

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from agent import Agent, StorageError
from config import Config
from llm import LLMError, OpenAILLMClient, redact_secrets


def display(value: object) -> None:
    print(redact_secrets(json.dumps(value, ensure_ascii=False, indent=2)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent agent v0.1: continuity from external memory")
    parser.add_argument("--debug", action="store_true", help="Show full contexts and proposed/committed updates")
    parser.add_argument("--storage", type=Path, help="Storage directory (one running process per directory)")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or newer is required")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    config = Config.from_env(args.storage)
    client = OpenAILLMClient(config.model)
    try:
        agent = Agent(config, client, debug=args.debug)
    except (OSError, ValueError, sqlite3.Error):
        print("Cannot load storage. Check JSON schemas, file permissions, and the database; existing files were not reset.", file=sys.stderr)
        return 1
    print(redact_secrets(f"{agent.identity.name}: Ready."))
    print("Commands: /memories /state /identity /debug /quit")
    try:
        while True:
            try:
                user = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user.casefold() in {"quit", "/quit"}:
                break
            if user == "/memories":
                display([asdict(m) for m in agent.memory.all()])
            elif user == "/state":
                display(asdict(agent.state))
            elif user == "/identity":
                display(asdict(agent.identity))
            elif user == "/debug":
                agent.debug = not agent.debug
                print(f"Debug {'on' if agent.debug else 'off'}.")
            elif user.startswith("/"):
                print("Unknown command. Use /memories /state /identity /debug /quit.")
            else:
                try:
                    print(redact_secrets(f"{agent.identity.name}: {agent.respond(user)}"))
                except LLMError as error:
                    print(redact_secrets(str(error)), file=sys.stderr)
                except StorageError as error:
                    print(str(error), file=sys.stderr)
                    return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Any committed update will be recovered on restart.")
    finally:
        agent.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

