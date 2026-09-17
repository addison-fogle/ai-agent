# Persistent agent · v0.1

A small terminal agent named Elias. Each model invocation can be stateless while
the application appears to maintain an identity across restarts. The code lives
at the repository root; no framework, background process, or vector database is involved.

```text
persistent identity
+ autobiographical memory
+ retrieved episodic memory
+ persistent goals
+ reconstruction of context
= behavioral continuity
```

This project **does not demonstrate consciousness, subjective experience,
continuous awareness, genuine emotions, or personal identity in the philosophical
sense**. Its self-summary and beliefs are editable application data. An old model
invocation ends; information survives on disk; a new invocation receives that
information; behavioral continuity appears.

## Run

Requires Python 3.12 or newer. From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-5-mini"  # optional; this is the default
python app.py --debug
```

Use a Responses API model available to your account. `.env.example` documents the
variables; files named `.env` are **not** loaded automatically. Never commit keys.
The only third-party runtime dependency is the OpenAI SDK; pytest is for testing.

```text
Elias: Ready.
You: I'm building a compiler in Rust.
Elias: ...
You: /memories
... inspect whether the extractor actually saved the fact ...
You: /quit

# Start python app.py again in a new process:
Elias: Ready.
You: What project was I working on?
Elias: ... answer using retrieved records ...
```

Live model wording and extraction decisions vary. If extraction fails, the reply
still appears, a warning is logged, and that turn adds no durable information.
Nothing from an earlier process exists in the recent conversation buffer.

| Command | Effect |
| --- | --- |
| `/memories` | Display all SQLite memory records, IDs and source turn IDs |
| `/state` | Display the current self-summary, goals, and beliefs |
| `/identity` | Display the loaded stable identity |
| `/debug` | Toggle full context and update tracing |
| `/quit` or `quit` | Exit; EOF and Ctrl-C also exit |

Inspection commands work without an API key. Normal storage defaults to `storage/`
beside `app.py`, regardless of the shell's working directory. For a separate,
private agent state, use `python app.py --storage /path/to/private/storage` or
`AGENT_STORAGE_DIR`. Use **one running process per storage directory**.

Missing storage directories and files are initialized automatically. Malformed
existing identity/state JSON causes a clear startup error and is not reset.
Manually edit `identity.json` while the app is stopped, then restart to load it.
The extractor cannot modify identity.

## Architecture

```text
User
  |
  v
Agent Controller
  +---- Identity (JSON; stable until manually edited)
  +---- Self State (JSON; updated after interactions)
  +---- Memory Retrieval (SQLite; keyword score)
  +---- Recent conversation (bounded; this process only)
  |
  v
Context Builder --> LLM call #1 --> Response
                                      |
                                      v
                            LLM call #2: Memory Extractor
                                      |
                               Validate entire JSON
                                      |
                         SQLite memories + state journal
                                      |
                             Atomic self-state JSON export
```

`agent.py` depends on the `LLMClient` protocol, whose only operation is
`generate(messages) -> str`. `llm.py` contains the OpenAI implementation. Every
request supplies the complete messages array, uses `store=False`, and omits
conversation and previous-response IDs. The SDK object is reused only as an HTTP
client; no model response state is chained between calls. See the official
[Responses API reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
and [default model documentation](https://developers.openai.com/api/docs/models/gpt-5-mini).
`store=False` controls response storage for API retrieval; it is not a blanket
promise of zero provider data retention.

The first context contains behavioral rules, identity, self-state, retrieved
memories, up to six recent user/assistant turn pairs, and the new message. Stored
records occupy a data message; behavioral instructions occupy the system message.
Rules require evidence for memory claims, uncertainty when records conflict, and
no claims of consciousness or activity between runs. `self` memories describe
actions of previous instances.

The second call receives the user message, reply, previous state (including goals),
retrieved memories and recent turns. It proposes durable memories and state updates.
`models.py` validates exact fields, allowed types, bounded text/lists, integer
importance from 1–10, and goal transitions **before any writes**. Invalid JSON,
invalid schema, and extraction API errors leave durable state unchanged. No full
transcript is automatically stored. A source turn UUID links memories created in
one interaction; it is provenance metadata, not a stored transcript or proof.

The extraction schema is shown in `prompts.py`. A null summary or belief list
preserves it; a non-null value replaces it. Goals accumulate, and completion moves
a known goal out of the active list. Goals are deduplicated case-insensitively;
reopening or renaming them is outside v0.1. The terminal waits for both calls
before displaying the reply.

## Retrieval and persistence

For each memory, `memory.py` computes:

```text
3.0 × Jaccard(query keywords, memory keywords)
+ 0.3 × importance / 10
+ 0.2 / (1 + age_in_days / 30)
```

Keywords are case-folded Unicode word tokens with a small explicit stopword list.
Return the top seven; ties prefer newer timestamps, then higher IDs. With no
keyword overlap, importance and recency provide a fallback. Tests can supply
`now` to obtain identical scores. This is a ranking, not a relevance guarantee.
All records are scanned, deliberately keeping the algorithm inspectable.
`MemoryStore.retrieve()` is the seam for a future semantic retriever.

Duplicate detection uses memory type plus normalized content. It ignores case,
whitespace, and the articles “a”, “an”, and “the”; it preserves punctuation,
negation, token order, and different programming language names. Matching entries
are skipped without changing their original provenance or importance. This is
conservative near-duplicate detection; paraphrases are not recognized.

SQLite commits the new memories and a proposed state snapshot in one transaction.
Then a temporary JSON file is flushed and atomically replaces `self_state.json`.
Only after that export succeeds is the pending snapshot removed. If the process
stops between steps, the next startup replays the SQLite snapshot idempotently.
If a pending update exists, recovery takes precedence over manual state edits.
Tests simulate an interrupted export and verify recovery with a new instance.
This handles process interruption; it is not a cross-platform power-loss or
concurrent-writer guarantee. Stop the app before copying or editing its storage.

## Inspect and test

Debug mode prints the loaded identity/state, retrieved records, **both complete
message arrays**, raw proposed extraction, committed IDs, skipped duplicates and
resulting state. The same context is available as `Agent.last_messages` and
`Agent.last_extraction_messages`. These are explicit prompt inputs, not hidden
reasoning or a record of model internals. Debug output is not saved automatically.

The API key is never placed in configuration dumps or prompts. Known environment
API key values and recognizable `sk-...` tokens are redacted from user input,
model output, and debug/inspection output. API exception bodies are not printed.
This is not a general secret detector: don't paste credentials. Debug output can
contain personal information, and memories are plaintext on disk. The included
identity/state JSON files are versionable seeds: use a separate storage directory
for real personal data, or carefully review changes before committing. SQLite,
virtual environments and `.env` files are ignored by Git.

```bash
python -m pytest -q
python demo.py
```

The offline demo needs only the standard library. It creates clean temporary
storage, launches a writer process, waits for it to exit, and starts an independent
reader process. A scripted fake LLM extracts the compiler fact in the first run;
in the second, it builds its response only from the retrieved memory section.
The reader starts with zero recent turns and an empty self-summary. The demo
prints process IDs and removes its temporary storage on completion. It tests
persistence and context construction, not the capabilities of a real model.

Tests cover initialization, inserts, retrieval ranking, recency/ties, conservative
deduplication, identity loading, state serialization and updates, missing/corrupt
files, malformed extraction, API failures, debug redaction, inspection commands,
transaction rollback, journal recovery, and process-independent persistence. All
LLM tests use fakes or mocked SDK calls; no network or API key is required.

## Limitations and next experiments

Prompt rules cannot formally guarantee truthful memory claims from a generative
model. The extractor can save mistakes; schema validation checks shape, not truth.
Old contradictory facts remain in storage, summaries can drift, and keyword
retrieval can miss relevant records. Memory is unbounded; there is no deletion,
consolidation, semantic search, encryption or multi-user/concurrent-writer support.
State lists and recent context are bounded by explicit limits, not a tokenizer;
large input can exceed a model's context window. Two sequential calls add latency
and cost. A failed extraction loses that opportunity to store the interaction.

Three next experiments, intentionally **not implemented**:

1. Semantic retrieval using embeddings: compare recall on paraphrased questions
   against the deterministic keyword baseline.
2. Memory decay and consolidation: track whether reducing redundant records
   improves retrieval without erasing useful facts or provenance.
3. Clone one storage directory into two agents: feed different experiences into
   each and compare how summaries, goals, and behavior diverge from shared origins.

## Repository tree

```text
ai-agent/
├── .env.example
├── .gitignore
├── README.md
├── agent.py
├── app.py
├── config.py
├── demo.py
├── llm.py
├── memory.py
├── models.py
├── prompts.py
├── pyproject.toml
├── requirements.txt
├── storage/
│   ├── identity.json
│   ├── self_state.json
│   └── memories.db          # created locally; ignored by Git
└── tests/
    ├── conftest.py
    ├── test_agent.py
    ├── test_cli.py
    ├── test_llm.py
    ├── test_memory.py
    └── test_persistence.py
```
