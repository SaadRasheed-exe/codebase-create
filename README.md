# Codebase Create

An LLM coding agent that turns a natural-language request into a working, tested Python project. It generates implementation and test files, runs them in a sandbox, reads the results, and iterates until all tests pass — fully autonomous.

## How It Works

```
prompt ──► agent_loop.run_agent()
              │
              ├─ turn 1:  model writes files via tools
              │           → run_tests() in sandbox
              │           → observation fed back
              ├─ turn N:  model reads failures, fixes code
              └─ DONE:    all tests pass  →  run ends
                           max_turns hit  →  failure summary
                           model stuck    →  early termination
```

The agent loop is **provider-neutral**: every LLM backend (OpenAI, Anthropic, Ollama, NVIDIA, Mock) exposes the same `complete()` interface. Tools — `write_file`, `read_file`, `list_files`, `run_tests` — are defined as JSON schemas; the dispatcher executes them against a temporary workspace and returns observations the model can act on. An event stream (`TurnStarted`, `ToolCalled`, `ObservationReady`, …) drives both the Rich and plain-text renderers.

### Termination policy

The loop stops as soon as one of these fires:

| Condition | Meaning |
|-----------|---------|
| Model returns text with no tool calls | Task complete |
| `run_tests` returns `success=True` | Tests pass — done |
| Same observation hashes repeated 4× | Stuck loop — bail early |
| `max_turns` reached | Budget exhausted — summary |
| Provider or sandbox error | Infrastructure failure — report |

A `nudge` (one-turn "try a different approach" prompt) is inserted on the third identical observation before the stuck-loop detector fires, giving the model one free shot at course correction.

### Why fixed temperature

The legacy pipeline used adaptive temperature scheduling that added complexity without measurable benefit. The agent loop uses **fixed temperature 0.1** (configurable via `AGENT_GENERATION_TEMPERATURE`): low enough for precise tool arguments, high enough to break degenerate loops via the stuck-loop detector instead of thermal jitter.

## Installation

```bash
pip install -r requirements.txt
```

> **Python ≥ 3.10 required.** Uses `X | Y` union types and `dataclass(slots=True)`.

The `docker` sandbox driver requires Docker. For offline or daemon-less use, the `subprocess` sandbox is the default.

### Backends

| Backend | Key env var(s) | Notes |
|---------|----------------|-------|
| `mock` | — (none) | Deterministic, offline; uses scenario presets |
| `ollama` | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) | Local models via OpenAI-compatible API |
| `nvidia` | `nvidia_api_key` | NVIDIA-hosted endpoints; base URL auto-set |
| `openai` | `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini, etc. |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude 3.5 Sonnet, etc. |

Set `AGENT_BACKEND` in your `.env` or environment. Copy `.env.example` to `.env` and fill in keys for the backends you use.

### Mock scenarios

Use `--mock-scenario` to exercise specific agent-loop behaviors offline:

| Scenario | Behavior |
|----------|----------|
| `happy_path` | Writes two files, tests pass immediately |
| `fix_after_failure` | First `run_tests` fails, second passes |
| `stuck_loop` | Always returns the same broken observation |
| `bad_tool_args` | Returns tool calls with wrong parameter names |
| `premature_finish` | Claims success without calling `run_tests` |

## Quickstart

```bash
# Offline (mock backend — works instantly, no keys needed)
python app.py "Build a factorial function."

# With Ollama
AGENT_BACKEND=ollama python app.py "Build a palindrome checker."

# With NVIDIA
AGENT_BACKEND=nvidia python app.py "Build a Fibonacci generator."
```

### CLI flags

```bash
python app.py "prompt" \
  --backend mock               # override AGENT_BACKEND
  --mock-scenario fix_after_failure
  --sandbox subprocess         # subprocess | docker
  --max-turns 20               # loop budget (default 12)
  --ui rich                    # rich | plain
  --json                       # machine-readable report
  --keep-artifacts             # preserve temp workspace
  --repl                       # interactive mode (default when no prompt)
```

### JSON output

```bash
python app.py "Build a binary search." --json --backend mock --sandbox subprocess
```

Produces a single JSON object with `success`, `failure_category`, `turns` (each with `tool_calls`, `tool_results`, `text`), `files` (workspace-relative path → content), and token counts. Pipe into `jq` for scripting.

## REPL (interactive mode)

Run without a prompt (or with `--repl`) to enter an interactive session:

```bash
python app.py                # default backend is mock
python app.py --backend ollama
```

Commands:

| Command | Effect |
|---------|--------|
| Any text | Send to the agent as a prompt |
| `/files` | List current workspace files |
| `/reset` | Discard workspace, start fresh |
| `/help` | Show available commands |
| `/exit`, `/quit`, `Ctrl-D` | Exit (exit code 0) |

The REPL preserves workspace state across prompts. Run `/reset` between unrelated tasks.

## Configuration

### Environment variables

```bash
AGENT_BACKEND=mock                   # mock | ollama | nvidia | openai | anthropic
AGENT_MODEL=google/gemma-2-2b-it     # model name (ignored by mock)
AGENT_TEST_TIMEOUT=15                # seconds per pytest run
AGENT_GENERATION_TEMPERATURE=0.1     # model sampling temperature
AGENT_SANDBOX=subprocess             # subprocess | docker
AGENT_KEEP_ARTIFACTS=false           # keep temp workspace for debugging
AGENT_MOCK_SCENARIO=happy_path       # mock scenario preset
AGENT_MAX_TOKENS=4096                # response token cap
AGENT_MAX_TURNS=12                   # agent loop budget
AGENT_DOCKER_IMAGE=python:3.11-slim  # base image for docker sandbox
AGENT_DOCKER_NETWORK_DISABLED=true   # isolate container networking
AGENT_DOCKER_MEMORY=512m             # container memory limit
AGENT_DOCKER_CPUS=1.0                # container CPU quota

# Provider keys
nvidia_api_key=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434/v1
OPENAI_BASE_URL=                     # override for OpenAI-compatible endpoints
```

### Docker sandbox

When `AGENT_SANDBOX=docker`, tests run in an isolated container. If the configured image lacks `pytest`, a derived image is built automatically with it installed.

```bash
python app.py "..." --sandbox docker
python app.py "..." --sandbox docker --docker-memory 1g --docker-cpus 1.5
```

> **Security note:** The Docker sandbox disables networking and enforces memory/CPU limits. For fully offline or daemon-less use, the `subprocess` sandbox is the default.

### Model thinking (reasoning traces)

Some models (Ollama qwen3, Anthropic Claude) can emit a reasoning trace alongside their response. This is hidden by default to keep output clean.

```bash
# Show thinking when the model produces it
python app.py "Build factorial." --backend ollama --model qwen3:8b --thinking

# Enable thinking request for Anthropic (opt-in via API)
python app.py "Build factorial." --backend anthropic --enable-thinking --thinking
```

Provider behavior:

| Provider | Thinking | How to view | Token accounting |
|----------|----------|-------------|------------------|
| Ollama (qwen3) | Auto-enabled | `--thinking` | Not separated by Ollama |
| Anthropic | Opt-in (`--enable-thinking`) | `--thinking` | Subtracted from output tokens |
| OpenAI (o-series) | Always-on, hidden | N/A (no content exposed) | Reported in JSON output |
| NVIDIA | Not supported | — | — |

When hidden, a `[thinking hidden — use --thinking to show]` hint appears in the output.

## Architecture

### Component map

| Module | Role |
|--------|------|
| `agent_loop.py` | Core loop: calls provider, dispatches tools, emits events, enforces termination |
| `providers/base.py` | `Provider` ABC and `ProviderError` |
| `providers/mock.py` | Scenario-driven mock provider for offline testing |
| `providers/mock_scenarios.py` | Scenario definitions (happy_path, fix_after_failure, …) |
| `providers/openai_compat.py` | OpenAI, Ollama, NVIDIA via `openai` SDK |
| `providers/anthropic_provider.py` | Anthropic Claude via `anthropic` SDK |
| `tools.py` | Tool dispatcher: write_file, read_file, list_files, run_tests |
| `executor.py` | Temporary workspace management and sandboxed pytest execution |
| `sandboxes/` | Sandbox drivers: subprocess and Docker |
| `test_results.py` | JUnit XML parsing and failure categorization |
| `config.py` | `AgentConfig` dataclass, env-var resolution |
| `models.py` | Shared vocabulary: `ToolCall`, `ToolResult`, `AgentTurn`, events, report |
| `prompts.py` | System prompt for the agent loop |
| `ui.py` | `RichRenderer` (interactive terminal) and `PlainRenderer` (scripting/JSON) |
| `repl.py` | Interactive REPL with persistent workspace |
| `app.py` | CLI entry point |

### Event stream

Every agent-loop turn emits a sequence of event objects:

```
TurnStarted(i)
  AssistantReplied(text)
  ToolCalled(tool_call)
    ObservationReady(tool_result)
      ToolCalled(tool_call)
        ObservationReady(tool_result)
          …
RunFinished(success, turns_used)
```

Both `RichRenderer` and `PlainRenderer` subscribe to this stream via a callback. Adding a new renderer (e.g., Textual, web UI) means writing a single class that handles these event types — no changes to the agent loop.

### Why regenerate tests every iteration?

Tests are part of the solution spec. By regenerating them alongside implementation, the agent can refine test coverage, recover from buggy test suites, and adapt to discovered edge cases. The tradeoff is less predictable convergence, but the stuck-loop detector catches degenerate cases early.

## Testing

```bash
# Full suite (offline, deterministic, ~12s)
python -m pytest -q
```

> **ROS 2 note:** If you have ROS 2 on your `PYTHONPATH` (e.g., Jazzy), `pytest.ini` already sets `addopts = -p no:ros2娃娃` to suppress leaked global plugins.

### What the tests cover

- **Models & config**: dataclass construction, env-var resolution, defaults
- **Executor**: workspace creation, path-traversal guard, UTF-8 correctness, artifact cleanup
- **Tools**: dispatcher routing, write/read/list/run_tests round-trip, path safety
- **Providers**: mock scenarios (happy_path, stuck_loop, bad_tool_args, premature_finish), provider factory, error propagation, OpenAI-compat normalization
- **Agent loop**: termination conditions (pass, stuck_loop, nudge, budget, provider_error), event emission, token accounting, workspace snapshot on report
- **UI & REPL**: Rich/plain renderer output, REPL grammar (lifecycle, EOF, interrupt-at-prompt, unknown commands), CLI exit codes (0/1/2), `--json` output format

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `DockerException: Error while fetching server API version` | No Docker daemon running. Use `--sandbox subprocess` or start Docker. |
| `ValueError: No nvidia_api_key found` | Backend needs a key. Set it in `.env` or via the env var named in the error. |
| Agent loops without progress | `stuck_loop` category in the report. Increase `AGENT_MAX_TURNS` or try a different model. |
| Tests timeout | Increase `AGENT_TEST_TIMEOUT` (default 15s). If generated code has an infinite loop, the model needs a different approach. |
| `AGENT_BACKEND=openai` fails silently | No `.env` file or key not set. Check `python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"`. |

## Roadmap

- [ ] **Textual dashboard** — real-time terminal UI with turn history, file tree, and verdict badges
- [ ] **Retry / exponential backoff** for provider 429/503 errors
- [ ] **Streaming** — token-by-token display while the model generates
- [ ] **Selective repair** — keep working tests, regenerate only the implementation
- [ ] **Live smoke tests** — CI matrix against real backends (keys required)

## License

MIT
