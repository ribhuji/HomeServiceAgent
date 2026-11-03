# Home-Service Agent

Local chat agent for booking, rescheduling, and cancelling home-service errands
against a deterministic mock provider API.

## Run locally with Docker

The recommended startup runs the LangGraph server and Agent Chat UI as two
local containers. Docker Desktop or any Docker-compatible runtime works. On
macOS, the lightweight Colima runtime is enough:

```bash
colima start
```

Provide the OpenRouter key through the shell; it is passed only to the
LangGraph container and never to the browser UI:

```bash
read -s OPENROUTER_API_KEY
export OPENROUTER_API_KEY
docker-compose up --build
```

Open `http://localhost:3000`. Stop the stack with `Ctrl+C`, or use
`docker-compose down` from another terminal.

Confirmed area and service preferences are stored in the ignored `data/agent.db`
SQLite file for the local demo. This deployment is intentionally single-user:
all threads share the fixed long-term-memory owner `home-service-user`.
Approval records and sanitized run traces are stored in the same local data
directory. Traces can be inspected under `data/traces/<thread-id>.jsonl`.
Each user turn is bounded by `HOME_SERVICE_AGENT_MAX_MODEL_TURNS`,
`HOME_SERVICE_AGENT_MAX_TOOL_CALLS`, and `HOME_SERVICE_AGENT_MAX_WALL_SECONDS`
(defaults: 3, 12, and 300 seconds). The model client and provider client also
own their individual request timeouts.

Useful local debugging commands:

```bash
docker-compose ps
docker-compose logs -f langgraph web
curl http://localhost:2024/info
curl http://localhost:2024/threads/<thread-id>
curl http://localhost:2024/threads/<thread-id>/state
curl http://localhost:2024/threads/<thread-id>/history
# inspect the corresponding local trace
tail -n +1 data/traces/<thread-id>.jsonl
```

The thread endpoints expose the current local development runtime's graph
status, checkpoint, and recent messages. The `langgraph dev` runtime is
in-memory, so its server threads are lost when that container restarts.

Rebuild after code or dependency changes:

```bash
docker-compose up --build
```

## Test

Run tests:

```bash
pytest
```

The implementation plan is tracked in
[IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md), and the design is
in [DESIGN.md](DESIGN.md).

## Submission deliverables

- [Design note](DESIGN.md): state model, approval, recovery, memory, and observability.
- [Generated eval table](deliverables/eval_table.csv): produced by `scripts/run_evals.py`.
- [Bug report](BUGS.md): three implementation bugs and their regression tests.
- [Timeout-after-success trace](deliverables/timeout-after-success.jsonl): raw JSONL trace.
- Recorded demo: to be added after the local screen recording.

## Optional live OpenRouter model

The offline scripted model is used by tests and evals. For a live demo, create a
local environment file and add an OpenRouter key:

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
```

The default model is `nvidia/nemotron-3.5-super-vl:free`. Never commit
`.env`; it is ignored by git.

## Current scope

The live agent gives the model read-only provider tools and approval-request
tools. LangGraph stores the workflow state; code, not the model, records exact
approval and invokes the mock hold, book, cancel, or reschedule operation. The
booking coordinator verifies uncertain results through the authoritative status
tool. The UI and server are local; OpenRouter is only the optional model
endpoint. Tests and evals use deterministic scripted models and mock scenarios.

