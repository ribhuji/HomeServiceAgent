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

Useful local debugging commands:

```bash
docker-compose ps
docker-compose logs -f langgraph web
curl http://localhost:2024/info
curl http://localhost:2024/threads/<thread-id>
curl http://localhost:2024/threads/<thread-id>/state
curl http://localhost:2024/threads/<thread-id>/history
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

## Optional live OpenRouter model

The offline scripted model is used by tests and evals. For a live demo, create a
local environment file and add an OpenRouter key:

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
```

The default model is `nvidia/nemotron-3-ultra-550b-a55b:free`. Never commit
`.env`; it is ignored by git.

## Current scope

The live agent uses one model call to extract request fields, then a LangGraph
workflow calls the deterministic mock provider directly. The repository also
contains a tested SQLite checkpoint bootstrap utility, but the current
`langgraph dev` server remains in-memory for thread history. The live UI
demonstrates request collection, provider search, and durable confirmed
area/preferences; it does not perform real booking.

