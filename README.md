# Home-Service Agent

Local chat agent for booking, rescheduling, and cancelling home-service errands
against a deterministic mock provider API.

## Development setup

Requires Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Run and test

Start the complete local visual stack (LangGraph server, FastAPI compatibility
service, and Agent Chat UI):

```bash
HomeServiceAgent-venv/bin/python start.py
```

Then open `http://127.0.0.1:3000`. The launcher initializes ignored SQLite
files under `data/` and shuts down all child processes on Ctrl+C. Install the
UI dependencies once beforehand with:

```bash
cd ui/home-service-agent-ui
npm install
cd ../..
```

Start the local API:

```bash
uvicorn home_service_agent.app:app --reload
```

Then open `http://127.0.0.1:8000/health`.

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

