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

