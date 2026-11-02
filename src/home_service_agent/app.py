"""HTTP entry point for the Home-Service Agent."""

from fastapi import FastAPI

from .config import Settings


settings = Settings.from_env()
app = FastAPI(title="Home-Service Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a minimal liveness response."""

    return {"status": "ok", "service": settings.app_name}

