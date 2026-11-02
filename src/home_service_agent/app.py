"""HTTP entry point for the Home-Service Agent."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .chat import ChatAgent
from .config import Settings


settings = Settings.from_env()
app = FastAPI(title="Home-Service Agent", version="0.1.0")
chat_agent = ChatAgent()


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    phase: str
    message: str
    providers: list[dict]


@app.get("/health")
def health() -> dict[str, str]:
    """Return a minimal liveness response."""

    return {"status": "ok", "service": settings.app_name}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return ChatResponse(**chat_agent.handle(**request.model_dump()))

