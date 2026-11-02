"""Optional OpenRouter live-model adapter.

The rest of the application depends on this small interface, so tests use a
scripted model and no API key is required for normal development.
"""

from dataclasses import dataclass
import os
from typing import Any

import httpx
from dotenv import load_dotenv


DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str | None
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    site_url: str | None = None
    app_name: str = "HomeServiceAgent"

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        load_dotenv()
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
            base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            site_url=os.getenv("OPENROUTER_SITE_URL"),
            app_name=os.getenv("OPENROUTER_APP_NAME", "HomeServiceAgent"),
        )


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig | None = None, *, timeout: float = 60.0) -> None:
        self.config = config or OpenRouterConfig.from_env()
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.config.api_key)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if not self.config.api_key:
            raise OpenRouterConfigurationError(
                "OPENROUTER_API_KEY is not configured; use the scripted model for offline runs"
            )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.site_url:
            headers["HTTP-Referer"] = self.config.site_url
        if self.config.app_name:
            headers["X-Title"] = self.config.app_name

        with httpx.Client(base_url=self.config.base_url, timeout=self.timeout) as client:
            response = client.post("/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

