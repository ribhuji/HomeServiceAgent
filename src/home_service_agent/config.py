"""Application configuration for the local development skeleton."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_name: str = "home-service-agent"
    environment: str = "development"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_name=os.getenv("HOME_SERVICE_AGENT_NAME", cls.app_name),
            environment=os.getenv("HOME_SERVICE_AGENT_ENV", cls.environment),
        )

