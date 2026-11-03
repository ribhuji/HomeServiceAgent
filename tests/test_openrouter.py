import pytest

from home_service_agent.openrouter import (
    DEFAULT_MODEL,
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterConfigurationError,
)


def test_openrouter_config_uses_free_nemotron_default() -> None:
    config = OpenRouterConfig(api_key=None)

    assert config.model == DEFAULT_MODEL
    assert config.model == "nvidia/nemotron-3.5-super-vl:free"
    assert config.base_url == "https://openrouter.ai/api/v1"
    assert OpenRouterClient(config).configured is False


def test_missing_key_fails_before_network_call() -> None:
    client = OpenRouterClient(OpenRouterConfig(api_key=None))

    with pytest.raises(OpenRouterConfigurationError, match="OPENROUTER_API_KEY"):
        client.chat([{"role": "user", "content": "hello"}])

