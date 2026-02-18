"""
LLM provider router.

Routes LLM requests to the appropriate provider (OpenAI or OpenRouter)
based on configuration.
"""

from typing import Union

from config.settings import Settings
from core.llm.openai_provider import OpenAIProvider
from core.llm.openrouter_provider import OpenRouterProvider


def get_llm_provider(settings: Settings) -> Union[OpenAIProvider, OpenRouterProvider]:
    """
    Get the appropriate LLM provider based on settings.

    Args:
        settings: Application settings

    Returns:
        LLM provider instance (OpenAI or OpenRouter)

    Raises:
        ValueError: If provider is not supported or API key is missing
    """
    provider = settings.default_llm_provider

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required but not configured")
        return OpenAIProvider(settings)

    elif provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OpenRouter API key is required but not configured")
        return OpenRouterProvider(settings)

    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: openai, openrouter"
        )
