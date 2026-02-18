"""
OpenRouter LLM provider using httpx.

Provides both synchronous and streaming text generation using OpenRouter API.
"""

from typing import Optional, AsyncIterator
import json

import httpx

from config.settings import Settings


class OpenRouterProvider:
    """OpenRouter LLM provider wrapper."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, settings: Settings):
        """
        Initialize OpenRouter provider.

        Args:
            settings: Application settings with API key and model config
        """
        self.settings = settings
        self.api_key = settings.openrouter_api_key
        self.model = settings.openrouter_model

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from OpenRouter.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Returns:
            Generated text response
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_tokens,
        }

        # Make API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.API_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # Extract response text
        return data["choices"][0]["message"]["content"]

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from OpenRouter.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Yields:
            Text chunks as they are generated
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_tokens,
            "stream": True,
        }

        # Make streaming API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                self.API_URL,
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()

                # Parse SSE stream
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        # Skip [DONE] marker
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.model
