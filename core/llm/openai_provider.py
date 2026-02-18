"""
OpenAI LLM provider using LangChain.

Provides both synchronous and streaming text generation using OpenAI models.
"""

from typing import Optional, AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings


class OpenAIProvider:
    """OpenAI LLM provider wrapper."""

    def __init__(self, settings: Settings):
        """
        Initialize OpenAI provider.

        Args:
            settings: Application settings with API key and model config
        """
        self.settings = settings
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.default_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from OpenAI.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Returns:
            Generated text response
        """
        # Create messages
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Override settings if provided
        llm = self.llm
        if temperature is not None or max_tokens is not None:
            llm = ChatOpenAI(
                api_key=self.settings.openai_api_key,
                model=self.settings.default_model,
                temperature=temperature if temperature is not None else self.settings.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.settings.max_tokens,
            )

        # Generate response
        response = await llm.ainvoke(messages)
        return response.content

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from OpenAI.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Yields:
            Text chunks as they are generated
        """
        # Create messages
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Override settings if provided
        llm = self.llm
        if temperature is not None or max_tokens is not None:
            llm = ChatOpenAI(
                api_key=self.settings.openai_api_key,
                model=self.settings.default_model,
                temperature=temperature if temperature is not None else self.settings.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.settings.max_tokens,
            )

        # Stream response
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.settings.default_model
