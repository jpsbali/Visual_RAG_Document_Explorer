"""
Document summarization using LLM.

Generates concise summaries of documents using map-reduce for long documents.
"""

import tiktoken
from typing import Union

from core.llm.openai_provider import OpenAIProvider
from core.llm.openrouter_provider import OpenRouterProvider


class DocumentSummarizer:
    """Document summarizer using LLM."""

    # Summarization prompt from implementation guide
    SUMMARIZATION_PROMPT = """Generate a concise summary of the following document. The summary should:
1. Capture the main topics and key findings
2. Be 3-5 sentences long
3. Include the most important facts, figures, and conclusions
4. Be useful for understanding the document's scope at a glance

Document title: {filename}
Document content:
{content}

Provide a concise summary."""

    # Map-reduce prompts for long documents
    CHUNK_SUMMARY_PROMPT = """Summarize the following section of a document in 2-3 sentences, focusing on key information:

{chunk}

Summary:"""

    FINAL_SUMMARY_PROMPT = """Combine the following section summaries into a single coherent summary of 3-5 sentences:

{summaries}

Final summary:"""

    def __init__(
        self,
        llm_provider: Union[OpenAIProvider, OpenRouterProvider],
        max_chunk_tokens: int = 4000,
    ):
        """
        Initialize document summarizer.

        Args:
            llm_provider: LLM provider to use for summarization
            max_chunk_tokens: Maximum tokens per chunk for map-reduce
        """
        self.llm_provider = llm_provider
        self.max_chunk_tokens = max_chunk_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def summarize(self, document: str, filename: str = "document") -> str:
        """
        Summarize a document.

        For short documents (<4000 tokens), summarizes directly.
        For long documents, uses map-reduce: chunk → summarize each → combine.

        Args:
            document: Document text to summarize
            filename: Document filename for context

        Returns:
            Summary text (3-5 sentences)
        """
        # Count tokens
        token_count = len(self.tokenizer.encode(document))

        if token_count <= self.max_chunk_tokens:
            # Short document - summarize directly
            return await self._summarize_direct(document, filename)
        else:
            # Long document - use map-reduce
            return await self._summarize_map_reduce(document, filename)

    async def _summarize_direct(self, document: str, filename: str) -> str:
        """
        Summarize a document directly.

        Args:
            document: Document text
            filename: Document filename

        Returns:
            Summary text
        """
        prompt = self.SUMMARIZATION_PROMPT.format(
            filename=filename,
            content=document,
        )

        summary = await self.llm_provider.generate(
            prompt=prompt,
            temperature=0.3,  # Slightly higher for more natural summaries
            max_tokens=300,
        )

        return summary.strip()

    async def _summarize_map_reduce(self, document: str, filename: str) -> str:
        """
        Summarize a long document using map-reduce.

        Args:
            document: Document text
            filename: Document filename

        Returns:
            Summary text
        """
        # Split document into chunks
        chunks = self._split_into_chunks(document)

        # Map: Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            prompt = self.CHUNK_SUMMARY_PROMPT.format(chunk=chunk)
            summary = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200,
            )
            chunk_summaries.append(summary.strip())

        # Reduce: Combine chunk summaries
        combined_summaries = "\n\n".join(
            [f"Section {i+1}: {summary}" for i, summary in enumerate(chunk_summaries)]
        )

        prompt = self.FINAL_SUMMARY_PROMPT.format(summaries=combined_summaries)
        final_summary = await self.llm_provider.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=300,
        )

        return final_summary.strip()

    def _split_into_chunks(self, text: str) -> list[str]:
        """
        Split text into chunks based on token count.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))

            if current_tokens + para_tokens > self.max_chunk_tokens and current_chunk:
                # Current chunk is full, start new chunk
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                # Add to current chunk
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add remaining chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    async def summarize_batch(
        self, documents: list[tuple[str, str]]
    ) -> list[str]:
        """
        Summarize multiple documents.

        Args:
            documents: List of (document_text, filename) tuples

        Returns:
            List of summaries
        """
        summaries = []
        for document, filename in documents:
            summary = await self.summarize(document, filename)
            summaries.append(summary)
        return summaries
