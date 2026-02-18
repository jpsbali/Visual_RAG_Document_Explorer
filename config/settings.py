"""
Global configuration settings for Visual RAG Document Explorer.

Uses pydantic-settings for environment variable management with .env file support.
All settings can be overridden via environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""
    
    # ============= API Keys =============
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    voyage_api_key: str = ""
    cohere_api_key: str = ""

    # ============= LLM Settings =============
    default_llm_provider: Literal["openai", "openrouter"] = "openai"
    default_model: str = "gpt-4o"
    openrouter_model: str = "anthropic/claude-sonnet-4"
    temperature: float = 0.0
    max_tokens: int = 2048

    # ============= Embedding Settings =============
    default_embedding_model: Literal["voyage", "bge-m3"] = "voyage"
    voyage_model: str = "voyage-3"
    bge_model: str = "BAAI/bge-m3"

    # ============= Vector DB Settings =============
    default_vector_db: Literal["qdrant", "milvus"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    milvus_url: str = "http://localhost:19530"
    milvus_collection: str = "documents"

    # ============= Chunking Settings =============
    chunk_size_min: int = 256
    chunk_size_max: int = 1024
    chunk_overlap: int = 128

    # ============= NER Settings =============
    ner_mode: Literal["gliner", "spacy", "ensemble"] = "ensemble"
    gliner_model: str = "urchade/gliner_multi_pii-v1"
    spacy_model: str = "en_core_web_trf"
    custom_entity_types: list[str] = []

    # ============= Search Settings =============
    search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    dense_weight: float = 0.5
    sparse_weight: float = 0.3
    metadata_weight: float = 0.2

    # ============= RAG Settings =============
    default_rag_strategy: Literal["simple", "crag", "srag", "advanced", "auto"] = "auto"
    rerank_top_k: int = 20
    final_top_k: int = 5
    enable_hyde: bool = False
    enable_compression: bool = True
    srag_max_iterations: int = 3
    srag_quality_threshold: float = 0.8
    crag_relevance_threshold: float = 0.5
    dedup_similarity_threshold: float = 0.95
    grounding_threshold: float = 0.7

    # ============= Memory Settings =============
    short_term_memory_size: int = 20
    context_window_threshold: float = 0.6
    enable_long_term_memory: bool = True

    # ============= Agent Settings =============
    enable_streaming: bool = True
    max_graph_execution_time: int = 120
    llm_context_window: int = 128000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
