"""
Embedding service router.

Routes embedding requests to the appropriate service (Voyage or BGE-M3)
based on configuration.
"""

from typing import Union

from config.settings import Settings
from core.embeddings.voyage_embeddings import VoyageEmbeddings
from core.embeddings.bge_m3_embeddings import BGEM3Embeddings


def get_embedding_service(
    settings: Settings,
) -> Union[VoyageEmbeddings, BGEM3Embeddings]:
    """
    Get the appropriate embedding service based on settings.

    Args:
        settings: Application settings

    Returns:
        Embedding service instance (Voyage or BGE-M3)

    Raises:
        ValueError: If embedding model is not supported or API key is missing
    """
    model = settings.default_embedding_model

    if model == "voyage":
        if not settings.voyage_api_key:
            raise ValueError("Voyage API key is required but not configured")
        return VoyageEmbeddings(settings)

    elif model == "bge-m3":
        return BGEM3Embeddings(settings)

    else:
        raise ValueError(
            f"Unsupported embedding model: {model}. "
            f"Supported models: voyage, bge-m3"
        )
