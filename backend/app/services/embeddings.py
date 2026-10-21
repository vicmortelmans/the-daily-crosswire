import logging
from typing import Optional

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        key = settings.effective_embedding_api_key
        if not key or key == "sk-replace-me":
            raise RuntimeError(
                "No embedding API key configured. Set EMBEDDING_API_KEY (or "
                "fall back to OPENAI_API_KEY) in .env."
            )
        _client = AsyncOpenAI(
            api_key=key,
            base_url=settings.effective_embedding_api_base,
        )
        logger.info("DEBUG embeddings.py Initialized embedding client with base URL %s and API key %s", settings.effective_embedding_api_base, key)
    return _client


async def embed(texts: list[str]) -> list[list[float]]:
    """Compute embeddings for a batch of strings in one API call."""
    logger.info("DEBUG embeddings.py Invoke embedding model %s for %d texts", settings.embedding_model, len(texts))
    if not texts:
        return []
    resp = await _get_client().embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    vectors = [item.embedding for item in resp.data]
    if vectors and len(vectors[0]) != settings.embedding_dim:
        logger.warning(
            "embedding dim mismatch: model returned %d, expected %d (EMBEDDING_DIM in settings)",
            len(vectors[0]),
            settings.embedding_dim,
        )
    return vectors


async def embed_one(text: str) -> list[float]:
    [vec] = await embed([text])
    return vec
