"""Embedding providers for Memoria — with LRU cache, retry, and automatic fallback."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import struct
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional

from memoria.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


# ── Abstract Base ─────────────────────────────────────


class EmbeddingProvider(ABC):
    """Base interface for all embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts. Default: sequential calls."""
        return [await self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


# ── OpenAI Provider ───────────────────────────────────


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding-3-small (1536 dimensions).

    Requires the `openai` package and a valid OPENAI_API_KEY env variable.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._model = model
        self._client: Optional[object] = None

    def _get_client(self):
        """Lazy-init the async OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI()
            except ImportError as e:
                raise EmbeddingError(
                    "openai package not installed. Install with: pip install openai"
                ) from e
        return self._client

    async def embed(self, text: str) -> List[float]:
        """Generate embedding via OpenAI API."""
        client = self._get_client()
        response = await client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding via OpenAI API (single call, more efficient)."""
        if not texts:
            return []
        client = self._get_client()
        response = await client.embeddings.create(input=texts, model=self._model)
        # Response items are sorted by index
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    @property
    def dimensions(self) -> int:
        """text-embedding-3-small produces 1536-dimensional vectors."""
        return 1536


# ── Local Provider (sentence-transformers) ────────────


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding using sentence-transformers (BAAI/bge-small-en-v1.5, 384 dims).

    Requires the `sentence-transformers` package.
    Falls back gracefully if unavailable.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: Optional[object] = None

    def _get_model(self):
        """Lazy-init the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except ImportError as e:
                raise EmbeddingError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                ) from e
        return self._model

    async def embed(self, text: str) -> List[float]:
        """Generate embedding using local model (run in executor to avoid blocking)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, text)

    def _embed_sync(self, text: str) -> List[float]:
        """Synchronous embedding computation."""
        model = self._get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding using local model."""
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_batch_sync, texts)

    def _embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch embedding."""
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    @property
    def dimensions(self) -> int:
        """BAAI/bge-small-en-v1.5 produces 384-dimensional vectors."""
        return 384


# ── Dummy Provider (testing) ──────────────────────────


class DummyEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based fake embeddings for testing.

    Generates consistent 1536-dim vectors from content hash.
    NOT suitable for production — only for unit tests.
    """

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions

    async def embed(self, text: str) -> List[float]:
        """Generate a deterministic pseudo-embedding from text hash."""
        digest = hashlib.sha512(text.encode("utf-8")).digest()
        # Extend hash bytes to fill the required dimensions
        needed_bytes = self._dimensions * 4  # 4 bytes per float32
        repeated = digest * ((needed_bytes // len(digest)) + 1)
        raw_bytes = repeated[:needed_bytes]

        # Unpack as float32 values and normalize to [-1, 1]
        values = list(struct.unpack(f"<{self._dimensions}f", raw_bytes))
        # Clamp and normalize
        norm = max(1e-10, sum(v * v for v in values) ** 0.5)
        return [v / norm for v in values]

    @property
    def dimensions(self) -> int:
        return self._dimensions


# ── Cached + Retry + Fallback Wrapper ─────────────────


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wraps any EmbeddingProvider with LRU cache, exponential-backoff retry, and fallback.

    Features:
    - **LRU Cache**: content hash → embedding, up to `cache_size` entries (default 10000).
    - **Retry**: On failure, retries up to 3 times with exponential backoff (1s, 2s, 4s).
    - **Fallback**: If primary provider repeatedly fails, switches to fallback provider
      and logs a warning.

    Example:
        provider = CachedEmbeddingProvider(
            primary=OpenAIEmbeddingProvider(),
            fallback=LocalEmbeddingProvider(),
        )
        embedding = await provider.embed("hello world")
    """

    def __init__(
        self,
        primary: EmbeddingProvider,
        fallback: Optional[EmbeddingProvider] = None,
        cache_size: int = 10000,
        max_retries: int = 3,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_size = cache_size
        self._max_retries = max_retries
        self._using_fallback = False

    def _cache_key(self, text: str) -> str:
        """Compute a short hash key for cache lookup."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    async def embed(self, text: str) -> List[float]:
        """Embed with cache check, retry, and fallback."""
        key = self._cache_key(text)

        # Cache hit
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        # Cache miss — compute embedding
        result = await self._embed_with_retry(text)

        # Store in cache with LRU eviction
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)  # Remove oldest
        self._cache[key] = result
        return result

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with per-item caching."""
        results: List[List[float]] = []
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Separate cached vs uncached
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                self._cache.move_to_end(key)
                results.append(self._cache[key])
            else:
                results.append([])  # placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Compute uncached embeddings
        if uncached_texts:
            provider = self._fallback if self._using_fallback else self._primary
            try:
                new_embeddings = await provider.embed_batch(uncached_texts)
            except Exception:
                # Fallback to one-by-one with retry
                new_embeddings = [
                    await self._embed_with_retry(t) for t in uncached_texts
                ]

            for idx, emb in zip(uncached_indices, new_embeddings):
                results[idx] = emb
                key = self._cache_key(texts[idx])
                if len(self._cache) >= self._cache_size:
                    self._cache.popitem(last=False)
                self._cache[key] = emb

        return results

    async def _embed_with_retry(self, text: str) -> List[float]:
        """Attempt embedding with exponential backoff retry and fallback."""
        provider = self._fallback if self._using_fallback else self._primary
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                return await provider.embed(text)
            except Exception as e:
                last_error = e
                wait_time = 2**attempt  # 1s, 2s, 4s
                logger.warning(
                    "Embedding attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1,
                    self._max_retries,
                    str(e),
                    wait_time,
                )
                await asyncio.sleep(wait_time)

        # All retries exhausted on primary — try fallback
        if not self._using_fallback and self._fallback is not None:
            logger.warning(
                "Primary embedding provider failed after %d retries. "
                "Switching to fallback provider.",
                self._max_retries,
            )
            self._using_fallback = True
            try:
                return await self._fallback.embed(text)
            except Exception as fallback_error:
                raise EmbeddingError(
                    f"Both primary and fallback embedding providers failed. "
                    f"Primary error: {last_error}. Fallback error: {fallback_error}"
                ) from fallback_error

        raise EmbeddingError(
            f"Embedding failed after {self._max_retries} retries: {last_error}"
        ) from last_error

    @property
    def dimensions(self) -> int:
        """Return dimensions of the active provider."""
        if self._using_fallback and self._fallback is not None:
            return self._fallback.dimensions
        return self._primary.dimensions

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Return cache statistics for monitoring."""
        return {
            "size": len(self._cache),
            "max_size": self._cache_size,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def reset_fallback(self) -> None:
        """Reset to primary provider (e.g., after connectivity is restored)."""
        self._using_fallback = False
        logger.info("Reset to primary embedding provider.")
