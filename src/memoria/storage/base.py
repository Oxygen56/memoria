"""Abstract storage adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from memoria.core.models import MemoryLayer, MemoryRecord, StorageStats


class MemoryStoreAdapter(ABC):
    """
    Implement this interface to add a new storage backend to Memoria.

    Core (required): insert, search_semantic, search_keyword, update, delete
    Extended (recommended): search_hybrid, get, get_batch, list_by_layer, get_stats
    """

    # ── Core (required) ──────────────────────────────

    @abstractmethod
    async def insert(self, records: list[MemoryRecord]) -> list[str]:
        """Insert memories. Returns list of inserted IDs."""
        ...

    @abstractmethod
    async def search_semantic(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        """Vector similarity search. `query_embedding` is pre-computed by the caller."""
        ...

    @abstractmethod
    async def search_keyword(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        """Full-text / keyword search."""
        ...

    @abstractmethod
    async def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Partial update a memory. Returns True if successful."""
        ...

    @abstractmethod
    async def delete(self, memory_ids: list[str]) -> int:
        """Delete memories. Returns count deleted."""
        ...

    # ── Extended (recommended) ────────────────────────

    async def search_hybrid(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        semantic_weight: float = 0.7,
        filters: dict[str, Any] | None = None,
        rrf_k: int = 60,
    ) -> list[MemoryRecord]:
        """Hybrid semantic + keyword search with RRF fusion.

        Args:
            query: Text query for keyword search.
            query_embedding: Pre-computed embedding for semantic search.
            top_k: Maximum results to return.
            semantic_weight: Weight for semantic results in hybrid scoring (0.0–1.0).
            filters: Optional filters to apply.
            rrf_k: RRF fusion parameter (higher = more uniform blending).
        """
        semantic_results = await self.search_semantic(query_embedding, top_k * 2, filters)
        keyword_results = await self.search_keyword(query, top_k * 2, filters)
        return self._rrf_fusion(semantic_results, keyword_results, top_k, k=rrf_k)

    async def get(self, memory_id: str) -> MemoryRecord | None:
        """Get a single memory by ID."""
        raise NotImplementedError

    async def get_batch(self, memory_ids: list[str]) -> list[MemoryRecord]:
        """Get multiple memories by ID."""
        results = []
        for mid in memory_ids:
            record = await self.get(mid)
            if record:
                results.append(record)
        return results

    async def list_by_layer(
        self, layer: MemoryLayer, limit: int = 1000, offset: int = 0
    ) -> list[MemoryRecord]:
        """List memories in a specific layer."""
        raise NotImplementedError

    async def get_stats(self) -> StorageStats:
        """Storage statistics."""
        raise NotImplementedError

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count memories matching filters."""
        raise NotImplementedError

    # ── Utility ──────────────────────────────────────

    @staticmethod
    def _rrf_fusion(
        list_a: list[MemoryRecord],
        list_b: list[MemoryRecord],
        top_k: int,
        k: int = 60,
    ) -> list[MemoryRecord]:
        """Reciprocal Rank Fusion — combines two ranked lists."""
        scores: dict[str, float] = {}
        id_to_item: dict[str, MemoryRecord] = {}

        for rank, item in enumerate(list_a):
            scores[item.id] = scores.get(item.id, 0) + 1.0 / (k + rank)
            id_to_item[item.id] = item

        for rank, item in enumerate(list_b):
            scores[item.id] = scores.get(item.id, 0) + 1.0 / (k + rank)
            id_to_item[item.id] = item

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [id_to_item[mid] for mid in sorted_ids if mid in id_to_item]
