"""In-memory storage adapter — full-featured, zero-dependency implementation.

Used as:
- Default adapter when external storage (LanceDB, Qdrant, etc.) is not installed.
- Test/dev adapter for unit tests and local prototyping.
- Fallback when configured backend fails to initialize.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from memoria.core.exceptions import MemoryNotFoundError, StorageError
from memoria.core.models import MemoryLayer, MemoryRecord, StorageStats
from memoria.storage.base import MemoryStoreAdapter


class InMemoryAdapter(MemoryStoreAdapter):
    """Pure-Python in-memory implementation of MemoryStoreAdapter.

    Features:
    - Dict-based storage with O(1) get/insert/delete by ID.
    - Cosine similarity for semantic search.
    - Substring + word-boundary matching for keyword search.
    - Full implementation of all required and extended methods.

    Thread Safety:
        NOT thread-safe. For concurrent access, wrap with asyncio.Lock externally.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the in-memory store.

        Args:
            **kwargs: Ignored (accepted for compatibility with adapter factory).
        """
        self._store: Dict[str, MemoryRecord] = {}

    # ── Core (required) ──────────────────────────────

    async def insert(self, records: List[MemoryRecord]) -> List[str]:
        """Insert memories into the store. Returns list of inserted IDs.

        Raises:
            StorageError: If a record with the same ID already exists.
        """
        inserted_ids: List[str] = []
        for record in records:
            if record.id in self._store:
                raise StorageError(
                    f"Memory with ID '{record.id}' already exists. "
                    "Use update() to modify existing records."
                )
            self._store[record.id] = record
            inserted_ids.append(record.id)
        return inserted_ids

    async def search_semantic(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """Vector cosine similarity search against stored embeddings.

        Args:
            query_embedding: Pre-computed query vector.
            top_k: Maximum number of results.
            filters: Optional filter dict (supports 'layer.$in', 'user_id', 'tags').

        Returns:
            Top-k records sorted by descending cosine similarity.
        """
        candidates = self._apply_filters(filters)
        scored: List[tuple[float, MemoryRecord]] = []

        for record in candidates:
            if record.embedding is None:
                continue
            sim = self._cosine_similarity(query_embedding, record.embedding)
            scored.append((sim, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    async def search_keyword(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """Full-text keyword search using substring and word matching.

        Scoring:
        - Exact substring match: +2.0
        - Per-word match (word boundary): +1.0 per word
        - Tag match: +1.5 per matching tag

        Args:
            query: Search query string.
            top_k: Maximum number of results.
            filters: Optional filter dict.

        Returns:
            Top-k records sorted by descending keyword relevance score.
        """
        candidates = self._apply_filters(filters)
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored: List[tuple[float, MemoryRecord]] = []

        for record in candidates:
            score = 0.0
            content_lower = record.content.lower()

            # Exact substring match
            if query_lower in content_lower:
                score += 2.0

            # Per-word matching
            content_words = set(content_lower.split())
            matching_words = query_words & content_words
            score += len(matching_words) * 1.0

            # Tag matching
            if record.tags:
                for tag in record.tags:
                    if any(word in tag.lower() for word in query_words):
                        score += 1.5

            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Partial update a memory record.

        Args:
            memory_id: The ID of the memory to update.
            updates: Dict of field names to new values.

        Returns:
            True if the record was found and updated.

        Raises:
            MemoryNotFoundError: If the memory_id doesn't exist.
        """
        if memory_id not in self._store:
            raise MemoryNotFoundError(f"Memory '{memory_id}' not found.")

        record = self._store[memory_id]
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)
        return True

    async def delete(self, memory_ids: List[str]) -> int:
        """Delete memories by ID. Returns count of actually deleted records."""
        deleted = 0
        for mid in memory_ids:
            if mid in self._store:
                del self._store[mid]
                deleted += 1
        return deleted

    # ── Extended (recommended) ────────────────────────

    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Get a single memory by ID. Returns None if not found."""
        return self._store.get(memory_id)

    async def get_batch(self, memory_ids: List[str]) -> List[MemoryRecord]:
        """Get multiple memories by ID. Skips missing IDs."""
        return [self._store[mid] for mid in memory_ids if mid in self._store]

    async def list_by_layer(
        self, layer: MemoryLayer, limit: int = 1000, offset: int = 0
    ) -> List[MemoryRecord]:
        """List all memories in a specific layer with pagination."""
        matches = [r for r in self._store.values() if r.layer == layer]
        # Sort by created_at descending (newest first)
        matches.sort(key=lambda r: r.created_at, reverse=True)
        return matches[offset : offset + limit]

    async def get_stats(self) -> StorageStats:
        """Compute storage statistics."""
        by_layer: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for record in self._store.values():
            layer_key = record.layer.value
            by_layer[layer_key] = by_layer.get(layer_key, 0) + 1

            type_key = record.memory_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

        return StorageStats(
            total_memories=len(self._store),
            by_layer=by_layer,
            by_type=by_type,
            storage_size_bytes=0,  # In-memory — no disk usage
            backend_type="memory",
        )

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count memories matching optional filters."""
        if filters is None:
            return len(self._store)
        return len(self._apply_filters(filters))

    # ── Utility ──────────────────────────────────────

    def _apply_filters(self, filters: Optional[Dict[str, Any]]) -> List[MemoryRecord]:
        """Apply filter dict to the store and return matching records.

        Supported filter formats:
        - {"layer": {"$in": ["hot", "warm"]}}
        - {"user_id": "some-user-id"}
        - {"tags": {"$contains": "tag-value"}}
        """
        if not filters:
            return list(self._store.values())

        candidates = list(self._store.values())

        for key, condition in filters.items():
            if key == "layer":
                if isinstance(condition, dict) and "$in" in condition:
                    allowed_layers = set(condition["$in"])
                    candidates = [
                        r for r in candidates if r.layer.value in allowed_layers
                    ]
                elif isinstance(condition, str):
                    candidates = [r for r in candidates if r.layer.value == condition]

            elif key == "user_id":
                candidates = [r for r in candidates if r.user_id == condition]

            elif key == "agent_id":
                candidates = [r for r in candidates if r.agent_id == condition]

            elif key == "tags":
                if isinstance(condition, dict) and "$contains" in condition:
                    tag_value = condition["$contains"]
                    candidates = [
                        r for r in candidates if tag_value in (r.tags or [])
                    ]

            elif key == "memory_type":
                if isinstance(condition, dict) and "$in" in condition:
                    allowed_types = set(condition["$in"])
                    candidates = [
                        r for r in candidates if r.memory_type.value in allowed_types
                    ]
                elif isinstance(condition, str):
                    candidates = [
                        r for r in candidates if r.memory_type.value == condition
                    ]

        return candidates

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b:
            return 0.0

        # Handle dimension mismatch by truncating to shorter
        min_len = min(len(a), len(b))
        if len(a) != len(b):
            a = a[:min_len]
            b = b[:min_len]

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)
