"""Storage router — routes memory operations to configured backends by layer."""

from __future__ import annotations

from typing import Any

from memoria.core.config import MemoriaConfig, StorageBackendConfig
from memoria.core.exceptions import ConfigurationError
from memoria.core.models import MemoryLayer, MemoryRecord, StorageStats
from memoria.storage.base import MemoryStoreAdapter


class AdapterRegistry:
    """Registry of available storage adapter types."""

    _adapters: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: type) -> None:
        cls._adapters[name] = adapter_cls

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._adapters:
            if name == "lancedb":
                try:
                    from memoria.storage.lancedb_adapter import LanceDBAdapter
                    cls.register("lancedb", LanceDBAdapter)
                    return LanceDBAdapter
                except ImportError:
                    # LanceDB not installed; fall back to in-memory
                    from memoria.storage.memory_adapter import InMemoryAdapter
                    cls.register("memory", InMemoryAdapter)
                    return InMemoryAdapter
            elif name == "memory":
                from memoria.storage.memory_adapter import InMemoryAdapter
                cls.register("memory", InMemoryAdapter)
                return InMemoryAdapter
            elif name == "file":
                from memoria.storage.file_adapter import FileAdapter
                cls.register("file", FileAdapter)
                return FileAdapter
            elif name == "pgvector":
                from memoria.storage.pgvector_adapter import PgVectorAdapter
                cls.register("pgvector", PgVectorAdapter)
                return PgVectorAdapter
            raise ConfigurationError(
                f"Unknown adapter type: {name}. "
                f"Available: {list(cls._adapters.keys())}. "
                f"Install with: pip install memoria[{name}]"
            )
        return cls._adapters[name]


class StorageRouter:
    """
    Routes memory operations to the appropriate backend.

    Supports multi-backend configurations:
    - Hot layer → Redis (sub-ms cache)
    - Warm layer → LanceDB (default) or Qdrant (production)
    - Cold layer → File archive
    - Graph layer → Neo4j (optional)
    """

    def __init__(self, config: MemoriaConfig):
        self._config = config
        self._backends: dict[str, MemoryStoreAdapter] = {}
        self._initialized = False

    # ── Initialization ───────────────────────────────

    async def initialize(self) -> None:
        if self._initialized:
            return

        # Map layer → backend config
        layer_configs = {
            "hot": self._config.hot_backend,
            "warm": self._config.warm_backend or StorageBackendConfig(type="memory"),
            "cold": self._config.cold_backend,
            "graph": self._config.graph_backend,
        }

        for name, backend_cfg in layer_configs.items():
            if backend_cfg is None:
                continue
            try:
                adapter_cls = AdapterRegistry.get(backend_cfg.type)
                self._backends[name] = adapter_cls(**backend_cfg.params)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to initialize {name} backend ({backend_cfg.type}): {e}"
                ) from e

        self._initialized = True

    # ── Routing ──────────────────────────────────────

    def _route(self, layer: MemoryLayer) -> MemoryStoreAdapter:
        """Route to the correct backend based on memory layer."""
        routing = {
            MemoryLayer.HOT: self._backends.get("hot", self._backends.get("warm")),
            MemoryLayer.WARM: self._backends.get("warm"),
            MemoryLayer.COLD: self._backends.get("cold", self._backends.get("warm")),
            MemoryLayer.OBLIVION: self._backends.get("cold", self._backends.get("warm")),
        }
        backend = routing.get(layer)
        if backend is None:
            raise ConfigurationError(
                f"No backend configured for layer {layer.value}. "
                f"Ensure at least a warm_backend is set."
            )
        return backend

    @property
    def warm_backend(self) -> MemoryStoreAdapter:
        backend = self._backends.get("warm")
        if backend is None:
            raise ConfigurationError("No warm backend configured.")
        return backend

    # ── Operations ───────────────────────────────────

    async def insert(self, records: list[MemoryRecord]) -> list[str]:
        """Route inserts to the correct backend(s) based on each record's layer."""
        by_layer: dict[MemoryLayer, list[MemoryRecord]] = {}
        for record in records:
            by_layer.setdefault(record.layer, []).append(record)

        all_ids = []
        for layer, batch in by_layer.items():
            backend = self._route(layer)
            ids = await backend.insert(batch)
            all_ids.extend(ids)
        return all_ids

    async def search_hybrid(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        semantic_weight: float = 0.7,
        filters: dict[str, Any] | None = None,
        rrf_k: int = 60,
    ) -> list[MemoryRecord]:
        """Search across HOT + WARM layers, deduplicate, re-rank."""
        results: list[MemoryRecord] = []
        seen_ids: set = set()

        for layer in [MemoryLayer.HOT, MemoryLayer.WARM]:
            backend = self._route(layer)
            try:
                layer_results = await backend.search_hybrid(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k * 2,
                    semantic_weight=semantic_weight,
                    filters=filters,
                )
                for rec in layer_results:
                    if rec.id not in seen_ids:
                        seen_ids.add(rec.id)
                        results.append(rec)
            except NotImplementedError:
                # Backend doesn't support hybrid; try semantic only
                try:
                    layer_results = await backend.search_semantic(
                        query_embedding, top_k * 2, filters
                    )
                    for rec in layer_results:
                        if rec.id not in seen_ids:
                            seen_ids.add(rec.id)
                            results.append(rec)
                except Exception:
                    pass

        # Rank: HOT memories first, then by decay_score × importance
        results.sort(
            key=lambda r: (
                0 if r.layer == MemoryLayer.HOT else 1,
                -(r.decay_score * r.effective_importance),
            )
        )
        return results[:top_k]

    # ── Passthrough Methods (delegate to warm backend) ──

    async def search_semantic(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        """Vector similarity search, delegated to warm backend."""
        return await self.warm_backend.search_semantic(query_embedding, top_k, filters)

    async def search_keyword(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        """Keyword search, delegated to warm backend."""
        return await self.warm_backend.search_keyword(query, top_k, filters)

    async def get(self, memory_id: str) -> MemoryRecord | None:
        """Try to find a memory across all layers."""
        for layer in [MemoryLayer.HOT, MemoryLayer.WARM, MemoryLayer.COLD]:
            backend = self._route(layer)
            try:
                rec = await backend.get(memory_id)
                if rec:
                    return rec
            except NotImplementedError:
                continue
        return None

    async def get_batch(self, memory_ids: list[str]) -> list[MemoryRecord]:
        results = []
        for mid in memory_ids:
            rec = await self.get(mid)
            if rec:
                results.append(rec)
        return results

    async def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Try to update a memory — searches across layers to find it."""
        for layer in [MemoryLayer.HOT, MemoryLayer.WARM, MemoryLayer.COLD]:
            backend = self._route(layer)
            try:
                if await backend.update(memory_id, updates):
                    return True
            except NotImplementedError:
                continue
        return False

    async def delete(self, memory_ids: list[str]) -> int:
        count = 0
        for layer in [MemoryLayer.HOT, MemoryLayer.WARM, MemoryLayer.COLD]:
            backend = self._route(layer)
            try:
                count += await backend.delete(memory_ids)
            except NotImplementedError:
                continue
        return count

    async def list_by_layer(
        self, layer: MemoryLayer, limit: int = 1000, offset: int = 0
    ) -> list[MemoryRecord]:
        backend = self._route(layer)
        return await backend.list_by_layer(layer, limit, offset)

    async def get_stats(self) -> dict[str, StorageStats]:
        stats = {}
        for name, backend in self._backends.items():
            try:
                stats[name] = await backend.get_stats()
            except NotImplementedError:
                pass
        return stats

    async def count(
        self, filters: dict[str, Any] | None = None, layer: MemoryLayer | None = None
    ) -> int:
        if layer:
            backend = self._route(layer)
            return await backend.count(filters)
        total = 0
        for backend in self._backends.values():
            try:
                total += await backend.count(filters)
            except NotImplementedError:
                pass
        return total
