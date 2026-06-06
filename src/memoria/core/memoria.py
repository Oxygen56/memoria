"""
Memoria — main entry point.

Orchestrates all four engines and the storage layer behind a clean,
framework-agnostic API.

Quick Start:
    from memoria import Memoria

    async with Memoria() as mem:
        await mem.remember("User prefers Redis pool size of 10")
        ctx = await mem.recall("help with database pools")
        print(ctx.hot)       # Always-injected memories
        print(ctx.relevant)  # Relevance-matched memories
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

from memoria.core.config import (
    AwarenessConfig,
    DecayConfig,
    FeedbackConfig,
    GraphConfig,
    MemoriaConfig,
    StorageBackendConfig,
)
from memoria.core.exceptions import ConfigurationError
from memoria.core.models import (
    ContextInjection,
    Contradiction,
    ContradictionResolution,
    DecayCycleReport,
    MemoriaStats,
    MemoryLayer,
    MemoryRecord,
    MemoryType,
    StorageStats,
)
from memoria.embedding.base import EmbeddingProvider, OpenAIEmbeddingProvider
from memoria.engines.awareness import AwarenessEngine
from memoria.engines.decay import DecayEngine
from memoria.engines.feedback import FeedbackEngine
from memoria.engines.orchestrator import EngineOrchestrator
from memoria.storage.base import MemoryStoreAdapter
from memoria.storage.router import StorageRouter


class Memoria:
    """
    Production-grade memory infrastructure for AI agents.

    Storage-agnostic, self-evolving, observable.

    Usage:
        async with Memoria() as mem:
            await mem.remember("User uses PostgreSQL 16")
            context = await mem.recall("database config")
            results = await mem.search("PostgreSQL")
    """

    def __init__(
        self,
        # Storage configuration
        warm_backend: Optional[Union[str, MemoryStoreAdapter]] = None,
        hot_backend: Optional[Union[str, MemoryStoreAdapter]] = None,
        cold_backend: Optional[Union[str, MemoryStoreAdapter]] = None,
        graph_backend: Optional[Union[str, MemoryStoreAdapter]] = None,
        # Embedding
        embedding: Optional[Union[str, EmbeddingProvider]] = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: Optional[int] = None,
        # Engine configs
        awareness_config: Optional[AwarenessConfig] = None,
        decay_config: Optional[DecayConfig] = None,
        feedback_config: Optional[FeedbackConfig] = None,
        graph_config: Optional[GraphConfig] = None,
        # Convenience
        config: Optional[MemoriaConfig] = None,
        config_path: Optional[str] = None,
        data_dir: str = "~/.memoria",
    ):
        # Build or load configuration
        if config:
            self._config = config
        elif config_path:
            self._config = self._load_config_from_file(config_path)
        else:
            self._config = self._build_config(
                warm_backend=warm_backend,
                hot_backend=hot_backend,
                cold_backend=cold_backend,
                graph_backend=graph_backend,
                awareness_config=awareness_config,
                decay_config=decay_config,
                feedback_config=feedback_config,
                graph_config=graph_config,
                data_dir=data_dir,
            )

        self._data_dir = os.path.expanduser(data_dir)
        os.makedirs(self._data_dir, exist_ok=True)

        # Storage layer
        self._storage = StorageRouter(self._config)

        # Embedding provider
        if isinstance(embedding, EmbeddingProvider):
            self._embedding = embedding
        elif isinstance(embedding, str) and embedding == "local":
            from memoria.embedding.base import LocalEmbeddingProvider
            self._embedding = LocalEmbeddingProvider()
        else:
            self._embedding = OpenAIEmbeddingProvider(
                model=embedding_model,
                dimensions=embedding_dimensions,
            )

        # Engines (initialized after storage is ready)
        self._awareness: Optional[AwarenessEngine] = None
        self._decay: Optional[DecayEngine] = None
        self._feedback: Optional[FeedbackEngine] = None
        self._graph: Optional["GraphEngine"] = None
        self._orchestrator: Optional[EngineOrchestrator] = None

        self._started_at: Optional[datetime] = None

    # ── Lifecycle ────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize storage and engines. Called automatically by context manager."""
        await self._storage.initialize()

        self._awareness = AwarenessEngine(
            storage=self._storage,
            embedding_provider=self._embedding,
            config=self._config.awareness,
        )
        self._decay = DecayEngine(
            storage=self._storage,
            config=self._config.decay,
        )
        self._feedback = FeedbackEngine(
            storage=self._storage,
            config=self._config.feedback,
        )

        from memoria.engines.graph import GraphEngine
        self._graph = GraphEngine(config=self._config.graph)

        # Engine orchestrator
        self._orchestrator = EngineOrchestrator(
            awareness=self._awareness,
            decay=self._decay,
            feedback=self._feedback,
            graph=self._graph,
        )

        self._started_at = datetime.now(timezone.utc)

    async def close(self) -> None:
        """Gracefully close all connections."""
        self._started_at = None

    async def __aenter__(self) -> "Memoria":
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ── Core API: Remember ───────────────────────────

    async def remember(
        self,
        content: str,
        memory_type: Union[MemoryType, str] = MemoryType.FACT,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        related_to: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> MemoryRecord:
        """
        Store a new memory.

        Args:
            content: The memory text (e.g., "User prefers Redis pool size of 10").
            memory_type: Classification: FACT, PREFERENCE, EVENT, DECISION, SKILL, etc.
            importance: 0.0–1.0. Higher = resists decay longer. Default 0.5.
            tags: Searchable labels (e.g., ["database", "redis", "config"]).
            metadata: Arbitrary key-value pairs.
            related_to: IDs of related memories.
            user_id: Optional user identifier for multi-tenant setups.
            session_id: Optional session identifier.

        Returns:
            The created MemoryRecord.
        """
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        # Determine initial layer based on importance
        initial_layer = MemoryLayer.WARM
        if importance >= 0.85 or memory_type in (MemoryType.PREFERENCE, MemoryType.CONSTRAINT):
            initial_layer = MemoryLayer.HOT

        # Generate embedding
        embedding = await self._embedding.embed(content)

        record = MemoryRecord(
            content=content,
            memory_type=memory_type,
            layer=initial_layer,
            importance=importance,
            tags=tags or [],
            custom_metadata=metadata or {},
            related_memories=related_to or [],
            user_id=user_id,
            session_id=session_id,
            embedding=embedding,
            created_by="user" if user_id else "agent",
        )

        # Persist
        await self._storage.insert([record])

        # Check for contradictions
        if self._feedback:
            await self._feedback.detect_contradictions(record)

        # Build graph relations
        if self._graph:
            await self._graph.build_relations(record)

        return record

    # ── Core API: Recall ─────────────────────────────

    async def recall(
        self,
        input_text: str,
        token_budget: int = 200,
        context_window: Optional[List[str]] = None,
    ) -> ContextInjection:
        """
        Proactively recall relevant memories for the current conversation turn.

        Uses the Awareness Engine to compute a semantic fingerprint and
        inject relevant memories without the agent needing to explicitly search.

        Call this at the start of every agent turn.

        Args:
            input_text: Current user message or agent task.
            token_budget: Max tokens to inject. Default 200.
            context_window: Recent conversation turns for context enrichment.

        Returns:
            ContextInjection with hot memories (always present) and
            relevant memories (ranked by relevance × decay × importance).
        """
        if self._awareness is None:
            raise ConfigurationError("Memoria not initialized. Use `async with Memoria() as mem:`")

        result = await self._awareness.get_context(
            input_text=input_text,
            context_window=context_window,
            token_budget=token_budget,
        )

        # Post-process with orchestrator (filter contradictions, boost graph-connected)
        if self._orchestrator:
            result = await self._orchestrator.post_process_context(result, input_text)

        return result

    # ── Core API: Search ─────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 10,
        memory_type: Optional[Union[MemoryType, str]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[MemoryRecord]:
        """
        Explicit memory search — when the agent knows what it's looking for.

        Performs hybrid semantic + keyword search.

        Args:
            query: Search query text.
            top_k: Max results to return.
            memory_type: Optional filter by memory type.
            tags: Optional filter by tags.

        Returns:
            Ranked list of matching MemoryRecords.
        """
        query_embedding = await self._embedding.embed(query)

        filters: Dict[str, Any] = {}
        if memory_type:
            filters["memory_type"] = memory_type.value if isinstance(memory_type, MemoryType) else memory_type

        results = await self._storage.search_hybrid(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters if filters else None,
        )

        return results

    # ── Core API: Mutations ──────────────────────────

    async def forget(self, memory_id: str) -> bool:
        """Manually delete a memory. Returns True if deleted."""
        count = await self._storage.delete([memory_id])
        return count > 0

    async def reinforce(self, memory_id: str, amount: float = 0.1) -> bool:
        """Manually boost a memory's importance."""
        record = await self._storage.get(memory_id)
        if record is None:
            return False
        new_importance = min(1.0, record.importance + amount)
        return await self._storage.update(memory_id, {"importance": new_importance})

    async def edit(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        memory_type: Optional[Union[MemoryType, str]] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> bool:
        """Manually edit a memory's fields."""
        updates: Dict[str, Any] = {}
        if content is not None:
            updates["content"] = content
            # Re-embed if content changed
            updates["embedding"] = await self._embedding.embed(content)
        if importance is not None:
            updates["importance"] = max(0.0, min(1.0, importance))
        if memory_type is not None:
            updates["memory_type"] = (
                memory_type.value if isinstance(memory_type, MemoryType) else memory_type
            )
        if tags is not None:
            updates["tags"] = tags

        updates.update(kwargs)
        updates["last_modified"] = datetime.now(timezone.utc).isoformat()

        return await self._storage.update(memory_id, updates)

    # ── Introspection ────────────────────────────────

    async def stats(self) -> MemoriaStats:
        """Get overall system statistics."""
        storage_stats = await self._storage.get_stats()
        warm_stats = storage_stats.get("warm", StorageStats(backend_type="lancedb"))

        # Compute a simple health score
        health = 100.0
        if warm_stats.total_memories > 0:
            oblivion_count = warm_stats.by_layer.get("oblivion", 0)
            if oblivion_count > warm_stats.total_memories * 0.1:
                health -= 20  # Too many decaying memories

        uptime = 0.0
        if self._started_at:
            uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()

        return MemoriaStats(
            storage=warm_stats,
            health_score=health,
            uptime_seconds=uptime,
        )

    async def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """Get a single memory by ID."""
        return await self._storage.get(memory_id)

    async def list_memories(
        self,
        layer: Optional[Union[MemoryLayer, str]] = None,
        memory_type: Optional[Union[MemoryType, str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """List memories with optional filtering."""
        if layer:
            if isinstance(layer, str):
                layer = MemoryLayer(layer)
            return await self._storage.list_by_layer(layer, limit, offset)
        # Without layer, search across warm layer
        return await self._storage.list_by_layer(MemoryLayer.WARM, limit, offset)

    async def search_by_tag(self, tag: str, limit: int = 20) -> List[MemoryRecord]:
        """Find all memories with a specific tag."""
        # Simple approach: use keyword search on the tag
        return await self._storage.search_keyword(query=tag, top_k=limit)

    # ── Contradictions ───────────────────────────────

    # ── Feedback Loop ────────────────────────────────

    async def report_usage(
        self,
        context_injection: ContextInjection,
        used_memory_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Report which memories from the last recall were actually used by the agent.

        Call this after each agent turn to close the feedback loop.
        Memories that were injected but not used get penalized (decay accelerates).
        Memories that were used get reinforced (importance boosted).

        Args:
            context_injection: The ContextInjection returned by the last recall().
            used_memory_ids: IDs of memories the agent actually referenced/used.
                            If None, all injected memories are considered used.
        """
        if self._feedback is None:
            return

        # Collect all injected memory IDs
        all_injected_ids: Set[str] = set()
        for item in context_injection.relevant:
            all_injected_ids.add(item.memory.id)
        for mem in context_injection.hot:
            all_injected_ids.add(mem.id)

        if used_memory_ids is None:
            # If not specified, assume all were used (no penalty)
            used_set = all_injected_ids
        else:
            used_set = set(used_memory_ids)

        # Reinforce used memories
        for mem_id in used_set & all_injected_ids:
            await self._feedback.on_memory_accessed(mem_id, context="auto_report")

        # Penalize ignored memories
        ignored = all_injected_ids - used_set
        for mem_id in ignored:
            await self._feedback.on_memory_ignored(mem_id)

    # ── Graph Queries ────────────────────────────────

    async def query_graph(self, entity_name: str, hops: int = 3):
        """Query the knowledge graph for related entities and memories."""
        if self._graph is None:
            return None
        return await self._graph.query_graph(entity_name, hops)

    async def get_related_memories(self, memory_id: str, max_hops: int = 2) -> List[str]:
        """Get memory IDs related through the knowledge graph."""
        if self._graph is None:
            return []
        return await self._graph.get_related_memories(memory_id, max_hops)

    async def get_contradictions(self) -> List[Contradiction]:
        """Get all flagged contradictions for manual review."""
        # Find memories with contradiction_of set
        results = []
        warm_memories = await self._storage.list_by_layer(
            MemoryLayer.WARM, limit=1000
        )
        for mem in warm_memories:
            if mem.contradiction_of:
                other = await self._storage.get(mem.contradiction_of)
                if other:
                    results.append(Contradiction(
                        new_memory_id=mem.id,
                        existing_memory_id=other.id,
                        new_content=mem.content,
                        existing_content=other.content,
                    ))
        return results

    async def resolve_contradiction(
        self,
        contradiction: Contradiction,
        resolution: ContradictionResolution,
    ) -> bool:
        """Resolve a contradiction."""
        if resolution == ContradictionResolution.KEEP_NEW:
            await self._storage.delete([contradiction.existing_memory_id])
        elif resolution == ContradictionResolution.KEEP_OLD:
            await self._storage.delete([contradiction.new_memory_id])
        elif resolution == ContradictionResolution.MERGE:
            # Simple merge: keep both, add cross-reference
            await self._storage.update(contradiction.new_memory_id, {
                "related_memories": [contradiction.existing_memory_id],
            })
        # KEEP_BOTH: do nothing, flags stay
        return True

    # ── Maintenance ──────────────────────────────────

    async def run_decay_cycle(self) -> DecayCycleReport:
        """Manually trigger a decay processing cycle."""
        if self._decay is None:
            raise ConfigurationError("Memoria not initialized.")
        return await self._decay.process_decay_cycle()

    # ── Configuration Helpers ────────────────────────

    def _build_config(
        self,
        warm_backend: Any = None,
        hot_backend: Any = None,
        cold_backend: Any = None,
        graph_backend: Any = None,
        awareness_config: Optional[AwarenessConfig] = None,
        decay_config: Optional[DecayConfig] = None,
        feedback_config: Optional[FeedbackConfig] = None,
        graph_config: Optional[GraphConfig] = None,
        data_dir: str = "~/.memoria",
    ) -> MemoriaConfig:
        """Build MemoriaConfig from constructor arguments."""

        def _to_backend_config(backend: Any) -> Optional[StorageBackendConfig]:
            if backend is None:
                return None
            if isinstance(backend, MemoryStoreAdapter):
                return StorageBackendConfig(type="memory")  # fallback to in-memory
            if isinstance(backend, str):
                if "://" in backend:
                    scheme, rest = backend.split("://", 1)
                    return StorageBackendConfig(type=scheme, params={"url": f"{scheme}://{rest}"})
                return StorageBackendConfig(type=backend)
            if isinstance(backend, StorageBackendConfig):
                return backend
            return None

        warm_cfg = _to_backend_config(warm_backend)
        if warm_cfg is None:
            # Try LanceDB first; fall back to in-memory
            try:
                import lancedb  # noqa: F401
                warm_cfg = StorageBackendConfig(
                    type="lancedb",
                    params={"path": f"{data_dir}/data"},
                )
            except ImportError:
                warm_cfg = StorageBackendConfig(type="memory")

        return MemoriaConfig(
            warm_backend=warm_cfg,
            hot_backend=_to_backend_config(hot_backend),
            cold_backend=_to_backend_config(cold_backend),
            graph_backend=_to_backend_config(graph_backend),
            awareness=awareness_config or AwarenessConfig(),
            decay=decay_config or DecayConfig(),
            feedback=feedback_config or FeedbackConfig(),
            graph=graph_config or GraphConfig(),
            data_dir=data_dir,
        )

    @staticmethod
    def _load_config_from_file(path: str) -> MemoriaConfig:
        """Load configuration from a YAML file."""
        import yaml

        with open(os.path.expanduser(path)) as f:
            data = yaml.safe_load(f)

        # Build config from YAML data
        storage = data.get("storage", {})
        awareness = data.get("awareness", {})
        decay = data.get("decay", {})
        feedback = data.get("feedback", {})
        graph = data.get("graph", {})

        return MemoriaConfig(
            warm_backend=StorageBackendConfig(**storage["warm_backend"])
            if "warm_backend" in storage else StorageBackendConfig(type="lancedb"),
            hot_backend=StorageBackendConfig(**storage["hot_backend"])
            if "hot_backend" in storage else None,
            cold_backend=StorageBackendConfig(**storage["cold_backend"])
            if "cold_backend" in storage else None,
            graph_backend=StorageBackendConfig(**storage["graph_backend"])
            if "graph_backend" in storage else None,
            awareness=AwarenessConfig(**awareness) if awareness else AwarenessConfig(),
            decay=DecayConfig(**decay) if decay else DecayConfig(),
            feedback=FeedbackConfig(**feedback) if feedback else FeedbackConfig(),
            graph=GraphConfig(**graph) if graph else GraphConfig(),
        )
