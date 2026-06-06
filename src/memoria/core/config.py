"""Configuration models for Memoria."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AwarenessConfig:
    """Configuration for the Awareness Engine."""

    relevance_threshold: float = 0.35
    max_candidates: int = 50
    token_budget_default: int = 200
    token_budget_max: int = 500
    context_window_size: int = 3
    fingerprint_version: str = "v2"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_fallback: bool = True
    semantic_weight: float = 0.50
    keyword_weight: float = 0.25
    recency_weight: float = 0.15
    tag_weight: float = 0.10

    # Hybrid search parameters
    rrf_k: int = 60  # RRF fusion k parameter (higher = more uniform blending)
    hybrid_semantic_ratio: float = 0.7  # Semantic weight in hybrid search (0.0–1.0)


@dataclass
class DecayConfig:
    """Configuration for the Decay Engine."""

    decay_cycle_interval_hours: int = 6
    consolidate_before_deletion: bool = True
    max_memories_per_cycle: int = 10000
    custom_half_lives: Dict[str, float] = field(default_factory=dict)
    auto_promote_to_hot_threshold: float = 0.85

    # Layer thresholds
    warm_to_cold_threshold: float = 0.3
    cold_to_oblivion_threshold: float = 0.05

    # Adaptive decay interval configuration
    adaptive_interval: bool = True  # Whether to enable adaptive interval scheduling
    min_interval_hours: float = 1.0  # Minimum decay cycle interval
    max_interval_hours: float = 24.0  # Maximum decay cycle interval
    background_enabled: bool = False  # Whether to enable background auto-decay


@dataclass
class FeedbackConfig:
    """Configuration for the Feedback Engine."""

    boost_per_access: float = 0.05
    max_boost: float = 0.5
    disuse_penalty_rate: float = 0.1
    contradiction_detection: bool = True
    contradiction_llm_threshold: float = 0.7


@dataclass
class GraphConfig:
    """Configuration for the Graph Engine."""

    use_llm_extraction: bool = False
    max_hops_default: int = 3
    max_graph_size: int = 100000
    persist_interval_minutes: int = 15


@dataclass
class StorageBackendConfig:
    """Configuration for a single storage backend."""

    type: str = "lancedb"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoriaConfig:
    """Master configuration for Memoria."""

    # Storage
    warm_backend: Optional[StorageBackendConfig] = None
    hot_backend: Optional[StorageBackendConfig] = None
    cold_backend: Optional[StorageBackendConfig] = None
    graph_backend: Optional[StorageBackendConfig] = None

    # Embedding
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 20
    embedding_fallback_provider: str = "local_onnx"
    embedding_fallback_model: str = "BAAI/bge-small-en-v1.5"

    # Engines
    awareness: AwarenessConfig = field(default_factory=AwarenessConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    # Data directory
    data_dir: str = "~/.memoria"

    @classmethod
    def default(cls) -> "MemoriaConfig":
        return cls(
            warm_backend=StorageBackendConfig(
                type="lancedb",
                params={"path": "~/.memoria/data"},
            ),
        )
