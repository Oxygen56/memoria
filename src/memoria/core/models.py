"""Core data models for Memoria."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# ── Enums ────────────────────────────────────────


class MemoryType(str, Enum):
    """Classification of memory by its semantic nature."""

    FACT = "fact"  # "User's database is PostgreSQL 16"
    PREFERENCE = "preference"  # "User prefers concise responses"
    EVENT = "event"  # "Ran migration X on 2026-06-01"
    DECISION = "decision"  # "Chose PostgreSQL over MySQL for JSON support"
    RELATIONSHIP = "relationship"  # "Migration X depends on config Y"
    SKILL = "skill"  # "User knows how to use Docker Compose"
    CONSTRAINT = "constraint"  # "Never use port 5432 — it's reserved"


class MemoryLayer(str, Enum):
    """Storage tier for a memory based on access frequency and importance."""

    HOT = "hot"  # Injected into every context (~200 tokens budget)
    WARM = "warm"  # Retrieved on demand (semantic + keyword search)
    COLD = "cold"  # Archived, manually retrievable only
    OBLIVION = "oblivion"  # Scheduled for deletion


class ContradictionResolution(str, Enum):
    KEEP_NEW = "keep_new"
    KEEP_OLD = "keep_old"
    KEEP_BOTH = "keep_both"
    MERGE = "merge"


# ── Core Data Classes ────────────────────────────


@dataclass
class MemoryRecord:
    """The universal memory atom — a single fact the agent remembers."""

    # Identity
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")

    # Content
    content: str = ""
    memory_type: MemoryType = MemoryType.FACT
    layer: MemoryLayer = MemoryLayer.WARM

    # Lifecycle
    importance: float = 0.5  # 0.0–1.0
    decay_score: float = 1.0  # 1.0 → 0.0 over time
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Decay parameters
    half_life_days: float = 30.0
    decay_acceleration: float = 1.0

    # Relationships
    related_memories: list[str] = field(default_factory=list)
    source_conversation_id: str | None = None
    contradiction_of: str | None = None

    # Metadata
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    tags: list[str] = field(default_factory=list)
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    # Embedding (populated at insert time by embedding provider)
    embedding: list[float] | None = None

    # Audit
    created_by: str = "agent"
    version: int = 1

    @property
    def effective_importance(self) -> float:
        """Base importance + access-frequency bonus."""
        recency_bonus = min(0.3, self.access_count * 0.01)
        return min(1.0, self.importance + recency_bonus)

    @property
    def is_stale(self) -> bool:
        """True if not accessed in > 3 half-lives."""
        days = (datetime.now(timezone.utc) - self.last_accessed).days
        return days > (3 * self.half_life_days)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "layer": self.layer.value,
            "importance": self.importance,
            "decay_score": self.decay_score,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "last_modified": self.last_modified.isoformat(),
            "half_life_days": self.half_life_days,
            "decay_acceleration": self.decay_acceleration,
            "related_memories": self.related_memories,
            "source_conversation_id": self.source_conversation_id,
            "contradiction_of": self.contradiction_of,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "tags": self.tags,
            "custom_metadata": self.custom_metadata,
            "embedding": self.embedding,
            "created_by": self.created_by,
            "version": self.version,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryRecord:
        return cls(
            id=d.get("id", f"mem_{uuid.uuid4().hex[:12]}"),
            content=d.get("content", ""),
            memory_type=MemoryType(d["memory_type"]) if "memory_type" in d else MemoryType.FACT,
            layer=MemoryLayer(d["layer"]) if "layer" in d else MemoryLayer.WARM,
            importance=d.get("importance", 0.5),
            decay_score=d.get("decay_score", 1.0),
            created_at=_parse_datetime(d.get("created_at")),
            last_accessed=_parse_datetime(d.get("last_accessed")),
            access_count=d.get("access_count", 0),
            last_modified=_parse_datetime(d.get("last_modified")),
            half_life_days=d.get("half_life_days", 30.0),
            decay_acceleration=d.get("decay_acceleration", 1.0),
            related_memories=d.get("related_memories", []),
            source_conversation_id=d.get("source_conversation_id"),
            contradiction_of=d.get("contradiction_of"),
            user_id=d.get("user_id"),
            agent_id=d.get("agent_id"),
            session_id=d.get("session_id"),
            tags=d.get("tags", []),
            custom_metadata=d.get("custom_metadata", {}),
            embedding=d.get("embedding"),
            created_by=d.get("created_by", "agent"),
            version=d.get("version", 1),
        )


@dataclass
class MemoryEvent:
    """Emitted on every memory state change — powers the Dashboard WebSocket."""

    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    # created|accessed|modified|decayed|archived|forgotten|contradicted|reinforced
    event_type: str = ""
    memory_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    previous_state: dict[str, Any] | None = None
    new_state: dict[str, Any] | None = None
    trigger: str = "system"


@dataclass
class InjectItem:
    """A single memory injected into agent context."""

    memory: MemoryRecord
    relevance_score: float = 0.0


@dataclass
class ContextInjection:
    """Result of Awareness Engine: memories to inject into agent context."""

    hot: list[MemoryRecord] = field(default_factory=list)
    relevant: list[InjectItem] = field(default_factory=list)
    total_tokens: int = 0
    fingerprint_version: str = "v2"


@dataclass
class SemanticFingerprint:
    """Lightweight semantic signature of a conversation turn."""

    embedding: list[float] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)
    search_query: str = ""
    version: str = "v2"


@dataclass
class Contradiction:
    """A detected conflict between two memories."""

    new_memory_id: str = ""
    existing_memory_id: str = ""
    new_content: str = ""
    existing_content: str = ""
    resolution: ContradictionResolution = ContradictionResolution.KEEP_BOTH


@dataclass
class DecayCycleReport:
    """Result of a Decay Engine processing cycle."""

    events: list[MemoryEvent] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class StorageStats:
    """Statistics for a storage backend."""

    total_memories: int = 0
    by_layer: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    storage_size_bytes: int = 0
    backend_type: str = ""


@dataclass
class MemoriaStats:
    """Overall system statistics."""

    storage: StorageStats = field(default_factory=StorageStats)
    health_score: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class MemoryAccessLog:
    """Record of a single memory access event."""

    id: str = field(default_factory=lambda: f"acc_{uuid.uuid4().hex[:8]}")
    memory_id: str = ""
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: str = ""  # truncated to 200 chars
    session_id: str | None = None
    trigger: str = "awareness"  # awareness, explicit_search, manual


# ── Helpers ──────────────────────────────────────


def _parse_datetime(val: Any) -> datetime:
    if val is None:
        return datetime.now(timezone.utc)
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.fromisoformat(val)
    return datetime.now(timezone.utc)
