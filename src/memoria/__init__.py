"""
Memoria — Production-Grade Agent Memory System.

Storage-agnostic, self-evolving, observable memory infrastructure for AI agents.

Quick Start:
    from memoria import Memoria

    mem = Memoria()  # LanceDB embedded, zero-config
    mem.remember("User prefers Redis pool size of 10")
    context = mem.recall("help with database pools")
"""

from memoria.core.config import (
    AwarenessConfig,
    DecayConfig,
    FeedbackConfig,
    GraphConfig,
    MemoriaConfig,
)
from memoria.core.exceptions import (
    ConfigurationError,
    MemoriaError,
    MemoryNotFoundError,
    StorageError,
)
from memoria.core.memoria import Memoria
from memoria.core.models import (
    ContextInjection,
    Contradiction,
    ContradictionResolution,
    DecayCycleReport,
    InjectItem,
    MemoriaStats,
    MemoryEvent,
    MemoryLayer,
    MemoryRecord,
    MemoryType,
    SemanticFingerprint,
    StorageStats,
)

__version__ = "0.2.0"
__all__ = [
    "Memoria",
    "MemoryRecord",
    "MemoryType",
    "MemoryLayer",
    "MemoryEvent",
    "ContextInjection",
    "InjectItem",
    "Contradiction",
    "ContradictionResolution",
    "SemanticFingerprint",
    "DecayCycleReport",
    "MemoriaStats",
    "StorageStats",
    "MemoriaConfig",
    "AwarenessConfig",
    "DecayConfig",
    "FeedbackConfig",
    "GraphConfig",
    "MemoriaError",
    "ConfigurationError",
    "StorageError",
    "MemoryNotFoundError",
]
