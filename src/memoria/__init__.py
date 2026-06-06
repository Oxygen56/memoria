"""
Memoria — Production-Grade Agent Memory System.

Storage-agnostic, self-evolving, observable memory infrastructure for AI agents.

Quick Start:
    from memoria import Memoria

    mem = Memoria()  # LanceDB embedded, zero-config
    mem.remember("User prefers Redis pool size of 10")
    context = mem.recall("help with database pools")
"""

from memoria.core.models import (
    MemoryRecord,
    MemoryType,
    MemoryLayer,
    MemoryEvent,
    ContextInjection,
    InjectItem,
    Contradiction,
    ContradictionResolution,
    SemanticFingerprint,
    DecayCycleReport,
    MemoriaStats,
    StorageStats,
)
from memoria.core.memoria import Memoria
from memoria.core.config import MemoriaConfig, AwarenessConfig, DecayConfig, FeedbackConfig, GraphConfig
from memoria.core.exceptions import MemoriaError, ConfigurationError, StorageError, MemoryNotFoundError

__version__ = "0.1.0"
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
