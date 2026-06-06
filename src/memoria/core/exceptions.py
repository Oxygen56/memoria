"""Custom exceptions for Memoria."""


class MemoriaError(Exception):
    """Base exception for all Memoria errors."""

    pass


class ConfigurationError(MemoriaError):
    """Raised when configuration is invalid."""

    pass


class StorageError(MemoriaError):
    """Raised when storage operation fails."""

    pass


class MemoryNotFoundError(MemoriaError):
    """Raised when a memory ID doesn't exist."""

    pass


class EmbeddingError(MemoriaError):
    """Raised when embedding generation fails."""

    pass


class AdapterNotFoundError(MemoriaError):
    """Raised when a storage adapter type is unknown."""

    pass


class GraphError(MemoriaError):
    """Raised when graph operations fail."""

    pass


class ContradictionError(MemoriaError):
    """Raised when contradiction detection fails."""

    pass
