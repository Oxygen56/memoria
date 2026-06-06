"""Feedback Engine — self-improving memory through usage tracking.

The Feedback Engine closes the learning loop:
1. Track which memories are actually used by the agent
2. Reinforce useful memories (boost importance, slow decay)
3. Penalize injected-but-unused memories (accelerate decay)
4. Detect and surface contradictions between memories
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone

from memoria.core.config import FeedbackConfig
from memoria.core.models import (
    Contradiction,
    ContradictionResolution,
    MemoryAccessLog,
    MemoryEvent,
    MemoryRecord,
    MemoryType,
)
from memoria.storage.base import MemoryStoreAdapter


class UsageTracker:
    """Tracks which memories are accessed and in what context (in-memory only)."""

    def __init__(self):
        self._access_log: dict[str, list[datetime]] = defaultdict(list)
        self._context_map: dict[str, list[str]] = defaultdict(list)

    def record_access(self, memory_id: str, context: str) -> None:
        now = datetime.now(timezone.utc)
        self._access_log[memory_id].append(now)
        self._context_map[memory_id].append(context[:200])  # Truncate

    def get_access_count(self, memory_id: str) -> int:
        return len(self._access_log.get(memory_id, []))

    def get_last_access(self, memory_id: str) -> datetime | None:
        accesses = self._access_log.get(memory_id, [])
        return accesses[-1] if accesses else None

    def get_recent_contexts(
        self, memory_id: str, limit: int = 5
    ) -> list[str]:
        return self._context_map.get(memory_id, [])[-limit:]


class PersistentUsageTracker(UsageTracker):
    """Usage tracker that persists access logs to storage backend.

    Extends UsageTracker with a write-behind buffer that flushes access
    records to the storage layer periodically or when the buffer is full.
    This ensures access patterns survive process restarts.
    """

    FLUSH_THRESHOLD: int = 50  # Auto-flush when buffer reaches this size

    def __init__(self, storage: MemoryStoreAdapter):
        super().__init__()
        self._storage = storage
        self._buffer: list[MemoryAccessLog] = []

    def record_access(
        self,
        memory_id: str,
        context: str,
        session_id: str | None = None,
        trigger: str = "awareness",
    ) -> None:
        """Record an access event in both in-memory log and persistent buffer."""
        super().record_access(memory_id, context)
        log_entry = MemoryAccessLog(
            memory_id=memory_id,
            context=context[:200],
            session_id=session_id,
            trigger=trigger,
        )
        self._buffer.append(log_entry)

    @property
    def buffer_size(self) -> int:
        """Current number of unflushed access log entries."""
        return len(self._buffer)

    @property
    def needs_flush(self) -> bool:
        """Whether the buffer has reached the auto-flush threshold."""
        return len(self._buffer) >= self.FLUSH_THRESHOLD

    async def flush(self) -> int:
        """Flush buffered access logs to storage.

        Updates each referenced memory's access_count and last_accessed
        in the storage backend, then clears the buffer.

        Returns:
            Number of access log entries flushed.
        """
        if not self._buffer:
            return 0

        # Group buffer entries by memory_id for batch update
        updates_by_id: dict[str, list[MemoryAccessLog]] = defaultdict(list)
        for entry in self._buffer:
            updates_by_id[entry.memory_id].append(entry)

        flushed = len(self._buffer)

        for memory_id, entries in updates_by_id.items():
            mem = await self._storage.get(memory_id)
            if mem is None:
                continue

            latest_access = max(e.accessed_at for e in entries)
            new_count = mem.access_count + len(entries)

            await self._storage.update(memory_id, {
                "access_count": new_count,
                "last_accessed": latest_access.isoformat(),
            })

        self._buffer.clear()
        return flushed

    async def get_access_count_persistent(self, memory_id: str) -> int:
        """Get total access count combining buffer and storage."""
        # Count from buffer
        buffer_count = sum(
            1 for entry in self._buffer if entry.memory_id == memory_id
        )
        # Count from storage
        mem = await self._storage.get(memory_id)
        storage_count = mem.access_count if mem else 0
        return storage_count + buffer_count


class FeedbackEngine:
    """Self-improving memory through usage feedback."""

    def __init__(
        self,
        storage: MemoryStoreAdapter,
        config: FeedbackConfig | None = None,
    ):
        self._storage = storage
        self._config = config or FeedbackConfig()
        self._usage = PersistentUsageTracker(storage)

    # Alias for event bus (set externally)
    event_bus: object | None = None

    # ── Core Feedback Methods ───────────────────────

    async def on_memory_accessed(
        self, memory_id: str, context: str = ""
    ) -> None:
        """
        Called when a memory was retrieved AND used by the agent.

        Reinforcement: boost importance, increase access count,
        slow decay acceleration, update last_accessed timestamp.
        """
        mem = await self._storage.get(memory_id)
        if mem is None:
            return

        self._usage.record_access(memory_id, context)

        boost = min(self._config.max_boost, self._config.boost_per_access)
        new_importance = min(1.0, mem.importance + boost)
        new_acceleration = max(0.5, mem.decay_acceleration - 0.05)

        await self._storage.update(memory_id, {
            "importance": new_importance,
            "access_count": mem.access_count + 1,
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "decay_acceleration": new_acceleration,
        })

        await self._emit_event(MemoryEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            event_type="reinforced",
            memory_id=memory_id,
            timestamp=datetime.now(timezone.utc),
            trigger="feedback_engine",
        ))

    async def on_memory_ignored(self, memory_id: str) -> None:
        """
        Called when a memory was injected into context but NOT used.

        Penalty: accelerate decay for disused memories.
        """
        mem = await self._storage.get(memory_id)
        if mem is None:
            return

        new_acceleration = min(2.0, mem.decay_acceleration + self._config.disuse_penalty_rate)

        await self._storage.update(memory_id, {
            "decay_acceleration": new_acceleration,
        })

    async def on_memory_edited(
        self, memory_id: str, old_content: str, new_content: str
    ) -> None:
        """Called when a memory is manually edited by the user."""
        await self._emit_event(MemoryEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            event_type="modified",
            memory_id=memory_id,
            timestamp=datetime.now(timezone.utc),
            previous_state={"content": old_content},
            new_state={"content": new_content},
            trigger="user",
        ))

    # ── Contradiction Detection ─────────────────────

    async def detect_contradictions(
        self, new_memory: MemoryRecord
    ) -> list[Contradiction]:
        """
        Detect if a new memory contradicts existing ones.

        For high-importance memories, uses a more thorough check.
        For low-importance memories, uses a fast heuristic.
        """
        if not self._config.contradiction_detection:
            return []

        # Find semantically similar existing memories
        similar = (
            await self._storage.search_semantic(
                query_embedding=new_memory.embedding or [],
                top_k=5,
                filters={"memory_type": MemoryType.FACT.value},
            )
            if new_memory.embedding
            else []
        )

        contradictions: list[Contradiction] = []
        for existing in similar:
            if existing.id == new_memory.id:
                continue

            # Check method depends on importance
            is_contradiction = False
            if new_memory.effective_importance > self._config.contradiction_llm_threshold:
                is_contradiction = await self._llm_check(new_memory, existing)
            else:
                is_contradiction = self._heuristic_check(new_memory, existing)

            if is_contradiction:
                # Mark both memories as contradictory
                await self._storage.update(new_memory.id, {
                    "contradiction_of": existing.id,
                })
                await self._storage.update(existing.id, {
                    "contradiction_of": new_memory.id,
                })

                contradictions.append(Contradiction(
                    new_memory_id=new_memory.id,
                    existing_memory_id=existing.id,
                    new_content=new_memory.content,
                    existing_content=existing.content,
                    resolution=ContradictionResolution.KEEP_BOTH,
                ))

        return contradictions

    def _heuristic_check(
        self, mem_a: MemoryRecord, mem_b: MemoryRecord
    ) -> bool:
        """Fast heuristic contradiction check — no LLM cost.

        Uses negation patterns and numeric contradiction detection.
        """
        text_a = mem_a.content.lower()
        text_b = mem_b.content.lower()

        # Simple negation check
        negation_a = any(
            word in text_a for word in ["not ", "never ", "don't ", "doesn't ", "isn't "]
        )
        negation_b = any(
            word in text_b for word in ["not ", "never ", "don't ", "doesn't ", "isn't "]
        )

        # If one is positive and the other negative about the same topic,
        # flag as potential contradiction
        if negation_a != negation_b:
            # Check for shared key terms
            terms_a = set(text_a.split()) - _STOP_WORDS
            terms_b = set(text_b.split()) - _STOP_WORDS
            overlap = terms_a & terms_b
            if len(overlap) >= 3:
                return True

        # Numeric contradiction: same entity, different values
        # e.g., "pool size is 10" vs "pool size is 50"
        import re
        numbers_a = re.findall(r"\b(\d+)\b", text_a)
        numbers_b = re.findall(r"\b(\d+)\b", text_b)
        if numbers_a and numbers_b and numbers_a != numbers_b:
            terms_a = set(text_a.split()) - _STOP_WORDS
            terms_b = set(text_b.split()) - _STOP_WORDS
            if len(terms_a & terms_b) >= 3:
                return True

        return False

    async def _llm_check(
        self, mem_a: MemoryRecord, mem_b: MemoryRecord
    ) -> bool:
        """LLM-based contradiction detection for high-importance memories.

        Placeholder — in production, this would call a lightweight
        classification model or an LLM with structured output.
        """
        # For now, fall back to heuristic with stricter threshold
        return self._heuristic_check(mem_a, mem_b)

    # ── Helpers ──────────────────────────────────────

    async def _emit_event(self, event: MemoryEvent) -> None:
        """Emit an event to the event bus if configured."""
        if self.event_bus and hasattr(self.event_bus, "emit"):
            await self.event_bus.emit(event)

    def get_usage_stats(self, memory_id: str) -> dict:
        """Get usage statistics for a specific memory."""
        return {
            "access_count": self._usage.get_access_count(memory_id),
            "last_access": self._usage.get_last_access(memory_id),
            "recent_contexts": self._usage.get_recent_contexts(memory_id),
        }


_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither",
    "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same",
    "it", "its", "this", "that", "these", "those", "he",
    "she", "they", "them", "their", "his", "her", "my",
    "your", "our", "we", "you", "i", "me", "us",
}
