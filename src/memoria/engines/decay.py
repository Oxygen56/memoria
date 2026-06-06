"""Decay Engine — mathematical forgetting via the Ebbinghaus curve.

Unlike Mem0, Letta, and Zep which never forget (growing indefinitely),
the Decay Engine applies a deterministic, explainable decay curve to
every memory. Important memories resist decay. Unused ones fade.
Preferences are eternal.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional

from memoria.core.config import DecayConfig
from memoria.core.models import (
    DecayCycleReport,
    MemoryEvent,
    MemoryLayer,
    MemoryRecord,
    MemoryType,
)
from memoria.storage.base import MemoryStoreAdapter

logger = logging.getLogger(__name__)


class DecayEngine:
    """
    Mathematical forgetting — deterministic, explainable, tunable.

    Core formula:  retention = importance_factor × e^(-t/τ) + access_bonus

    Where:
      t  = days since last access
      τ  = memory-type-specific half-life (in days)
      importance_factor = 0.3 + 0.7 × effective_importance
      access_bonus = min(0.3, log(1 + access_count) × 0.05)

    Half-life by type (in days):
      PREFERENCE:   ∞  — never decays
      CONSTRAINT:   ∞  — never decays
      SKILL:        90
      DECISION:     60
      RELATIONSHIP: 45
      FACT:         30
      EVENT:        14
    """

    HALF_LIFE: Dict[MemoryType, float] = {
        MemoryType.PREFERENCE: float("inf"),
        MemoryType.CONSTRAINT: float("inf"),
        MemoryType.SKILL: 90.0,
        MemoryType.DECISION: 60.0,
        MemoryType.RELATIONSHIP: 45.0,
        MemoryType.FACT: 30.0,
        MemoryType.EVENT: 14.0,
    }

    # Layer transition thresholds
    WARM_TO_COLD_THRESHOLD: float = 0.3
    COLD_TO_OBLIVION_THRESHOLD: float = 0.05

    def __init__(
        self,
        storage: MemoryStoreAdapter,
        config: Optional[DecayConfig] = None,
    ):
        self._storage = storage
        self._config = config or DecayConfig()
        self._background_task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()

        # Apply custom half-life overrides
        for type_name, days in self._config.custom_half_lives.items():
            try:
                mt = MemoryType(type_name)
                self.HALF_LIFE[mt] = days
            except ValueError:
                pass

    # ── Public API ──────────────────────────────────

    def compute_adaptive_interval(self, total_memories: int) -> float:
        """Compute adaptive decay interval based on memory count.

        Strategy:
        - < 100 memories: 24 hours (low urgency)
        - 100-1000: 12 hours
        - 1000-5000: 6 hours (default)
        - 5000-10000: 3 hours
        - > 10000: 1 hour (high urgency)

        The result is clamped to [min_interval_hours, max_interval_hours]
        from config.

        Returns interval in hours.
        """
        if total_memories < 100:
            interval = 24.0
        elif total_memories < 1000:
            interval = 12.0
        elif total_memories < 5000:
            interval = 6.0
        elif total_memories < 10000:
            interval = 3.0
        else:
            interval = 1.0

        # Clamp to configured bounds
        interval = max(self._config.min_interval_hours, interval)
        interval = min(self._config.max_interval_hours, interval)
        return interval

    async def start_background_decay(
        self,
        get_memory_count_fn: Callable[[], Awaitable[int]],
    ) -> None:
        """Start background decay cycle with adaptive scheduling.

        Uses asyncio.create_task for lightweight scheduling.
        Recalculates interval after each cycle based on current memory count.

        Args:
            get_memory_count_fn: Async callable that returns the total number
                of memories currently stored.
        """
        if self._background_task is not None and not self._background_task.done():
            logger.warning("Background decay is already running.")
            return

        self._stop_event.clear()
        self._background_task = asyncio.create_task(
            self._background_loop(get_memory_count_fn)
        )
        logger.info("Background decay scheduler started.")

    async def stop_background_decay(self) -> None:
        """Stop the background decay task gracefully."""
        if self._background_task is None or self._background_task.done():
            logger.debug("No background decay task to stop.")
            return

        self._stop_event.set()
        self._background_task.cancel()
        try:
            await self._background_task
        except asyncio.CancelledError:
            pass
        self._background_task = None
        logger.info("Background decay scheduler stopped.")

    async def _background_loop(
        self,
        get_memory_count_fn: Callable[[], Awaitable[int]],
    ) -> None:
        """Internal loop that runs decay cycles at adaptive intervals."""
        logger.info("Background decay loop entering main loop.")
        while not self._stop_event.is_set():
            try:
                # Determine wait interval
                if self._config.adaptive_interval:
                    count = await get_memory_count_fn()
                    interval_hours = self.compute_adaptive_interval(count)
                else:
                    interval_hours = float(
                        self._config.decay_cycle_interval_hours
                    )

                interval_seconds = interval_hours * 3600.0
                logger.info(
                    "Next decay cycle in %.1f hours (%d seconds).",
                    interval_hours,
                    int(interval_seconds),
                )

                # Sleep, but respect stop_event
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=interval_seconds,
                    )
                    # If we reach here, stop was requested
                    break
                except asyncio.TimeoutError:
                    # Timeout means it's time to run a cycle
                    pass

                # Execute decay cycle
                report = await self.process_decay_cycle()
                logger.info(
                    "Decay cycle completed: %s (took %d ms)",
                    report.stats,
                    report.stats.get("processing_duration_ms", 0),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in background decay cycle.")
                # Back off on error — wait fixed interval before retrying
                await asyncio.sleep(
                    self._config.decay_cycle_interval_hours * 3600.0
                )

    async def process_decay_cycle(self) -> DecayCycleReport:
        """
        Run a full decay cycle over all WARM + COLD memories.

        Call periodically or on explicit trigger.
        Processes up to max_memories_per_cycle per call.

        The returned report includes:
        - processing_duration_ms: Time taken for this cycle (milliseconds).
        - next_interval_minutes: Suggested next interval (minutes), based on
          adaptive calculation if enabled.
        """
        start_time = time.monotonic()

        events: List[MemoryEvent] = []
        stats = {
            "promoted": 0,
            "demoted": 0,
            "archived": 0,
            "scheduled_for_deletion": 0,
            "deleted": 0,
            "unchanged": 0,
        }

        # Process WARM and COLD layers
        for layer in [MemoryLayer.WARM, MemoryLayer.COLD]:
            offset = 0
            batch_size = min(1000, self._config.max_memories_per_cycle)

            while True:
                memories = await self._storage.list_by_layer(
                    layer, limit=batch_size, offset=offset
                )
                if not memories:
                    break

                for mem in memories:
                    old_score = mem.decay_score
                    old_layer = mem.layer

                    # Compute new decay score
                    new_score = self.compute_decay(mem)
                    mem.decay_score = new_score

                    # Determine layer transition
                    new_layer = self._determine_layer(mem, new_score)
                    mem.layer = new_layer

                    # Persist changes
                    await self._storage.update(mem.id, {
                        "decay_score": new_score,
                        "layer": new_layer.value,
                        "last_modified": datetime.now(timezone.utc).isoformat(),
                    })

                    # Track stats
                    if old_layer != new_layer:
                        if new_layer == MemoryLayer.HOT:
                            stats["promoted"] += 1
                        elif new_layer == MemoryLayer.COLD:
                            stats["archived"] += 1
                        elif new_layer == MemoryLayer.OBLIVION:
                            stats["scheduled_for_deletion"] += 1
                        else:
                            stats["demoted"] += 1

                        events.append(MemoryEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:8]}",
                            event_type=(
                                "archived" if new_layer == MemoryLayer.COLD
                                else "forgotten" if new_layer == MemoryLayer.OBLIVION
                                else "promoted" if new_layer == MemoryLayer.HOT
                                else "decayed"
                            ),
                            memory_id=mem.id,
                            timestamp=datetime.now(timezone.utc),
                            previous_state={
                                "layer": old_layer.value,
                                "decay_score": old_score,
                            },
                            new_state={
                                "layer": new_layer.value,
                                "decay_score": new_score,
                            },
                            trigger="decay_engine",
                        ))
                    else:
                        stats["unchanged"] += 1

                offset += batch_size
                if len(memories) < batch_size:
                    break

        # Execute deletion for OBLIVION-marked memories
        oblivion = await self._storage.list_by_layer(
            MemoryLayer.OBLIVION, limit=self._config.max_memories_per_cycle
        )
        for mem in oblivion:
            if self._config.consolidate_before_deletion:
                await self._consolidate(mem)
            await self._storage.delete([mem.id])
            stats["deleted"] += 1

        # Compute processing duration
        duration_ms = int((time.monotonic() - start_time) * 1000)
        stats["processing_duration_ms"] = duration_ms

        # Compute next suggested interval
        total_processed = sum(
            v for k, v in stats.items()
            if k not in ("processing_duration_ms", "next_interval_minutes")
        )
        if self._config.adaptive_interval:
            suggested_hours = self.compute_adaptive_interval(total_processed)
        else:
            suggested_hours = float(self._config.decay_cycle_interval_hours)
        stats["next_interval_minutes"] = int(suggested_hours * 60)

        logger.info(
            "Decay cycle done in %d ms — processed %d memories, "
            "next suggested interval: %.1f hours.",
            duration_ms,
            total_processed,
            suggested_hours,
        )

        return DecayCycleReport(
            events=events,
            stats=stats,
            timestamp=datetime.now(timezone.utc),
        )

    def compute_decay(self, memory: MemoryRecord) -> float:
        """Compute the decay score for a single memory.

        This is a pure function — deterministic and testable.
        """
        # Eternal memories never decay
        if memory.memory_type in (MemoryType.PREFERENCE, MemoryType.CONSTRAINT):
            return 1.0

        last_accessed = memory.last_accessed
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)
        days_since = (datetime.now(timezone.utc) - last_accessed).days
        half_life = self.HALF_LIFE.get(memory.memory_type, 30.0)

        # Base Ebbinghaus decay
        base_decay = math.exp(-max(0, days_since) / half_life)

        # Importance modulation: important memories decay slower
        importance_factor = 0.3 + (0.7 * memory.effective_importance)

        # Access frequency bonus: frequently accessed memories resist decay
        access_bonus = min(0.3, math.log(1 + memory.access_count) * 0.05)

        # Disuse penalty: if never accessed, accelerate decay
        disuse_penalty = (
            memory.decay_acceleration
            if memory.access_count == 0
            else 1.0
        )

        return min(1.0, (base_decay * importance_factor + access_bonus) / disuse_penalty)

    def _determine_layer(
        self, memory: MemoryRecord, decay_score: float
    ) -> MemoryLayer:
        """Determine which layer a memory belongs in based on its score."""
        # Eternal memories stay HOT
        if memory.memory_type in (MemoryType.PREFERENCE, MemoryType.CONSTRAINT):
            return MemoryLayer.HOT

        # Combined metric for promotion
        combined = (decay_score + memory.effective_importance) / 2

        if combined > self._config.auto_promote_to_hot_threshold:
            return MemoryLayer.HOT
        elif decay_score > self.WARM_TO_COLD_THRESHOLD:
            return MemoryLayer.WARM
        elif decay_score > self.COLD_TO_OBLIVION_THRESHOLD:
            return MemoryLayer.COLD
        else:
            return MemoryLayer.OBLIVION

    async def _consolidate(self, memory: MemoryRecord) -> None:
        """Create a compressed summary before deleting."""
        summary = f"[Consolidated] {memory.content[:200]}..."
        record = MemoryRecord(
            content=summary,
            memory_type=MemoryType.EVENT,
            layer=MemoryLayer.COLD,
            importance=0.1,
            tags=["consolidated"] + memory.tags,
            related_memories=[memory.id],
            created_by="decay_engine",
        )
        await self._storage.insert([record])
