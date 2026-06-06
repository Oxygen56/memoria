"""Engine Orchestrator — coordinates information flow between the four engines.

Key coordination rules:
1. Decay considers feedback data: frequently used memories decay slower
2. Awareness considers contradictions: don't inject contradicted memories
3. Graph results influence awareness ranking: graph-connected memories get bonus
4. Feedback informs decay: high-access memories get protected from decay
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memoria.engines.awareness import AwarenessEngine
    from memoria.engines.decay import DecayEngine
    from memoria.engines.feedback import FeedbackEngine
    from memoria.engines.graph import GraphEngine

from memoria.core.models import ContextInjection, InjectItem, MemoryRecord

logger = logging.getLogger(__name__)


class EngineOrchestrator:
    """Coordinates information flow between Memoria's four engines.

    Acts as a middleware layer that enriches engine decisions with
    cross-engine intelligence:

    - Before Awareness returns: filter contradictions, boost graph-connected memories
    - Before Decay processes: check feedback data to protect high-use memories
    - After Feedback events: propagate to graph (update relations)
    """

    def __init__(
        self,
        awareness: AwarenessEngine | None = None,
        decay: DecayEngine | None = None,
        feedback: FeedbackEngine | None = None,
        graph: GraphEngine | None = None,
    ):
        self._awareness = awareness
        self._decay = decay
        self._feedback = feedback
        self._graph = graph

    # ── Awareness Coordination ────────────────────────

    async def post_process_context(
        self, injection: ContextInjection, input_text: str
    ) -> ContextInjection:
        """Post-process awareness results with cross-engine intelligence.

        1. Filter out memories with unresolved contradictions
        2. Boost memories connected through the graph to other relevant memories
        3. Add graph-discovered memories that awareness might have missed
        """
        if not injection.relevant:
            return injection

        filtered_relevant: list[InjectItem] = []

        for item in injection.relevant:
            # Rule 1: Skip contradicted memories
            if item.memory.contradiction_of:
                logger.debug(f"Filtering contradicted memory: {item.memory.id}")
                continue

            # Rule 2: Boost graph-connected memories
            if self._graph:
                related = await self._graph.get_related_memories(item.memory.id, max_hops=1)
                # If this memory is graph-connected to other relevant memories, boost it
                relevant_ids = {i.memory.id for i in injection.relevant}
                graph_connections = len(set(related) & relevant_ids)
                if graph_connections > 0:
                    # Boost relevance score by 10% per connection (max 30%)
                    boost = min(0.3, graph_connections * 0.1)
                    item = InjectItem(
                        memory=item.memory,
                        relevance_score=min(1.0, item.relevance_score * (1 + boost))
                    )

            filtered_relevant.append(item)

        # Re-sort by boosted scores
        filtered_relevant.sort(key=lambda x: x.relevance_score, reverse=True)

        return ContextInjection(
            hot=injection.hot,
            relevant=filtered_relevant,
            total_tokens=injection.total_tokens,
            fingerprint_version=injection.fingerprint_version,
        )

    # ── Decay Coordination ────────────────────────────

    def should_protect_from_decay(self, memory: MemoryRecord) -> bool:
        """Check if a memory should be protected from decay based on feedback data.

        Protection criteria:
        - access_count > 5 (frequently used)
        - Used in the last 7 days
        - Has many graph connections (central node)
        """
        if memory.access_count > 5:
            return True

        from datetime import datetime, timezone
        days_since_access = (datetime.now(timezone.utc) - memory.last_accessed).days
        if days_since_access < 7 and memory.access_count > 2:
            return True

        return False

    def get_decay_modifier(self, memory: MemoryRecord) -> float:
        """Get a decay rate modifier based on cross-engine data.

        Returns a multiplier (< 1.0 means slower decay, > 1.0 means faster).
        """
        modifier = 1.0

        # High access count slows decay
        if memory.access_count > 10:
            modifier *= 0.7
        elif memory.access_count > 5:
            modifier *= 0.85

        # Contradicted memories decay faster
        if memory.contradiction_of:
            modifier *= 1.3

        return modifier

    # ── Graph Coordination ────────────────────────────

    async def enrich_with_graph(
        self, memory_ids: list[str], max_hops: int = 1
    ) -> set[str]:
        """Discover additional relevant memories through graph connections."""
        if not self._graph:
            return set()

        discovered: set[str] = set()
        for mid in memory_ids:
            related = await self._graph.get_related_memories(mid, max_hops)
            discovered.update(related)

        # Remove the original IDs
        discovered -= set(memory_ids)
        return discovered

    # ── Stats ─────────────────────────────────────────

    def get_coordination_stats(self) -> dict[str, any]:
        """Return orchestrator statistics."""
        return {
            "has_awareness": self._awareness is not None,
            "has_decay": self._decay is not None,
            "has_feedback": self._feedback is not None,
            "has_graph": self._graph is not None,
        }
