"""Awareness Engine — proactive memory recall without explicit queries.

The core differentiator of Memoria. Unlike all competitors that wait for
the user/agent to explicitly search for memories, the Awareness Engine
computes a semantic fingerprint of every conversation turn and automatically
injects relevant memories before the agent needs to ask.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from memoria.core.config import AwarenessConfig
from memoria.core.models import (
    ContextInjection,
    InjectItem,
    MemoryLayer,
    MemoryRecord,
    SemanticFingerprint,
)
from memoria.embedding.base import EmbeddingProvider
from memoria.storage.base import MemoryStoreAdapter


class AwarenessEngine:
    """
    Proactive memory recall — "Knows Before You Ask."

    Computes a semantic fingerprint of every conversation turn,
    automatically matches against the memory store, and injects
    relevant memories into the agent's context window.

    No explicit search query required from the user or agent.
    """

    # Class-level tiktoken encoder cache (avoid re-creation per call)
    _tiktoken_encoder: object | None = None

    def __init__(
        self,
        storage: MemoryStoreAdapter,
        embedding_provider: EmbeddingProvider,
        config: Optional[AwarenessConfig] = None,
    ):
        self._storage = storage
        self._embedding = embedding_provider
        self._config = config or AwarenessConfig()

    # ── Public API ──────────────────────────────────

    async def get_context(
        self,
        input_text: str,
        context_window: Optional[List[str]] = None,
        token_budget: Optional[int] = None,
    ) -> ContextInjection:
        """
        Main entry point. Called at the start of every agent turn.

        Returns memories to inject into the agent's system prompt or context.

        Args:
            input_text: Current user message or agent task description.
            context_window: Recent conversation turns for context enrichment.
            token_budget: Max tokens to inject. Falls back to config default.

        Returns:
            ContextInjection with hot memories (always present) and
            relevant memories (matched by semantic fingerprint).
        """
        if token_budget is None:
            token_budget = self._config.token_budget_default
        token_budget = min(token_budget, self._config.token_budget_max)

        context_window = context_window or []

        # Step 1: Compute semantic fingerprint
        fingerprint = await self._compute_fingerprint(input_text, context_window)

        # Step 2: Hybrid search (semantic + keyword) against WARM layer
        candidates = await self._storage.search_hybrid(
            query=fingerprint.search_query,
            query_embedding=fingerprint.embedding,
            top_k=self._config.max_candidates,
            filters={
                "layer": {
                    "$in": [
                        MemoryLayer.HOT.value,
                        MemoryLayer.WARM.value,
                    ]
                }
            },
            rrf_k=self._config.rrf_k,
        )

        # Step 3: Score relevance of each candidate against the fingerprint
        scored: List[Tuple[MemoryRecord, float]] = []
        for mem in candidates:
            relevance = self._score_relevance(mem, fingerprint)
            if relevance >= self._config.relevance_threshold:
                scored.append((mem, relevance))

        # Step 4: Rank by composite score (relevance × decay × importance)
        scored.sort(
            key=lambda x: x[1] * x[0].decay_score * x[0].effective_importance,
            reverse=True,
        )

        # Step 5: Pack into token budget
        injected: List[InjectItem] = []
        token_count = 0

        for mem, score in scored:
            mem_tokens = self._estimate_tokens(mem.content)
            if token_count + mem_tokens > token_budget:
                break
            injected.append(InjectItem(memory=mem, relevance_score=round(score, 4)))
            token_count += mem_tokens

        # Step 6: HOT layer memories are always included
        hot_memories = [m for m in candidates if m.layer == MemoryLayer.HOT]
        hot_tokens = sum(self._estimate_tokens(m.content) for m in hot_memories)

        return ContextInjection(
            hot=hot_memories,
            relevant=injected,
            total_tokens=token_count + hot_tokens,
            fingerprint_version=fingerprint.version,
        )

    # ── Fingerprint Computation ─────────────────────

    async def _compute_fingerprint(
        self,
        input_text: str,
        context_window: List[str],
    ) -> SemanticFingerprint:
        """Build a semantic fingerprint of the current conversation state."""
        # Combine current input with recent context
        recent_context = context_window[-self._config.context_window_size :]
        combined = " ".join(recent_context + [input_text])

        # Get embedding from the configured provider
        embedding = await self._embedding.embed(combined)

        # Extract key terms for keyword search (lightweight NLP, no LLM cost)
        key_terms = self._extract_key_terms(combined)

        # Build a search query that balances specificity with coverage
        search_query = self._build_search_query(key_terms, input_text)

        return SemanticFingerprint(
            embedding=embedding,
            key_terms=key_terms,
            search_query=search_query,
            version=self._config.fingerprint_version,
        )

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract meaningful key terms without an LLM call.

        Uses regex patterns for common technical identifiers
        plus noun phrase heuristics.
        """
        terms: List[str] = []

        # Pattern 1: Technical identifiers
        tech_patterns = [
            r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",  # CamelCase
            r"\b[a-z]+(?:[._-][a-z]+)+\b",  # snake_case, kebab-case, dot.separated
            r"\b[A-Z]{2,}\b",  # Acronyms (API, SQL, AWS)
            r"\b\d+(?:\.\d+)?(?:gb|mb|kb|ms|s|m)?\b",  # Numbers with units
            r"\b(?:https?://|www\.)[^\s]+\b",  # URLs
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, text)
            terms.extend(matches)

        # Pattern 2: Significant capitalized words (proper nouns)
        proper_nouns = re.findall(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b", text)
        terms.extend(proper_nouns)

        # Pattern 3: Domain-specific terms (lowercase, 4+ chars, not stop words)
        stop_words = {
            "this", "that", "with", "from", "have", "been", "were", "they",
            "them", "then", "than", "about", "which", "what", "when", "where",
            "would", "could", "should", "there", "their", "these", "those",
            "doing", "being", "just", "also", "very", "really", "only", "into",
            "over", "under", "after", "before", "between", "through",
        }
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        terms.extend(w for w in words if w not in stop_words)

        # Deduplicate preserving order
        seen = set()
        unique = []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)
        return unique[:20]  # Cap at 20 terms

    def _build_search_query(self, key_terms: List[str], input_text: str) -> str:
        """Build a keyword search query from extracted terms."""
        # Use the most distinctive terms (rarer = better for search)
        distinctive = sorted(
            key_terms,
            key=lambda t: len(t),
            reverse=True,
        )
        query = " ".join(distinctive[:10])
        return query if query.strip() else input_text[:200]

    # ── Relevance Scoring ────────────────────────────

    def _score_relevance(
        self,
        memory: MemoryRecord,
        fingerprint: SemanticFingerprint,
    ) -> float:
        """Multi-factor relevance scoring.

        Factors (weighted):
        - Semantic similarity (cosine between embeddings) — 50%
        - Keyword overlap (how many key terms appear in the memory) — 25%
        - Recency (recently accessed memories score higher) — 15%
        - Tag match (tag overlap bonus) — 10%
        """
        # 1. Semantic similarity
        semantic_score = 0.0
        if memory.embedding and fingerprint.embedding:
            semantic_score = self._cosine_similarity(
                memory.embedding, fingerprint.embedding
            )

        # 2. Keyword overlap
        keyword_score = 0.0
        if fingerprint.key_terms:
            content_lower = memory.content.lower()
            matches = sum(
                1 for term in fingerprint.key_terms
                if term.lower() in content_lower
            )
            keyword_score = matches / max(1, len(fingerprint.key_terms))

        # 3. Recency
        days_since = (datetime.now(timezone.utc) - memory.last_accessed).days
        recency_score = math.exp(-days_since / 7.0)  # 7-day half-life

        # 4. Tag match
        tag_score = 0.0
        if memory.tags and fingerprint.key_terms:
            tag_matches = sum(
                1 for tag in memory.tags
                if any(term.lower() in tag.lower() for term in fingerprint.key_terms)
            )
            tag_score = min(1.0, tag_matches * 0.1)

        # Weighted combination
        return (
            self._config.semantic_weight * semantic_score
            + self._config.keyword_weight * keyword_score
            + self._config.recency_weight * recency_score
            + self._config.tag_weight * tag_score
        )

    # ── Utilities ────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            # Truncate to shorter
            min_len = min(len(a), len(b))
            a, b = a[:min_len], b[:min_len]

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    @classmethod
    def _estimate_tokens(cls, text: str) -> int:
        """Accurate token estimation using tiktoken with cached encoder.

        Uses the cl100k_base encoding (GPT-4 / text-embedding-3-small).
        Falls back to character-based heuristic (~4 chars/token) if tiktoken
        is unavailable or raises an error.
        """
        try:
            import tiktoken

            if cls._tiktoken_encoder is None:
                cls._tiktoken_encoder = tiktoken.encoding_for_model("gpt-4")
            return len(cls._tiktoken_encoder.encode(text))
        except Exception:
            # Fallback to rough estimation
            return max(1, len(text) // 4)
