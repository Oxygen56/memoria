#!/usr/bin/env python3
"""Basic Memoria usage — remember, recall, and search.

Demonstrates the core API:
  - Storing memories with different types and importance levels
  - Recalling relevant memories for a given input (proactive injection)
  - Searching memories explicitly by query

Uses DummyEmbeddingProvider so no API keys or external services are needed.
"""

import asyncio

from memoria import Memoria, MemoryType
from memoria.embedding.base import DummyEmbeddingProvider


async def main():
    # Initialize with DummyEmbeddingProvider for local testing (no API key needed)
    async with Memoria(embedding=DummyEmbeddingProvider()) as mem:
        # ── Store memories ──────────────────────────────────
        print("Storing memories...")

        m1 = await mem.remember(
            "User prefers dark mode",
            memory_type="preference",
            importance=0.8,
            tags=["ui", "theme"],
        )
        print(f"  Stored: {m1.content} (id={m1.id}, layer={m1.layer.value})")

        m2 = await mem.remember(
            "User's name is Alice",
            memory_type="fact",
            importance=0.6,
            tags=["user", "identity"],
        )
        print(f"  Stored: {m2.content} (id={m2.id}, layer={m2.layer.value})")

        m3 = await mem.remember(
            "Meeting with Bob at 3pm tomorrow",
            memory_type="event",
            importance=0.5,
            tags=["calendar", "meeting"],
        )
        print(f"  Stored: {m3.content} (id={m3.id}, layer={m3.layer.value})")

        m4 = await mem.remember(
            "Never deploy on Fridays without QA sign-off",
            memory_type="constraint",
            importance=0.9,
            tags=["deployment", "process"],
        )
        print(f"  Stored: {m4.content} (id={m4.id}, layer={m4.layer.value})")

        # ── Recall relevant memories ────────────────────────
        print("\n=== Recall: 'What does the user like?' ===")
        context = await mem.recall("What does the user like?")

        if context.hot:
            print("  [HOT memories — always injected]")
            for item in context.hot:
                print(f"    • {item.content} (type={item.memory_type.value})")

        if context.relevant:
            print("  [RELEVANT memories — ranked by relevance]")
            for item in context.relevant:
                print(
                    f"    • {item.memory.content} "
                    f"(score={item.relevance_score:.3f})"
                )

        if not context.hot and not context.relevant:
            print("  (no memories matched)")

        print(f"  Total tokens used: {context.total_tokens}")

        # ── Search memories explicitly ──────────────────────
        print("\n=== Search: 'user preferences' ===")
        results = await mem.search("user preferences")
        print(f"  Found {len(results)} result(s):")
        for r in results:
            print(
                f"    [{r.memory_type.value}] {r.content} "
                f"(importance={r.importance:.2f}, decay={r.decay_score:.2f})"
            )

        # ── Search with type filter ─────────────────────────
        print("\n=== Search: type=event ===")
        events = await mem.search("meeting", memory_type=MemoryType.EVENT)
        print(f"  Found {len(events)} event(s):")
        for e in events:
            print(f"    • {e.content}")

        # ── Get system stats ────────────────────────────────
        print("\n=== System Stats ===")
        stats = await mem.stats()
        print(f"  Total memories: {stats.storage.total_memories}")
        print(f"  By layer: {stats.storage.by_layer}")
        print(f"  Health score: {stats.health_score:.1f}")
        print(f"  Uptime: {stats.uptime_seconds:.1f}s")

        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
