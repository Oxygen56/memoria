#!/usr/bin/env python3
"""Ebbinghaus decay demonstration — watch memories fade over time.

Demonstrates:
  - How decay scores change based on memory type and importance
  - Manual decay cycle triggering
  - Layer transitions (HOT → WARM → COLD → OBLIVION)
  - How preferences/constraints are immune to decay

Uses DummyEmbeddingProvider so no API keys or external services are needed.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from memoria import Memoria, MemoryType, MemoryLayer, DecayCycleReport
from memoria.embedding.base import DummyEmbeddingProvider


async def main():
    async with Memoria(embedding=DummyEmbeddingProvider()) as mem:
        # ── Create memories with varying types and importance ──
        print("=== Creating memories with different types ===\n")

        memories = []

        # Preference — should NEVER decay (eternal)
        m = await mem.remember(
            "User prefers vim keybindings",
            memory_type="preference",
            importance=0.7,
        )
        memories.append(m)
        print(f"  [preference] {m.content}")
        print(f"    → initial decay_score={m.decay_score:.4f}, layer={m.layer.value}")

        # Constraint — should NEVER decay (eternal)
        m = await mem.remember(
            "Never use port 5432 — reserved for production DB",
            memory_type="constraint",
            importance=0.9,
        )
        memories.append(m)
        print(f"  [constraint] {m.content}")
        print(f"    → initial decay_score={m.decay_score:.4f}, layer={m.layer.value}")

        # Fact — half-life 30 days
        m = await mem.remember(
            "User's database is PostgreSQL 16",
            memory_type="fact",
            importance=0.5,
        )
        memories.append(m)
        print(f"  [fact] {m.content}")
        print(f"    → initial decay_score={m.decay_score:.4f}, layer={m.layer.value}")

        # Event — half-life 14 days (decays fastest)
        m = await mem.remember(
            "Ran database migration v42 on 2026-06-01",
            memory_type="event",
            importance=0.3,
        )
        memories.append(m)
        print(f"  [event] {m.content}")
        print(f"    → initial decay_score={m.decay_score:.4f}, layer={m.layer.value}")

        # Skill — half-life 90 days (decays slowest among non-eternal)
        m = await mem.remember(
            "User knows Docker Compose for multi-service setups",
            memory_type="skill",
            importance=0.6,
        )
        memories.append(m)
        print(f"  [skill] {m.content}")
        print(f"    → initial decay_score={m.decay_score:.4f}, layer={m.layer.value}")

        # ── Simulate time passage by modifying last_accessed ──
        print("\n=== Simulating time passage (45 days) ===\n")

        # Manually adjust last_accessed to simulate 45 days passing
        simulated_past = datetime.now(timezone.utc) - timedelta(days=45)
        for m in memories:
            await mem.edit(m.id, last_accessed=simulated_past.isoformat())

        # ── Run decay cycle ─────────────────────────────────
        print("Running decay cycle...\n")
        report: DecayCycleReport = await mem.run_decay_cycle()

        print(f"  Cycle completed at: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Stats: {report.stats}")

        # ── Show decay results ──────────────────────────────
        print("\n=== Decay Results (after 45 days of inactivity) ===\n")
        print(f"  {'Type':<12} {'Content':<45} {'Decay':<8} {'Layer':<10}")
        print(f"  {'─'*12} {'─'*45} {'─'*8} {'─'*10}")

        for m in memories:
            updated = await mem.get_memory(m.id)
            if updated:
                # After decay cycle, layer may be stored as string or enum
                layer_str = (
                    updated.layer.value
                    if hasattr(updated.layer, "value")
                    else str(updated.layer)
                )
                type_str = (
                    updated.memory_type.value
                    if hasattr(updated.memory_type, "value")
                    else str(updated.memory_type)
                )
                print(
                    f"  {type_str:<12} "
                    f"{updated.content[:43]:<45} "
                    f"{updated.decay_score:<8.4f} "
                    f"{layer_str:<10}"
                )

        # ── Show layer transitions from events ──────────────
        if report.events:
            print(f"\n=== Layer Transitions ({len(report.events)} events) ===\n")
            for evt in report.events:
                prev = evt.previous_state or {}
                new = evt.new_state or {}
                print(
                    f"  [{evt.event_type}] memory={evt.memory_id[:16]}... "
                    f"{prev.get('layer', '?')} → {new.get('layer', '?')} "
                    f"(decay: {prev.get('decay_score', 0):.4f} → {new.get('decay_score', 0):.4f})"
                )
        else:
            print("\n  No layer transitions occurred (memories still above thresholds)")

        # ── Explain the decay formula ───────────────────────
        print("\n=== Decay Formula ===")
        print("  retention = importance_factor × e^(-t/τ) + access_bonus")
        print("  Where:")
        print("    t = days since last access")
        print("    τ = type-specific half-life (FACT=30d, EVENT=14d, SKILL=90d)")
        print("    importance_factor = 0.3 + 0.7 × effective_importance")
        print("    access_bonus = min(0.3, log(1 + access_count) × 0.05)")
        print("\n  PREFERENCE and CONSTRAINT types never decay (τ = ∞)")

        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
