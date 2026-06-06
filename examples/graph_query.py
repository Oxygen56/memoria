#!/usr/bin/env python3
"""Knowledge graph query demonstration.

Demonstrates:
  - How Memoria automatically extracts entities and relations from memories
  - Querying the knowledge graph by entity name
  - Multi-hop traversal to discover connected knowledge
  - Getting related memories through graph connections

Uses DummyEmbeddingProvider so no API keys or external services are needed.
"""

import asyncio

from memoria import Memoria, MemoryType
from memoria.embedding.base import DummyEmbeddingProvider


async def main():
    async with Memoria(embedding=DummyEmbeddingProvider()) as mem:
        # ── Store interconnected memories ───────────────────
        print("=== Storing interconnected memories ===\n")

        # Technology stack memories — graph engine extracts entities automatically
        m1 = await mem.remember(
            "Project uses FastAPI for the REST API layer",
            memory_type="fact",
            importance=0.7,
            tags=["tech-stack", "backend"],
        )
        print(f"  1. {m1.content}")

        m2 = await mem.remember(
            "FastAPI depends on Pydantic for data validation",
            memory_type="relationship",
            importance=0.6,
            tags=["tech-stack", "dependencies"],
        )
        print(f"  2. {m2.content}")

        m3 = await mem.remember(
            "PostgreSQL is the primary database, deployed on AWS",
            memory_type="fact",
            importance=0.8,
            tags=["tech-stack", "database"],
        )
        print(f"  3. {m3.content}")

        m4 = await mem.remember(
            "Chose PostgreSQL over MySQL for JSON support",
            memory_type="decision",
            importance=0.7,
            tags=["tech-stack", "decision"],
        )
        print(f"  4. {m4.content}")

        m5 = await mem.remember(
            "Redis is used for caching and session storage",
            memory_type="fact",
            importance=0.6,
            tags=["tech-stack", "caching"],
        )
        print(f"  5. {m5.content}")

        m6 = await mem.remember(
            "Docker Compose integrates with Redis and PostgreSQL for local dev",
            memory_type="skill",
            importance=0.5,
            tags=["devops", "local-dev"],
        )
        print(f"  6. {m6.content}")

        # ── Query the knowledge graph ───────────────────────
        print("\n=== Graph Query: 'PostgreSQL' (2 hops) ===\n")

        result = await mem.query_graph("PostgreSQL", hops=2)

        if result is None:
            print("  Graph engine not available.")
        else:
            print(f"  Entities found: {len(result.entities)}")
            for ent in result.entities:
                print(f"    • {ent.name} (type={ent.entity_type})")

            print(f"\n  Relations found: {len(result.relations)}")
            for rel in result.relations:
                # Resolve entity names for display
                source_name = next(
                    (e.name for e in result.entities if e.id == rel.source_entity_id),
                    rel.source_entity_id[:12],
                )
                target_name = next(
                    (e.name for e in result.entities if e.id == rel.target_entity_id),
                    rel.target_entity_id[:12],
                )
                print(
                    f"    {source_name} --[{rel.relation_type}]--> {target_name} "
                    f"(confidence={rel.confidence:.2f})"
                )

            print(f"\n  Related memory IDs: {len(result.related_memory_ids)}")
            for mid in result.related_memory_ids:
                record = await mem.get_memory(mid)
                if record:
                    print(f"    • {record.content}")

        # ── Query another entity ────────────────────────────
        print("\n=== Graph Query: 'FastAPI' (3 hops) ===\n")

        result2 = await mem.query_graph("FastAPI", hops=3)

        if result2 and result2.entities:
            print(f"  Discovered {len(result2.entities)} connected entities:")
            for ent in result2.entities:
                print(f"    • {ent.name} ({ent.entity_type})")

            if result2.paths:
                print(f"\n  Traversal paths ({len(result2.paths)}):")
                for i, path in enumerate(result2.paths[:5]):  # Show first 5 paths
                    # Resolve path entity IDs to names
                    path_names = []
                    for eid in path:
                        entity = next(
                            (e for e in result2.entities if e.id == eid), None
                        )
                        path_names.append(entity.name if entity else eid[:8])
                    print(f"    Path {i+1}: {' → '.join(path_names)}")
        else:
            print("  No results for 'FastAPI'")

        # ── Get related memories through the graph ──────────
        print(f"\n=== Related Memories for: '{m4.content}' ===\n")

        related_ids = await mem.get_related_memories(m4.id, max_hops=2)
        if related_ids:
            print(f"  Found {len(related_ids)} related memory(ies):")
            for rid in related_ids:
                record = await mem.get_memory(rid)
                if record:
                    print(f"    • [{record.memory_type.value}] {record.content}")
        else:
            print("  No related memories found through graph connections")

        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
