# Show HN: Memoria – Memory infrastructure for AI agents with forgetting curves

I built an open-source memory layer for AI agents that treats memory as a living system — not just a vector store with CRUD operations.

**What it is:** Memoria is a Python-native memory infrastructure that gives AI agents human-like memory capabilities: memories decay over time following Ebbinghaus forgetting curves, related context surfaces proactively without explicit queries, and a knowledge graph captures entity relationships for multi-hop reasoning.

**Why I built it:** Every "memory" solution I tried for my agents was basically a vector DB wrapper. Store embeddings, retrieve by similarity, done. But human memory doesn't work that way — we forget, we make unexpected connections, and relevant memories pop into our heads without being asked. I wanted agents to have that.

**How it works — four cooperating engines:**

- **Decay Engine** — Each of the 7 memory types (fact, preference, event, decision, relationship, skill, constraint) has a different half-life. A user's food preference decays slower than a one-off event mention. Memories that aren't reinforced fade naturally.

- **Awareness Engine** — Proactive recall. Instead of waiting for a query, it monitors context and injects relevant memories before you ask. Think "Oh, this user mentioned they hate JSON last week" surfacing automatically.

- **Graph Engine** — Extracts entities and relationships, enabling multi-hop reasoning ("User works at Company X → Company X is in fintech → suggest fintech-relevant memories").

- **Orchestrator** — Coordinates all engines: contradiction filtering, graph-connectivity boosting, decay protection for reinforced memories.

**Storage:** Three-tier hot/warm/cold architecture with automatic OBLIVION cleanup. Storage-agnostic via adapters (InMemory, ChromaDB, PgVector) — zero vendor lock-in.

**Contradiction detection** uses a provider pattern (LLM, CrossEncoder, or Heuristic) so you can trade accuracy for speed.

The whole thing is ~3k lines of Python, no server required, pip-installable.

**vs existing tools:** Mem0 gives you basic CRUD; Letta requires server deployment; Zep's best features are closed-source. Memoria is fully open, lightweight, and treats memory as a first-class cognitive system.

GitHub: https://github.com/Oxygen56/memoria

Would love feedback on the decay model parameters and the awareness engine's proactive injection strategy. Issues and PRs welcome.
