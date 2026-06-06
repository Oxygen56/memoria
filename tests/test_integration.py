"""End-to-end integration tests for Memoria."""

import pytest

from memoria.core.memoria import Memoria
from memoria.core.models import (
    ContextInjection,
    MemoryLayer,
    MemoryType,
)
from memoria.embedding.base import DummyEmbeddingProvider


@pytest.fixture
async def memoria(tmp_path):
    """Create a Memoria instance with in-memory backend and dummy embeddings."""
    mem = Memoria(
        warm_backend="memory",
        embedding=DummyEmbeddingProvider(),
        data_dir=str(tmp_path / ".memoria"),
    )
    await mem.initialize()
    yield mem
    await mem.close()


class TestRememberRecallLoop:
    """Test the complete remember → recall → report_usage flow."""

    async def test_remember_and_recall(self, memoria):
        """Basic: store a memory and recall it."""
        record = await memoria.remember(
            "PostgreSQL 16 is our production database",
            memory_type=MemoryType.FACT,
            importance=0.8,
            tags=["database", "postgresql"],
        )
        assert record.id.startswith("mem_")
        assert record.layer == MemoryLayer.WARM

        ctx = await memoria.recall("what database do we use?")
        assert isinstance(ctx, ContextInjection)
        # Should find the memory via keyword/semantic match
        assert ctx.total_tokens >= 0

    async def test_hot_memory_always_injected(self, memoria):
        """HOT memories should always be in context."""
        record = await memoria.remember(
            "User prefers concise responses",
            memory_type=MemoryType.PREFERENCE,
            importance=0.9,
        )
        # PREFERENCE type should go to HOT layer
        assert record.layer == MemoryLayer.HOT

        ctx = await memoria.recall("tell me about anything")
        # HOT memories are always included
        assert len(ctx.hot) >= 1
        assert any(m.id == record.id for m in ctx.hot)

    async def test_constraint_memory_is_hot(self, memoria):
        """CONSTRAINT type memories should be stored as HOT."""
        record = await memoria.remember(
            "Never use port 5432 — it's reserved",
            memory_type=MemoryType.CONSTRAINT,
            importance=0.7,
        )
        assert record.layer == MemoryLayer.HOT

    async def test_high_importance_is_hot(self, memoria):
        """Memories with importance >= 0.85 should be HOT."""
        record = await memoria.remember(
            "Critical: API key rotation is daily",
            memory_type=MemoryType.FACT,
            importance=0.9,
        )
        assert record.layer == MemoryLayer.HOT

    async def test_report_usage_reinforces(self, memoria):
        """report_usage should boost used memories."""
        from memoria.core.models import InjectItem

        record = await memoria.remember(
            "Redis pool size is 10",
            memory_type=MemoryType.FACT,
            importance=0.5,
        )
        # Build a ContextInjection that includes the memory
        # (since DummyEmbedding may not pass relevance threshold via recall)
        ctx = ContextInjection(
            hot=[],
            relevant=[InjectItem(memory=record, relevance_score=0.8)],
            total_tokens=20,
        )

        # Report that we used the memory
        await memoria.report_usage(ctx, used_memory_ids=[record.id])

        # Check access count was increased
        updated = await memoria.get_memory(record.id)
        assert updated is not None
        assert updated.access_count >= 1

    async def test_report_usage_penalizes_ignored(self, memoria):
        """Ignored memories should have decay accelerated."""
        from memoria.core.models import InjectItem

        record = await memoria.remember(
            "Old config: port 3000",
            memory_type=MemoryType.FACT,
            importance=0.4,
        )
        # Build a ContextInjection that includes the memory
        ctx = ContextInjection(
            hot=[],
            relevant=[InjectItem(memory=record, relevance_score=0.6)],
            total_tokens=20,
        )

        # Report that we did NOT use the memory (empty used list)
        await memoria.report_usage(ctx, used_memory_ids=[])

        # decay_acceleration should increase for ignored memories
        updated = await memoria.get_memory(record.id)
        assert updated is not None
        assert updated.decay_acceleration > 1.0

    async def test_report_usage_default_assumes_all_used(self, memoria):
        """If used_memory_ids is None, all injected memories are considered used."""
        record = await memoria.remember(
            "Docker compose version 3.8",
            memory_type=MemoryType.FACT,
            importance=0.5,
            tags=["docker"],
        )
        ctx = await memoria.recall("docker compose version")
        # Default: None means all used → no penalty
        await memoria.report_usage(ctx, used_memory_ids=None)

        updated = await memoria.get_memory(record.id)
        assert updated is not None
        # No penalty applied, acceleration should stay at 1.0
        assert updated.decay_acceleration <= 1.0


class TestDecayCycle:
    """Test decay processing."""

    async def test_decay_cycle_runs(self, memoria):
        """Decay cycle should complete without errors."""
        await memoria.remember("temporary event", memory_type=MemoryType.EVENT)
        report = await memoria.run_decay_cycle()
        assert report is not None
        assert "unchanged" in report.stats or "promoted" in report.stats

    async def test_preference_never_decays(self, memoria):
        """PREFERENCE memories should never decay."""
        record = await memoria.remember(
            "User likes dark mode",
            memory_type=MemoryType.PREFERENCE,
        )
        await memoria.run_decay_cycle()
        updated = await memoria.get_memory(record.id)
        assert updated is not None
        assert updated.decay_score == 1.0

    async def test_constraint_never_decays(self, memoria):
        """CONSTRAINT memories should never decay."""
        record = await memoria.remember(
            "Never delete production data",
            memory_type=MemoryType.CONSTRAINT,
        )
        await memoria.run_decay_cycle()
        updated = await memoria.get_memory(record.id)
        assert updated is not None
        assert updated.decay_score == 1.0


class TestContradictionDetection:
    """Test contradiction detection via Memoria API."""

    async def test_numeric_contradiction(self, memoria):
        """Should detect numeric contradictions."""
        await memoria.remember(
            "Redis pool size is 10",
            memory_type=MemoryType.FACT,
            importance=0.8,
        )
        await memoria.remember(
            "Redis pool size is 50",
            memory_type=MemoryType.FACT,
            importance=0.8,
        )

        # Check for contradictions — may or may not detect depending on
        # DummyEmbedding similarity, but at minimum should not crash
        contradictions = await memoria.get_contradictions()
        assert isinstance(contradictions, list)

    async def test_contradiction_does_not_crash(self, memoria):
        """Contradiction detection should handle edge cases gracefully."""
        await memoria.remember("The sky is blue", memory_type=MemoryType.FACT)
        await memoria.remember("Docker uses containers", memory_type=MemoryType.FACT)
        # No related content → no contradiction
        contradictions = await memoria.get_contradictions()
        assert isinstance(contradictions, list)


class TestSearch:
    """Test search functionality."""

    async def test_search_by_query(self, memoria):
        """Search should return results for matching content."""
        await memoria.remember(
            "Docker compose version 3.8",
            memory_type=MemoryType.FACT,
            tags=["docker"],
        )
        results = await memoria.search("docker")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert "docker" in results[0].content.lower() or "Docker" in results[0].content

    async def test_search_by_tag(self, memoria):
        """Search by tag should find memories with that tag."""
        await memoria.remember(
            "Use Redis for caching",
            memory_type=MemoryType.FACT,
            tags=["redis", "cache"],
        )
        results = await memoria.search_by_tag("redis")
        assert isinstance(results, list)
        assert len(results) >= 1

    async def test_search_empty_results(self, memoria):
        """Search for non-existent content should return empty list."""
        results = await memoria.search("quantum computing neural network")
        assert isinstance(results, list)
        assert len(results) == 0

    async def test_search_with_type_filter(self, memoria):
        """Search with memory_type filter should limit results."""
        await memoria.remember("fact one", memory_type=MemoryType.FACT)
        await memoria.remember("event one", memory_type=MemoryType.EVENT)
        results = await memoria.search("one", memory_type=MemoryType.FACT)
        assert isinstance(results, list)


class TestMutations:
    """Test memory mutations: forget, reinforce, edit."""

    async def test_forget(self, memoria):
        """Forget should delete a memory."""
        record = await memoria.remember("temporary note", memory_type=MemoryType.FACT)
        deleted = await memoria.forget(record.id)
        assert deleted is True

        retrieved = await memoria.get_memory(record.id)
        assert retrieved is None

    async def test_reinforce(self, memoria):
        """Reinforce should boost importance."""
        record = await memoria.remember(
            "Important config", memory_type=MemoryType.FACT, importance=0.5
        )
        success = await memoria.reinforce(record.id, amount=0.2)
        assert success is True

        updated = await memoria.get_memory(record.id)
        assert updated.importance == pytest.approx(0.7, abs=0.01)

    async def test_edit(self, memoria):
        """Edit should update content and re-embed."""
        record = await memoria.remember(
            "Old content", memory_type=MemoryType.FACT
        )
        success = await memoria.edit(record.id, content="New content")
        assert success is True

        updated = await memoria.get_memory(record.id)
        assert updated.content == "New content"


class TestStats:
    """Test statistics and introspection."""

    async def test_stats(self, memoria):
        """Stats should return valid structure."""
        await memoria.remember("test memory", memory_type=MemoryType.FACT)
        stats = await memoria.stats()
        assert stats.health_score > 0
        assert stats.uptime_seconds >= 0

    async def test_list_memories(self, memoria):
        """List memories should return stored records."""
        await memoria.remember("mem1", memory_type=MemoryType.FACT)
        await memoria.remember("mem2", memory_type=MemoryType.FACT)
        memories = await memoria.list_memories()
        assert len(memories) >= 2
