"""Unit tests for individual engines."""

import pytest
from datetime import datetime, timezone, timedelta

from memoria.engines.decay import DecayEngine
from memoria.engines.feedback import FeedbackEngine, PersistentUsageTracker
from memoria.engines.orchestrator import EngineOrchestrator
from memoria.core.models import (
    ContextInjection,
    InjectItem,
    MemoryLayer,
    MemoryRecord,
    MemoryType,
)
from memoria.core.config import DecayConfig, FeedbackConfig
from memoria.storage.memory_adapter import InMemoryAdapter


@pytest.fixture
def storage():
    """Create a fresh InMemoryAdapter."""
    return InMemoryAdapter()


@pytest.fixture
def decay_engine(storage):
    """Create a DecayEngine with in-memory storage."""
    return DecayEngine(storage=storage, config=DecayConfig())


@pytest.fixture
def feedback_engine(storage):
    """Create a FeedbackEngine with in-memory storage."""
    return FeedbackEngine(storage=storage, config=FeedbackConfig())


class TestDecayEngine:
    """Test decay computation logic."""

    def test_preference_never_decays(self, decay_engine):
        """PREFERENCE memories always return decay_score = 1.0."""
        mem = MemoryRecord(memory_type=MemoryType.PREFERENCE, importance=0.5)
        score = decay_engine.compute_decay(mem)
        assert score == 1.0

    def test_constraint_never_decays(self, decay_engine):
        """CONSTRAINT memories always return decay_score = 1.0."""
        mem = MemoryRecord(memory_type=MemoryType.CONSTRAINT, importance=0.3)
        score = decay_engine.compute_decay(mem)
        assert score == 1.0

    def test_event_decays_fast(self, decay_engine):
        """EVENT memories should decay noticeably after 14 days (their half-life)."""
        mem = MemoryRecord(
            memory_type=MemoryType.EVENT,
            importance=0.5,
            last_accessed=datetime.now(timezone.utc) - timedelta(days=14),
        )
        score = decay_engine.compute_decay(mem)
        assert score < 0.8  # Should have decayed significantly

    def test_recent_memory_retains_high_score(self, decay_engine):
        """Recently accessed memory should have high decay score."""
        mem = MemoryRecord(
            memory_type=MemoryType.FACT,
            importance=0.7,
            last_accessed=datetime.now(timezone.utc),
        )
        score = decay_engine.compute_decay(mem)
        assert score > 0.7

    def test_skill_decays_slower_than_event(self, decay_engine):
        """SKILL (90-day half-life) should retain more than EVENT (14-day half-life)."""
        past = datetime.now(timezone.utc) - timedelta(days=30)
        event = MemoryRecord(memory_type=MemoryType.EVENT, last_accessed=past)
        skill = MemoryRecord(memory_type=MemoryType.SKILL, last_accessed=past)
        assert decay_engine.compute_decay(skill) > decay_engine.compute_decay(event)

    def test_fact_decays_slower_than_event(self, decay_engine):
        """FACT (30-day half-life) should retain more than EVENT (14-day half-life)."""
        past = datetime.now(timezone.utc) - timedelta(days=14)
        event = MemoryRecord(memory_type=MemoryType.EVENT, last_accessed=past)
        fact = MemoryRecord(memory_type=MemoryType.FACT, last_accessed=past)
        assert decay_engine.compute_decay(fact) > decay_engine.compute_decay(event)

    def test_high_importance_resists_decay(self, decay_engine):
        """Higher importance should yield higher decay score."""
        past = datetime.now(timezone.utc) - timedelta(days=20)
        low_imp = MemoryRecord(
            memory_type=MemoryType.FACT, importance=0.2, last_accessed=past
        )
        high_imp = MemoryRecord(
            memory_type=MemoryType.FACT, importance=0.9, last_accessed=past
        )
        assert decay_engine.compute_decay(high_imp) > decay_engine.compute_decay(low_imp)

    def test_access_bonus_helps_resist_decay(self, decay_engine):
        """Frequently accessed memories should decay slower."""
        past = datetime.now(timezone.utc) - timedelta(days=20)
        no_access = MemoryRecord(
            memory_type=MemoryType.FACT,
            last_accessed=past,
            access_count=0,
        )
        high_access = MemoryRecord(
            memory_type=MemoryType.FACT,
            last_accessed=past,
            access_count=50,
        )
        assert decay_engine.compute_decay(high_access) > decay_engine.compute_decay(no_access)

    def test_decay_score_bounded_0_to_1(self, decay_engine):
        """Decay score should always be in [0, 1]."""
        past = datetime.now(timezone.utc) - timedelta(days=365)
        mem = MemoryRecord(
            memory_type=MemoryType.EVENT,
            importance=0.1,
            last_accessed=past,
            access_count=0,
            decay_acceleration=2.0,
        )
        score = decay_engine.compute_decay(mem)
        assert 0.0 <= score <= 1.0


class TestAdaptiveInterval:
    """Test adaptive decay interval computation."""

    def test_small_memory_pool(self, decay_engine):
        """< 100 memories → 24 hour interval."""
        assert decay_engine.compute_adaptive_interval(50) == 24.0

    def test_medium_memory_pool(self, decay_engine):
        """100-1000 memories → 12 hour interval."""
        assert decay_engine.compute_adaptive_interval(500) == 12.0

    def test_large_memory_pool(self, decay_engine):
        """1000-5000 memories → 6 hour interval."""
        assert decay_engine.compute_adaptive_interval(2000) == 6.0

    def test_very_large_memory_pool(self, decay_engine):
        """5000-10000 memories → 3 hour interval."""
        assert decay_engine.compute_adaptive_interval(8000) == 3.0

    def test_huge_memory_pool(self, decay_engine):
        """> 10000 memories → 1 hour interval."""
        assert decay_engine.compute_adaptive_interval(20000) == 1.0

    def test_boundary_values(self, decay_engine):
        """Boundary memory counts."""
        assert decay_engine.compute_adaptive_interval(100) == 12.0
        assert decay_engine.compute_adaptive_interval(1000) == 6.0
        assert decay_engine.compute_adaptive_interval(5000) == 3.0
        assert decay_engine.compute_adaptive_interval(10000) == 1.0


class TestDecayCycleProcess:
    """Test the full decay cycle process."""

    async def test_decay_cycle_empty_store(self, decay_engine):
        """Decay cycle on empty store should succeed with empty stats."""
        report = await decay_engine.process_decay_cycle()
        assert report is not None
        assert "unchanged" in report.stats
        assert report.stats["unchanged"] == 0

    async def test_decay_cycle_with_memories(self, decay_engine, storage):
        """Decay cycle should process stored memories."""
        mem = MemoryRecord(
            id="mem_decay_test",
            content="test memory",
            memory_type=MemoryType.FACT,
            layer=MemoryLayer.WARM,
        )
        await storage.insert([mem])

        report = await decay_engine.process_decay_cycle()
        assert report is not None
        assert report.stats.get("processing_duration_ms", 0) >= 0


class TestFeedbackEngine:
    """Test feedback engine reinforcement and penalty."""

    async def test_access_boosts_importance(self, feedback_engine, storage):
        """Accessing a memory should increase its importance."""
        mem = MemoryRecord(
            id="mem_fb1", importance=0.5, content="test feedback"
        )
        await storage.insert([mem])

        await feedback_engine.on_memory_accessed("mem_fb1", "test context")
        updated = await storage.get("mem_fb1")
        assert updated.importance > 0.5

    async def test_access_increments_count(self, feedback_engine, storage):
        """Accessing a memory should increment access_count."""
        mem = MemoryRecord(
            id="mem_fb2", access_count=0, content="test count"
        )
        await storage.insert([mem])

        await feedback_engine.on_memory_accessed("mem_fb2", "context")
        updated = await storage.get("mem_fb2")
        assert updated.access_count == 1

    async def test_access_slows_decay(self, feedback_engine, storage):
        """Accessing should reduce decay_acceleration."""
        mem = MemoryRecord(
            id="mem_fb3", decay_acceleration=1.0, content="test decay"
        )
        await storage.insert([mem])

        await feedback_engine.on_memory_accessed("mem_fb3", "context")
        updated = await storage.get("mem_fb3")
        assert updated.decay_acceleration < 1.0

    async def test_ignore_accelerates_decay(self, feedback_engine, storage):
        """Ignoring a memory should increase decay_acceleration."""
        mem = MemoryRecord(
            id="mem_fb4", decay_acceleration=1.0, content="test ignore"
        )
        await storage.insert([mem])

        await feedback_engine.on_memory_ignored("mem_fb4")
        updated = await storage.get("mem_fb4")
        assert updated.decay_acceleration > 1.0

    async def test_ignore_caps_at_2(self, feedback_engine, storage):
        """decay_acceleration should not exceed 2.0."""
        mem = MemoryRecord(
            id="mem_fb5", decay_acceleration=1.95, content="test cap"
        )
        await storage.insert([mem])

        await feedback_engine.on_memory_ignored("mem_fb5")
        updated = await storage.get("mem_fb5")
        assert updated.decay_acceleration <= 2.0

    async def test_nonexistent_memory_no_crash(self, feedback_engine):
        """Operations on non-existent memory should not crash."""
        await feedback_engine.on_memory_accessed("nonexistent_id", "context")
        await feedback_engine.on_memory_ignored("nonexistent_id")


class TestPersistentUsageTracker:
    """Test the PersistentUsageTracker buffer."""

    async def test_buffer_accumulates(self, storage):
        """Buffer should accumulate access records."""
        tracker = PersistentUsageTracker(storage)
        tracker.record_access("mem_1", "ctx1")
        tracker.record_access("mem_1", "ctx2")
        assert tracker.buffer_size == 2

    async def test_needs_flush(self, storage):
        """needs_flush should trigger at threshold."""
        tracker = PersistentUsageTracker(storage)
        assert tracker.needs_flush is False
        for i in range(PersistentUsageTracker.FLUSH_THRESHOLD):
            tracker.record_access("mem_x", f"ctx_{i}")
        assert tracker.needs_flush is True

    async def test_flush_clears_buffer(self, storage):
        """Flush should clear the buffer."""
        mem = MemoryRecord(id="mem_flush", access_count=0, content="flush test")
        await storage.insert([mem])

        tracker = PersistentUsageTracker(storage)
        tracker.record_access("mem_flush", "ctx1")
        tracker.record_access("mem_flush", "ctx2")

        flushed = await tracker.flush()
        assert flushed == 2
        assert tracker.buffer_size == 0

        updated = await storage.get("mem_flush")
        assert updated.access_count == 2

    async def test_flush_empty_buffer(self, storage):
        """Flushing empty buffer should return 0."""
        tracker = PersistentUsageTracker(storage)
        flushed = await tracker.flush()
        assert flushed == 0


class TestOrchestrator:
    """Test EngineOrchestrator coordination logic."""

    async def test_filters_contradictions(self):
        """Post-process should filter memories with contradiction_of set."""
        orch = EngineOrchestrator()

        mem1 = MemoryRecord(id="m1", content="old info", contradiction_of="m2")
        mem2 = MemoryRecord(id="m2", content="new info")

        injection = ContextInjection(
            hot=[],
            relevant=[
                InjectItem(memory=mem1, relevance_score=0.8),
                InjectItem(memory=mem2, relevance_score=0.7),
            ],
            total_tokens=50,
        )

        result = await orch.post_process_context(injection, "test query")
        # mem1 should be filtered (has contradiction_of set)
        assert len(result.relevant) == 1
        assert result.relevant[0].memory.id == "m2"

    async def test_empty_relevant_passthrough(self):
        """Empty relevant list should pass through unchanged."""
        orch = EngineOrchestrator()
        injection = ContextInjection(
            hot=[MemoryRecord(id="hot1", content="always here")],
            relevant=[],
            total_tokens=10,
        )
        result = await orch.post_process_context(injection, "test")
        assert result.hot == injection.hot
        assert result.relevant == []

    async def test_preserves_hot_memories(self):
        """Post-process should preserve HOT memories unchanged."""
        orch = EngineOrchestrator()
        hot_mem = MemoryRecord(id="hot_pref", content="user pref")
        warm_mem = MemoryRecord(id="warm1", content="some fact")

        injection = ContextInjection(
            hot=[hot_mem],
            relevant=[InjectItem(memory=warm_mem, relevance_score=0.6)],
            total_tokens=20,
        )

        result = await orch.post_process_context(injection, "test")
        assert len(result.hot) == 1
        assert result.hot[0].id == "hot_pref"

    def test_should_protect_from_decay(self):
        """Frequently accessed memories should be protected."""
        orch = EngineOrchestrator()
        mem = MemoryRecord(id="m_protect", access_count=10, content="important")
        assert orch.should_protect_from_decay(mem) is True

    def test_should_not_protect_rarely_accessed(self):
        """Rarely accessed memories should not be protected."""
        orch = EngineOrchestrator()
        mem = MemoryRecord(
            id="m_no_protect",
            access_count=1,
            content="not important",
            last_accessed=datetime.now(timezone.utc) - timedelta(days=30),
        )
        assert orch.should_protect_from_decay(mem) is False

    def test_decay_modifier_high_access(self):
        """High access count should slow decay (modifier < 1.0)."""
        orch = EngineOrchestrator()
        mem = MemoryRecord(id="m_mod", access_count=15, content="test")
        modifier = orch.get_decay_modifier(mem)
        assert modifier < 1.0

    def test_decay_modifier_contradiction(self):
        """Contradicted memories should decay faster (modifier > 1.0)."""
        orch = EngineOrchestrator()
        mem = MemoryRecord(
            id="m_contra", content="old", contradiction_of="other_id"
        )
        modifier = orch.get_decay_modifier(mem)
        assert modifier > 1.0

    def test_coordination_stats(self):
        """Stats should reflect configured engines."""
        orch = EngineOrchestrator()
        stats = orch.get_coordination_stats()
        assert stats["has_awareness"] is False
        assert stats["has_decay"] is False
        assert stats["has_feedback"] is False
        assert stats["has_graph"] is False
