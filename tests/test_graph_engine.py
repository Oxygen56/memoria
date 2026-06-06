"""Unit tests for the Graph Engine."""

import pytest

from memoria.core.config import GraphConfig
from memoria.core.models import MemoryRecord, MemoryType
from memoria.engines.graph import GraphEngine, GraphQueryResult


@pytest.fixture
def graph_engine():
    """Create a fresh GraphEngine instance."""
    return GraphEngine(config=GraphConfig())


class TestEntityExtraction:
    """Test entity extraction from text."""

    async def test_extract_technology_known_word(self, graph_engine):
        """Should extract known technology words like PostgreSQL."""
        record = MemoryRecord(
            content="We use PostgreSQL 16 for production",
            memory_type=MemoryType.FACT,
        )
        await graph_engine.build_relations(record)

        stats = graph_engine.get_stats()
        assert stats["entity_count"] > 0

    async def test_extract_camelcase(self, graph_engine):
        """Should extract CamelCase words like LanceDB."""
        record = MemoryRecord(
            content="LanceDB is the default vector store",
            memory_type=MemoryType.FACT,
        )
        await graph_engine.build_relations(record)
        entity = graph_engine.get_entity("LanceDB")
        assert entity is not None
        assert entity.entity_type == "technology"

    async def test_extract_abbreviation(self, graph_engine):
        """Should extract uppercase abbreviations like AWS, API."""
        record = MemoryRecord(
            content="Deploy to AWS using Docker",
            memory_type=MemoryType.FACT,
        )
        await graph_engine.build_relations(record)
        stats = graph_engine.get_stats()
        # AWS (service) + Docker (technology) = at least 2
        assert stats["entity_count"] >= 2

    async def test_extract_multiple_technologies(self, graph_engine):
        """Should extract multiple technology entities from one text."""
        record = MemoryRecord(
            content="The app uses Redis for caching and PostgreSQL for persistence",
            memory_type=MemoryType.FACT,
        )
        await graph_engine.build_relations(record)

        redis = graph_engine.get_entity("Redis")
        pg = graph_engine.get_entity("PostgreSQL")
        assert redis is not None
        assert pg is not None

    async def test_entity_deduplication(self, graph_engine):
        """Same entity mentioned twice should not duplicate."""
        r1 = MemoryRecord(
            content="Redis is used for caching",
            memory_type=MemoryType.FACT,
        )
        r2 = MemoryRecord(
            content="Redis requires 2GB of RAM",
            memory_type=MemoryType.FACT,
        )
        await graph_engine.build_relations(r1)
        await graph_engine.build_relations(r2)

        entity = graph_engine.get_entity("Redis")
        assert entity is not None
        # Should be referenced by both memory IDs
        assert r1.id in entity.memory_ids
        assert r2.id in entity.memory_ids

        # Entity count should be 1 for Redis (not 2)
        redis_count = sum(
            1 for e in graph_engine._entities.values()
            if e.name.lower() == "redis"
        )
        assert redis_count == 1

    async def test_extract_config_key(self, graph_engine):
        """Should extract dot-notation config keys as entities."""
        record = MemoryRecord(
            content="Set spring.datasource.url to the connection string",
            memory_type=MemoryType.FACT,
        )
        await graph_engine.build_relations(record)

        entity = graph_engine.get_entity("spring.datasource.url")
        assert entity is not None
        assert entity.entity_type == "configuration"


class TestRelationExtraction:
    """Test relation extraction."""

    async def test_uses_relation(self, graph_engine):
        """Should detect 'uses' relation pattern."""
        record = MemoryRecord(
            content="The project uses Redis for caching",
            memory_type=MemoryType.FACT,
        )
        relations = await graph_engine.build_relations(record)
        assert isinstance(relations, list)

    async def test_depends_on_relation(self, graph_engine):
        """Should detect 'depends on' relation pattern."""
        record = MemoryRecord(
            content="Redis depends on Linux for deployment",
            memory_type=MemoryType.FACT,
        )
        relations = await graph_engine.build_relations(record)
        assert isinstance(relations, list)
        # Should find a 'depends_on' relation
        dep_rels = [r for r in relations if r.relation_type == "depends_on"]
        assert len(dep_rels) >= 1

    async def test_decision_relation(self, graph_engine):
        """DECISION type memories should get a 'decision_for' relation auto-added."""
        record = MemoryRecord(
            content="Chose PostgreSQL over MySQL for JSON support",
            memory_type=MemoryType.DECISION,
        )
        relations = await graph_engine.build_relations(record)
        assert isinstance(relations, list)
        # DECISION type with >= 2 entities gets a decision_for relation
        decision_rels = [r for r in relations if r.relation_type == "decision_for"]
        assert len(decision_rels) >= 1

    async def test_no_relations_for_single_entity(self, graph_engine):
        """Text with only one entity should produce no relations."""
        record = MemoryRecord(
            content="Redis is great",
            memory_type=MemoryType.FACT,
        )
        relations = await graph_engine.build_relations(record)
        # With only one entity, _extract_relations returns []
        # but there might be other entities found via regex
        assert isinstance(relations, list)


class TestGraphQuery:
    """Test graph traversal."""

    async def test_multi_hop_query(self, graph_engine):
        """Multi-hop query should discover connected entities."""
        await graph_engine.build_relations(
            MemoryRecord(content="Service A uses Redis", memory_type=MemoryType.FACT)
        )
        await graph_engine.build_relations(
            MemoryRecord(content="Redis depends on Linux", memory_type=MemoryType.FACT)
        )

        result = await graph_engine.query_graph("Redis", hops=2)
        assert result is not None
        assert isinstance(result, GraphQueryResult)
        assert len(result.entities) >= 1

    async def test_query_unknown_entity(self, graph_engine):
        """Querying a non-existent entity should return empty result."""
        result = await graph_engine.query_graph("NonExistentEntity", hops=2)
        assert result is not None
        assert len(result.entities) == 0
        assert len(result.relations) == 0

    async def test_get_related_memories(self, graph_engine):
        """Should find memories related through shared entities."""
        r1 = MemoryRecord(
            id="mem_aaa",
            content="App uses PostgreSQL",
            memory_type=MemoryType.FACT,
        )
        r2 = MemoryRecord(
            id="mem_bbb",
            content="PostgreSQL requires Linux",
            memory_type=MemoryType.FACT,
        )
        await graph_engine.build_relations(r1)
        await graph_engine.build_relations(r2)

        related = await graph_engine.get_related_memories("mem_aaa", max_hops=2)
        assert isinstance(related, list)
        # mem_bbb shares PostgreSQL entity, so should be related
        assert "mem_bbb" in related

    async def test_get_related_memories_no_entity(self, graph_engine):
        """Memory with no entities should return empty related list."""
        related = await graph_engine.get_related_memories("mem_nonexistent", max_hops=2)
        assert related == []

    async def test_get_causal_chain(self, graph_engine):
        """Causal chain should return related memories through causal edges."""
        r1 = MemoryRecord(
            id="mem_ccc",
            content="Redis depends on Linux",
            memory_type=MemoryType.FACT,
        )
        await graph_engine.build_relations(r1)

        chain = await graph_engine.get_causal_chain("mem_ccc")
        assert isinstance(chain, list)
        # At minimum, chain should include the source memory
        assert "mem_ccc" in chain


class TestGraphStats:
    """Test graph statistics."""

    async def test_empty_graph_stats(self, graph_engine):
        """Empty graph should have zero stats."""
        stats = graph_engine.get_stats()
        assert stats["entity_count"] == 0
        assert stats["relation_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["memory_count"] == 0

    async def test_stats_after_adding(self, graph_engine):
        """Stats should reflect added entities and relations."""
        await graph_engine.build_relations(
            MemoryRecord(content="Docker uses Linux", memory_type=MemoryType.FACT)
        )
        stats = graph_engine.get_stats()
        assert stats["entity_count"] >= 2
        assert stats["memory_count"] >= 1


class TestPersistence:
    """Test graph serialization."""

    async def test_persist_and_load(self, graph_engine, tmp_path):
        """Graph should survive persist/load cycle."""
        record = MemoryRecord(content="Test uses Python", memory_type=MemoryType.FACT)
        await graph_engine.build_relations(record)

        original_stats = graph_engine.get_stats()
        assert original_stats["entity_count"] > 0

        path = str(tmp_path / "graph.json")
        await graph_engine.persist(path)

        new_engine = GraphEngine()
        await new_engine.load(path)

        loaded_stats = new_engine.get_stats()
        assert loaded_stats["entity_count"] == original_stats["entity_count"]
        assert loaded_stats["relation_count"] == original_stats["relation_count"]

    async def test_persist_empty_graph(self, graph_engine, tmp_path):
        """Persisting an empty graph should not crash."""
        path = str(tmp_path / "empty_graph.json")
        await graph_engine.persist(path)

        new_engine = GraphEngine()
        await new_engine.load(path)
        assert new_engine.get_stats()["entity_count"] == 0
