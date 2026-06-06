"""Graph Engine — entity-relation knowledge graph for multi-hop reasoning.

Extracts entities and relationships from memories, builds a knowledge graph,
and enables graph-based queries like causal chains and multi-hop reasoning.
"""

from __future__ import annotations

import json
import re
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from memoria.core.config import GraphConfig
from memoria.core.models import MemoryRecord, MemoryType

logger = logging.getLogger(__name__)

# ── Known technology words for entity extraction ──────────────────────────────

_KNOWN_TECH_WORDS: Set[str] = {
    "python", "javascript", "typescript", "rust", "go", "java", "kotlin", "swift",
    "ruby", "php", "scala", "elixir", "haskell", "clojure", "lua", "perl",
    "docker", "kubernetes", "redis", "nginx", "apache", "postgres", "postgresql",
    "mysql", "mongodb", "sqlite", "elasticsearch", "kafka", "rabbitmq", "celery",
    "react", "vue", "angular", "svelte", "nextjs", "nuxt", "django", "flask",
    "fastapi", "express", "nestjs", "spring", "rails", "laravel",
    "git", "github", "gitlab", "bitbucket", "jenkins", "terraform", "ansible",
    "aws", "gcp", "azure", "heroku", "vercel", "netlify", "cloudflare",
    "linux", "ubuntu", "debian", "centos", "macos", "windows",
    "graphql", "grpc", "rest", "websocket", "http", "https",
    "pytest", "jest", "mocha", "cypress", "selenium",
    "numpy", "pandas", "scipy", "pytorch", "tensorflow", "keras",
    "pydantic", "sqlalchemy", "alembic", "celery", "dramatiq",
    "networkx", "lancedb", "qdrant", "neo4j", "prometheus", "grafana",
    "npm", "yarn", "pip", "conda", "homebrew", "cargo", "maven", "gradle",
    "vscode", "vim", "neovim", "emacs", "intellij", "pycharm",
    "openai", "anthropic", "langchain", "llamaindex", "huggingface",
}

# ── Uppercase abbreviation services ──────────────────────────────────────────

_SERVICE_ABBREVIATIONS: Set[str] = {
    "API", "SQL", "AWS", "GCP", "CLI", "SDK", "CDN", "DNS", "SSH", "TLS",
    "SSL", "JWT", "OAuth", "CORS", "CRUD", "ORM", "MVC", "MVP", "CI", "CD",
    "GPU", "CPU", "RAM", "SSD", "HDD", "LLM", "RAG", "NLP", "ML", "AI",
    "HTTP", "HTTPS", "TCP", "UDP", "FTP", "SMTP", "IMAP", "RPC", "REST",
    "JSON", "XML", "YAML", "TOML", "CSV", "HTML", "CSS", "DOM",
    "S3", "EC2", "RDS", "ECS", "EKS", "IAM", "VPC", "SQS", "SNS",
}

# ── Relation extraction patterns ─────────────────────────────────────────────

_RELATION_PATTERNS: List[Tuple[str, str]] = [
    (r"\buses(?:s|d)?\b|\busing\b", "uses"),
    (r"\bdepends?\s+on\b|\brequires?\b|\bneeds?\b", "depends_on"),
    (r"\bprefers?\b|\bchose\b|\bselected?\b|\bpicked?\b", "prefers"),
    (r"\bconfigured?\s+with\b|\bset\s+to\b|\bconfiguration\b", "configured_with"),
    (r"\breplaces?\b|\breplaced\b|\bmigrated?\s+from\b|\bswitch(?:ed)?\s+from\b", "replaces"),
    (r"\bcreated?\b|\bbuilt\b|\bwrote\b|\bdeveloped?\b", "created"),
    (r"\bintegrates?\s+with\b|\bconnects?\s+to\b", "integrates_with"),
    (r"\bextends?\b|\binherits?\s+from\b", "extends"),
    (r"\bimplements?\b", "implements"),
    (r"\bdeployed?\s+(?:on|to)\b|\bruns?\s+on\b", "deployed_on"),
]


@dataclass
class Entity:
    """An extracted entity from memory content."""

    id: str = field(default_factory=lambda: f"ent_{uuid.uuid4().hex[:8]}")
    name: str = ""
    entity_type: str = "unknown"  # person, technology, concept, tool, service, etc.
    memory_ids: List[str] = field(default_factory=list)  # memories mentioning this entity


@dataclass
class Relation:
    """A relationship between two entities."""

    id: str = field(default_factory=lambda: f"rel_{uuid.uuid4().hex[:8]}")
    source_entity_id: str = ""
    target_entity_id: str = ""
    relation_type: str = ""  # uses, depends_on, prefers, created, etc.
    memory_id: str = ""  # the memory this relation was extracted from
    confidence: float = 1.0


@dataclass
class GraphQueryResult:
    """Result of a graph query."""

    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    related_memory_ids: List[str] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)  # paths of entity IDs


class GraphEngine:
    """Knowledge graph engine for entity-relation reasoning.

    Manages an in-memory knowledge graph using NetworkX. Extracts entities and
    relationships from MemoryRecord instances using regex-based NER, builds a
    directed graph, and supports multi-hop traversal, causal chain reconstruction,
    and related-memory discovery.
    """

    def __init__(self, config: Optional[GraphConfig] = None) -> None:
        self._config = config or GraphConfig()
        self._graph = nx.DiGraph()
        self._entities: Dict[str, Entity] = {}  # entity_id → Entity
        self._entity_name_index: Dict[str, str] = {}  # normalized_name → entity_id
        self._relations: Dict[str, Relation] = {}  # relation_id → Relation
        self._memory_entities: Dict[str, List[str]] = {}  # memory_id → [entity_ids]

    # ── Core API ──────────────────────────────────────────────────────────────

    async def build_relations(self, memory: MemoryRecord) -> List[Relation]:
        """Extract entities and relations from a memory, update the graph.

        Args:
            memory: The memory record to process.

        Returns:
            List of newly created Relation objects.
        """
        text = memory.content
        memory_id = memory.id

        # Extract entities
        entities = self._extract_entities(text, memory.memory_type)

        # Link entities to the memory
        entity_ids: List[str] = []
        for ent in entities:
            existing = self._get_or_create_entity(ent.name, ent.entity_type)
            if memory_id not in existing.memory_ids:
                existing.memory_ids.append(memory_id)
            entity_ids.append(existing.id)

        self._memory_entities[memory_id] = entity_ids

        # Resolve entities to their canonical forms for relation extraction
        canonical_entities = [self._entities[eid] for eid in entity_ids]

        # Extract relations
        relations = self._extract_relations(text, canonical_entities, memory_id)

        # Auto-extract decision_for relations for DECISION type memories
        if memory.memory_type == MemoryType.DECISION and len(canonical_entities) >= 2:
            decision_rel = Relation(
                source_entity_id=canonical_entities[0].id,
                target_entity_id=canonical_entities[1].id,
                relation_type="decision_for",
                memory_id=memory_id,
                confidence=0.8,
            )
            relations.append(decision_rel)
            self._relations[decision_rel.id] = decision_rel

        # Add to the NetworkX graph
        self._add_to_graph(canonical_entities, relations)

        logger.debug(
            "Built %d entities, %d relations from memory %s",
            len(canonical_entities),
            len(relations),
            memory_id,
        )
        return relations

    async def query_graph(self, entity_name: str, hops: int = 3) -> GraphQueryResult:
        """Multi-hop traversal from a named entity.

        Args:
            entity_name: The name of the starting entity.
            hops: Maximum number of hops to traverse (default 3).

        Returns:
            GraphQueryResult with discovered entities, relations, memory IDs, and paths.
        """
        hops = min(hops, self._config.max_hops_default)
        result = GraphQueryResult()

        entity = self.get_entity(entity_name)
        if entity is None:
            return result

        # BFS traversal up to `hops` distance
        visited: Set[str] = set()
        frontier: Set[str] = {entity.id}
        all_paths: List[List[str]] = [[entity.id]]

        for _ in range(hops):
            next_frontier: Set[str] = set()
            new_paths: List[List[str]] = []

            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)

                # Explore outgoing and incoming edges
                if self._graph.has_node(node_id):
                    for neighbor in self._graph.successors(node_id):
                        if neighbor not in visited:
                            next_frontier.add(neighbor)
                            for path in all_paths:
                                if path[-1] == node_id:
                                    new_paths.append(path + [neighbor])

                    for neighbor in self._graph.predecessors(node_id):
                        if neighbor not in visited:
                            next_frontier.add(neighbor)
                            for path in all_paths:
                                if path[-1] == node_id:
                                    new_paths.append(path + [neighbor])

            all_paths.extend(new_paths)
            frontier = next_frontier

        # Include the last frontier in visited
        visited.update(frontier)

        # Collect results
        memory_ids_set: Set[str] = set()
        for eid in visited:
            if eid in self._entities:
                ent = self._entities[eid]
                result.entities.append(ent)
                memory_ids_set.update(ent.memory_ids)

        # Collect relations between visited entities
        for rel in self._relations.values():
            if rel.source_entity_id in visited and rel.target_entity_id in visited:
                result.relations.append(rel)

        result.related_memory_ids = list(memory_ids_set)
        result.paths = [p for p in all_paths if len(p) > 1]
        return result

    async def get_related_memories(self, memory_id: str, max_hops: int = 2) -> List[str]:
        """Get IDs of memories related through the graph.

        Args:
            memory_id: The source memory ID.
            max_hops: Maximum graph distance to traverse.

        Returns:
            List of related memory IDs (excluding the source).
        """
        entity_ids = self._memory_entities.get(memory_id, [])
        if not entity_ids:
            return []

        related_memory_ids: Set[str] = set()

        for eid in entity_ids:
            if not self._graph.has_node(eid):
                continue

            # BFS from this entity up to max_hops
            visited: Set[str] = set()
            frontier: Set[str] = {eid}

            for _ in range(max_hops):
                next_frontier: Set[str] = set()
                for node in frontier:
                    if node in visited:
                        continue
                    visited.add(node)
                    if self._graph.has_node(node):
                        for neighbor in self._graph.successors(node):
                            if neighbor not in visited:
                                next_frontier.add(neighbor)
                        for neighbor in self._graph.predecessors(node):
                            if neighbor not in visited:
                                next_frontier.add(neighbor)
                frontier = next_frontier

            visited.update(frontier)

            # Collect memory IDs from visited entities
            for visited_eid in visited:
                if visited_eid in self._entities:
                    related_memory_ids.update(self._entities[visited_eid].memory_ids)

        # Exclude the source memory
        related_memory_ids.discard(memory_id)
        return list(related_memory_ids)

    async def get_causal_chain(self, memory_id: str) -> List[str]:
        """Reconstruct a causal chain of decisions/events for a memory.

        Follows edges backward (predecessors) from entities associated with
        the given memory to find causal antecedents, then forward (successors)
        to find consequences.

        Args:
            memory_id: The memory to trace causality for.

        Returns:
            Ordered list of memory IDs forming the causal chain
            (antecedents → source → consequences).
        """
        entity_ids = self._memory_entities.get(memory_id, [])
        if not entity_ids:
            return []

        # Causal relation types
        causal_types = {"depends_on", "replaces", "decision_for", "created", "requires"}

        antecedent_mids: List[str] = []
        consequence_mids: List[str] = []

        for eid in entity_ids:
            if not self._graph.has_node(eid):
                continue

            # Walk backward: find predecessors connected by causal edges
            for pred in self._graph.predecessors(eid):
                edge_data = self._graph.get_edge_data(pred, eid)
                if edge_data and edge_data.get("relation_type") in causal_types:
                    if pred in self._entities:
                        for mid in self._entities[pred].memory_ids:
                            if mid != memory_id and mid not in antecedent_mids:
                                antecedent_mids.append(mid)

            # Walk forward: find successors connected by causal edges
            for succ in self._graph.successors(eid):
                edge_data = self._graph.get_edge_data(eid, succ)
                if edge_data and edge_data.get("relation_type") in causal_types:
                    if succ in self._entities:
                        for mid in self._entities[succ].memory_ids:
                            if mid != memory_id and mid not in consequence_mids:
                                consequence_mids.append(mid)

        # Chain: antecedents → source → consequences
        chain: List[str] = antecedent_mids + [memory_id] + consequence_mids
        return chain

    def get_entity(self, entity_name: str) -> Optional[Entity]:
        """Look up an entity by name.

        Args:
            entity_name: The entity name (case-insensitive lookup).

        Returns:
            The Entity if found, None otherwise.
        """
        normalized = self._normalize_entity_name(entity_name)
        entity_id = self._entity_name_index.get(normalized)
        if entity_id is None:
            return None
        return self._entities.get(entity_id)

    def get_stats(self) -> Dict[str, int]:
        """Return graph statistics.

        Returns:
            Dictionary with entity_count, relation_count, edge_count,
            memory_count, and connected_components.
        """
        return {
            "entity_count": len(self._entities),
            "relation_count": len(self._relations),
            "edge_count": self._graph.number_of_edges(),
            "memory_count": len(self._memory_entities),
            "connected_components": nx.number_weakly_connected_components(self._graph)
            if self._graph.number_of_nodes() > 0
            else 0,
        }

    # ── Entity/Relation Extraction ────────────────────────────────────────────

    def _extract_entities(self, text: str, memory_type: MemoryType) -> List[Entity]:
        """Extract entities from text using regex patterns.

        Patterns detected:
        - CamelCase words (e.g., PostgreSQL, LanceDB) → technology
        - Known technology words (docker, redis, etc.) → technology
        - Configuration keys (snake_case or dot.notation) → configuration
        - Uppercase abbreviations (API, SQL, AWS) → service
        - Version number patterns "Name X.Y" → technology
        - File/directory paths (/path/to/file) → resource

        Args:
            text: The text to extract entities from.
            memory_type: Type of memory (affects extraction priority).

        Returns:
            List of extracted Entity objects (may contain duplicates by name).
        """
        entities: List[Entity] = []
        seen_names: Set[str] = set()

        def _add(name: str, etype: str) -> None:
            normalized = self._normalize_entity_name(name)
            if normalized and normalized not in seen_names and len(normalized) > 1:
                seen_names.add(normalized)
                entities.append(Entity(name=name.strip(), entity_type=etype))

        # 1. CamelCase words (e.g., PostgreSQL, LanceDB, FastAPI)
        for match in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b", text):
            _add(match.group(1), "technology")

        # 2. Known technology words (case-insensitive)
        for match in re.finditer(r"\b([a-zA-Z][a-zA-Z0-9\-]*)\b", text):
            word = match.group(1)
            if word.lower() in _KNOWN_TECH_WORDS:
                _add(word, "technology")

        # 3. Version number patterns "Name X.Y.Z"
        for match in re.finditer(r"\b([A-Z][a-zA-Z]+)\s+(\d+(?:\.\d+)+)\b", text):
            name_with_version = f"{match.group(1)} {match.group(2)}"
            _add(name_with_version, "technology")

        # 4. Uppercase abbreviations (2+ capital letters)
        for match in re.finditer(r"\b([A-Z]{2,})\b", text):
            abbrev = match.group(1)
            if abbrev in _SERVICE_ABBREVIATIONS:
                _add(abbrev, "service")

        # 5. Configuration keys: dot.notation or long_snake_case
        for match in re.finditer(r"\b([a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*){2,})\b", text):
            _add(match.group(1), "configuration")
        for match in re.finditer(r"\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*){2,})\b", text):
            key = match.group(1)
            # Filter out common English phrases that happen to use underscores
            if not key.startswith(("this_", "that_", "the_")):
                _add(key, "configuration")

        # 6. File/directory paths
        for match in re.finditer(r"(/[a-zA-Z0-9._\-/]+(?:\.[a-zA-Z]{1,5})?)", text):
            path = match.group(1)
            if len(path) > 3 and "/" in path[1:]:  # at least /a/b
                _add(path, "resource")

        return entities

    def _extract_relations(
        self, text: str, entities: List[Entity], memory_id: str
    ) -> List[Relation]:
        """Extract relationships between entities from text.

        Uses pattern matching to detect relation types between entities
        that co-occur in the text.

        Relation patterns:
        - "X uses Y" → uses
        - "X depends on Y" → depends_on
        - "X prefers Y over Z" → prefers
        - "X is configured with Y" → configured_with
        - "chose X because/for Y" → decision_for
        - "X replaces/replaced Y" → replaces
        - "X requires Y" → requires

        Args:
            text: The source text.
            entities: List of entities found in the text.
            memory_id: ID of the source memory.

        Returns:
            List of extracted Relation objects.
        """
        if len(entities) < 2:
            return []

        relations: List[Relation] = []
        text_lower = text.lower()

        # For each pair of entities, check if a relation pattern exists
        for i, source_ent in enumerate(entities):
            for j, target_ent in enumerate(entities):
                if i == j:
                    continue

                # Check proximity: both entities should appear in the text
                source_pos = text_lower.find(source_ent.name.lower())
                target_pos = text_lower.find(target_ent.name.lower())

                if source_pos == -1 or target_pos == -1:
                    continue

                # Only consider pairs where source appears before target
                if source_pos >= target_pos:
                    continue

                # Extract the text between the two entity mentions
                between_start = source_pos + len(source_ent.name)
                between_end = target_pos
                between_text = text_lower[between_start:between_end]

                # Skip if too far apart (more than 100 chars between)
                if len(between_text) > 100:
                    continue

                # Match relation patterns in the between-text
                for pattern, rel_type in _RELATION_PATTERNS:
                    if re.search(pattern, between_text):
                        relation = Relation(
                            source_entity_id=source_ent.id,
                            target_entity_id=target_ent.id,
                            relation_type=rel_type,
                            memory_id=memory_id,
                            confidence=0.7,
                        )
                        relations.append(relation)
                        self._relations[relation.id] = relation
                        break  # Only one relation type per pair

        return relations

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for deduplication.

        Args:
            name: Raw entity name.

        Returns:
            Lowercased, stripped name with hyphens/spaces replaced by underscores.
        """
        return name.lower().strip().replace("-", "_").replace(" ", "_")

    def _get_or_create_entity(self, name: str, entity_type: str) -> Entity:
        """Get existing entity or create a new one.

        If an entity with the same normalized name exists, returns the existing one.
        Otherwise, creates a new entity and registers it.

        Args:
            name: The entity name.
            entity_type: The entity type classification.

        Returns:
            The existing or newly created Entity.
        """
        normalized = self._normalize_entity_name(name)
        existing_id = self._entity_name_index.get(normalized)

        if existing_id and existing_id in self._entities:
            return self._entities[existing_id]

        # Create new entity
        entity = Entity(name=name, entity_type=entity_type)
        self._entities[entity.id] = entity
        self._entity_name_index[normalized] = entity.id
        return entity

    # ── Graph Operations ──────────────────────────────────────────────────────

    def _add_to_graph(self, entities: List[Entity], relations: List[Relation]) -> None:
        """Add entities and relations to the NetworkX graph.

        Entities become nodes (with attributes), relations become directed edges.

        Args:
            entities: Entities to add as nodes.
            relations: Relations to add as edges.
        """
        # Add entity nodes
        for entity in entities:
            self._graph.add_node(
                entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
            )

        # Add relation edges
        for relation in relations:
            self._graph.add_edge(
                relation.source_entity_id,
                relation.target_entity_id,
                relation_id=relation.id,
                relation_type=relation.relation_type,
                memory_id=relation.memory_id,
                confidence=relation.confidence,
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    async def persist(self, path: str) -> None:
        """Serialize graph to a JSON file for persistence.

        Args:
            path: File path to write the serialized graph data.
        """
        data = {
            "entities": {
                eid: {
                    "name": e.name,
                    "type": e.entity_type,
                    "memory_ids": e.memory_ids,
                }
                for eid, e in self._entities.items()
            },
            "relations": [
                {
                    "id": r.id,
                    "source": r.source_entity_id,
                    "target": r.target_entity_id,
                    "type": r.relation_type,
                    "memory_id": r.memory_id,
                    "confidence": r.confidence,
                }
                for r in self._relations.values()
            ],
            "memory_entities": self._memory_entities,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Graph persisted to %s (%d entities, %d relations)", path, len(self._entities), len(self._relations))

    async def load(self, path: str) -> None:
        """Load graph from a persisted JSON file.

        Rebuilds the in-memory graph, entity index, and relation registry
        from a previously persisted file.

        Args:
            path: File path to load the serialized graph data from.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Clear current state
        self._graph.clear()
        self._entities.clear()
        self._entity_name_index.clear()
        self._relations.clear()
        self._memory_entities.clear()

        # Rebuild entities
        for eid, edata in data.get("entities", {}).items():
            entity = Entity(
                id=eid,
                name=edata["name"],
                entity_type=edata["type"],
                memory_ids=edata.get("memory_ids", []),
            )
            self._entities[eid] = entity
            self._entity_name_index[self._normalize_entity_name(entity.name)] = eid
            self._graph.add_node(
                eid,
                name=entity.name,
                entity_type=entity.entity_type,
            )

        # Rebuild relations
        for rdata in data.get("relations", []):
            relation = Relation(
                id=rdata["id"],
                source_entity_id=rdata["source"],
                target_entity_id=rdata["target"],
                relation_type=rdata["type"],
                memory_id=rdata.get("memory_id", ""),
                confidence=rdata.get("confidence", 1.0),
            )
            self._relations[relation.id] = relation
            self._graph.add_edge(
                relation.source_entity_id,
                relation.target_entity_id,
                relation_id=relation.id,
                relation_type=relation.relation_type,
                memory_id=relation.memory_id,
                confidence=relation.confidence,
            )

        # Rebuild memory-entity mapping
        self._memory_entities = data.get("memory_entities", {})

        logger.info(
            "Graph loaded from %s (%d entities, %d relations)",
            path,
            len(self._entities),
            len(self._relations),
        )
