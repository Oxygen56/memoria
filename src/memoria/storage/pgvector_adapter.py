"""PostgreSQL + pgvector storage adapter for production deployments.

Requires: asyncpg, pgvector
Install: pip install memoria-agent[pgvector]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from memoria.core.models import MemoryLayer, MemoryRecord, MemoryType, StorageStats
from memoria.storage.base import MemoryStoreAdapter

logger = logging.getLogger(__name__)


class PgVectorAdapter(MemoryStoreAdapter):
    """PostgreSQL + pgvector storage adapter.

    Table schema:
        CREATE TABLE IF NOT EXISTS memories (
            id VARCHAR(32) PRIMARY KEY,
            content TEXT NOT NULL,
            memory_type VARCHAR(20) NOT NULL,
            layer VARCHAR(20) NOT NULL DEFAULT 'warm',
            importance FLOAT NOT NULL DEFAULT 0.5,
            decay_score FLOAT NOT NULL DEFAULT 1.0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_modified TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            access_count INTEGER NOT NULL DEFAULT 0,
            half_life_days FLOAT NOT NULL DEFAULT 30.0,
            decay_acceleration FLOAT NOT NULL DEFAULT 1.0,
            related_memories JSONB DEFAULT '[]',
            tags JSONB DEFAULT '[]',
            custom_metadata JSONB DEFAULT '{}',
            embedding vector(1536),
            user_id VARCHAR(100),
            session_id VARCHAR(100),
            source_conversation_id VARCHAR(100),
            contradiction_of VARCHAR(32),
            created_by VARCHAR(50) DEFAULT 'agent',
            version INTEGER DEFAULT 1
        );

    Usage:
        adapter = PgVectorAdapter(
            host="localhost", port=5432,
            database="memoria", user="postgres", password="pass"
        )
        await adapter.initialize()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "memoria",
        user: str = "postgres",
        password: str = "",
        table_name: str = "memories",
        pool_min_size: int = 2,
        pool_max_size: int = 10,
        vector_dimensions: int = 1536,
        **kwargs: Any,
    ):
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._table_name = table_name
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._vector_dimensions = vector_dimensions
        self._pool = None

    # ── Lifecycle ─────────────────────────────────────

    async def initialize(self) -> None:
        """Create connection pool and ensure table/indexes exist."""
        import asyncpg

        self._pool = await asyncpg.create_pool(
            host=self._host,
            port=self._port,
            database=self._database,
            user=self._user,
            password=self._password,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
        )

        async with self._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id VARCHAR(32) PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type VARCHAR(20) NOT NULL,
                    layer VARCHAR(20) NOT NULL DEFAULT 'warm',
                    importance FLOAT NOT NULL DEFAULT 0.5,
                    decay_score FLOAT NOT NULL DEFAULT 1.0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_modified TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    access_count INTEGER NOT NULL DEFAULT 0,
                    half_life_days FLOAT NOT NULL DEFAULT 30.0,
                    decay_acceleration FLOAT NOT NULL DEFAULT 1.0,
                    related_memories JSONB DEFAULT '[]'::jsonb,
                    tags JSONB DEFAULT '[]'::jsonb,
                    custom_metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding vector({self._vector_dimensions}),
                    user_id VARCHAR(100),
                    session_id VARCHAR(100),
                    source_conversation_id VARCHAR(100),
                    contradiction_of VARCHAR(32),
                    created_by VARCHAR(50) DEFAULT 'agent',
                    version INTEGER DEFAULT 1
                )
            """)

            # Create indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_layer
                ON {self._table_name}(layer)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_type
                ON {self._table_name}(memory_type)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_tags
                ON {self._table_name} USING GIN(tags)
            """)

            # IVFFlat index for vector similarity — only create if table has rows
            row_count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self._table_name}"
            )
            if row_count and row_count > 0:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_embedding
                    ON {self._table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)

        logger.info("PgVectorAdapter initialized: %s:%d/%s", self._host, self._port, self._database)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ── Core methods ──────────────────────────────────

    async def insert(self, records: List[MemoryRecord]) -> List[str]:
        """Batch insert memories with ON CONFLICT DO NOTHING."""
        if not records:
            return []

        inserted_ids: List[str] = []
        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            for record in records:
                row = self._record_to_row(record)
                await conn.execute(
                    f"""
                    INSERT INTO {self._table_name} (
                        id, content, memory_type, layer, importance, decay_score,
                        created_at, last_accessed, last_modified, access_count,
                        half_life_days, decay_acceleration, related_memories, tags,
                        custom_metadata, embedding, user_id, session_id,
                        source_conversation_id, contradiction_of, created_by, version
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    row["id"], row["content"], row["memory_type"], row["layer"],
                    row["importance"], row["decay_score"], row["created_at"],
                    row["last_accessed"], row["last_modified"], row["access_count"],
                    row["half_life_days"], row["decay_acceleration"],
                    row["related_memories"], row["tags"], row["custom_metadata"],
                    row["embedding"], row["user_id"], row["session_id"],
                    row["source_conversation_id"], row["contradiction_of"],
                    row["created_by"], row["version"],
                )
                inserted_ids.append(record.id)

        return inserted_ids

    async def search_semantic(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """Vector similarity search using pgvector cosine distance."""
        filter_clause, params = self._build_filter_clause(filters)
        param_idx = len(params) + 1

        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        query = f"""
            SELECT *, 1 - (embedding <=> ${param_idx}::vector) as similarity
            FROM {self._table_name}
            WHERE embedding IS NOT NULL {filter_clause}
            ORDER BY embedding <=> ${param_idx}::vector
            LIMIT ${param_idx + 1}
        """
        params.extend([embedding_str, top_k])

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            rows = await conn.fetch(query, *params)

        return [self._row_to_record(row) for row in rows]

    async def search_keyword(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """Full-text search using PostgreSQL ts_vector."""
        filter_clause, params = self._build_filter_clause(filters)
        param_idx = len(params) + 1

        sql = f"""
            SELECT *,
                   ts_rank(to_tsvector('english', content), plainto_tsquery('english', ${param_idx})) as rank
            FROM {self._table_name}
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', ${param_idx})
                  {filter_clause}
            ORDER BY rank DESC
            LIMIT ${param_idx + 1}
        """
        params.extend([query, top_k])

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            rows = await conn.fetch(sql, *params)

        return [self._row_to_record(row) for row in rows]

    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Partial update using dynamic SET clause."""
        if not updates:
            return False

        set_parts: List[str] = []
        params: List[Any] = []
        idx = 1

        for key, value in updates.items():
            if key in ("id",):
                continue  # Never update primary key
            if key in ("related_memories", "tags", "custom_metadata"):
                set_parts.append(f"{key} = ${idx}::jsonb")
                params.append(json.dumps(value))
            elif key == "embedding":
                set_parts.append(f"{key} = ${idx}::vector")
                params.append("[" + ",".join(str(v) for v in value) + "]" if value else None)
            else:
                set_parts.append(f"{key} = ${idx}")
                params.append(value)
            idx += 1

        if not set_parts:
            return False

        # Always update last_modified
        set_parts.append(f"last_modified = ${idx}")
        params.append(datetime.now(timezone.utc))
        idx += 1

        params.append(memory_id)
        sql = f"""
            UPDATE {self._table_name}
            SET {', '.join(set_parts)}
            WHERE id = ${idx}
        """

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            result = await conn.execute(sql, *params)
            return result != "UPDATE 0"

    async def delete(self, memory_ids: List[str]) -> int:
        """Batch delete by IDs."""
        if not memory_ids:
            return 0

        placeholders = ", ".join(f"${i+1}" for i in range(len(memory_ids)))
        sql = f"DELETE FROM {self._table_name} WHERE id IN ({placeholders})"

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            result = await conn.execute(sql, *memory_ids)
            # result format: "DELETE N"
            return int(result.split()[-1])

    # ── Extended methods ──────────────────────────────

    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Get a single memory by ID."""
        sql = f"SELECT * FROM {self._table_name} WHERE id = $1"

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            row = await conn.fetchrow(sql, memory_id)

        if row is None:
            return None
        return self._row_to_record(row)

    async def list_by_layer(
        self, layer: MemoryLayer, limit: int = 1000, offset: int = 0
    ) -> List[MemoryRecord]:
        """List memories in a specific layer."""
        sql = f"""
            SELECT * FROM {self._table_name}
            WHERE layer = $1
            ORDER BY last_modified DESC
            LIMIT $2 OFFSET $3
        """

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            rows = await conn.fetch(sql, layer.value, limit, offset)

        return [self._row_to_record(row) for row in rows]

    async def get_stats(self) -> StorageStats:
        """Storage statistics."""
        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            total = await conn.fetchval(f"SELECT COUNT(*) FROM {self._table_name}")

            layer_rows = await conn.fetch(
                f"SELECT layer, COUNT(*) as cnt FROM {self._table_name} GROUP BY layer"
            )
            by_layer = {row["layer"]: row["cnt"] for row in layer_rows}

            type_rows = await conn.fetch(
                f"SELECT memory_type, COUNT(*) as cnt FROM {self._table_name} GROUP BY memory_type"
            )
            by_type = {row["memory_type"]: row["cnt"] for row in type_rows}

            size = await conn.fetchval(
                f"SELECT pg_total_relation_size('{self._table_name}')"
            )

        return StorageStats(
            total_memories=total or 0,
            by_layer=by_layer,
            by_type=by_type,
            storage_size_bytes=size or 0,
            backend_type="pgvector",
        )

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count memories matching filters."""
        filter_clause, params = self._build_filter_clause(filters)
        sql = f"SELECT COUNT(*) FROM {self._table_name} WHERE 1=1 {filter_clause}"

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            result = await conn.fetchval(sql, *params)

        return result or 0

    # ── Helpers ───────────────────────────────────────

    def _record_to_row(self, record: MemoryRecord) -> Dict[str, Any]:
        """Convert MemoryRecord to DB row dict."""
        embedding_str: Optional[str] = None
        if record.embedding:
            embedding_str = "[" + ",".join(str(v) for v in record.embedding) + "]"

        return {
            "id": record.id,
            "content": record.content,
            "memory_type": record.memory_type.value,
            "layer": record.layer.value,
            "importance": record.importance,
            "decay_score": record.decay_score,
            "created_at": record.created_at,
            "last_accessed": record.last_accessed,
            "last_modified": record.last_modified,
            "access_count": record.access_count,
            "half_life_days": record.half_life_days,
            "decay_acceleration": record.decay_acceleration,
            "related_memories": json.dumps(record.related_memories),
            "tags": json.dumps(record.tags),
            "custom_metadata": json.dumps(record.custom_metadata),
            "embedding": embedding_str,
            "user_id": record.user_id,
            "session_id": record.session_id,
            "source_conversation_id": record.source_conversation_id,
            "contradiction_of": record.contradiction_of,
            "created_by": record.created_by,
            "version": record.version,
        }

    def _row_to_record(self, row: Any) -> MemoryRecord:
        """Convert DB row (asyncpg Record) to MemoryRecord."""
        # Parse JSONB fields
        related = row["related_memories"]
        if isinstance(related, str):
            related = json.loads(related)

        tags = row["tags"]
        if isinstance(tags, str):
            tags = json.loads(tags)

        custom_meta = row["custom_metadata"]
        if isinstance(custom_meta, str):
            custom_meta = json.loads(custom_meta)

        # Parse embedding from pgvector string format "[0.1,0.2,...]"
        embedding = None
        raw_embedding = row.get("embedding")
        if raw_embedding is not None:
            if isinstance(raw_embedding, str):
                embedding = [float(v) for v in raw_embedding.strip("[]").split(",")]
            elif isinstance(raw_embedding, (list, tuple)):
                embedding = list(raw_embedding)

        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            layer=MemoryLayer(row["layer"]),
            importance=row["importance"],
            decay_score=row["decay_score"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            last_modified=row["last_modified"],
            access_count=row["access_count"],
            half_life_days=row["half_life_days"],
            decay_acceleration=row["decay_acceleration"],
            related_memories=related or [],
            tags=tags or [],
            custom_metadata=custom_meta or {},
            embedding=embedding,
            user_id=row["user_id"],
            session_id=row["session_id"],
            source_conversation_id=row["source_conversation_id"],
            contradiction_of=row["contradiction_of"],
            created_by=row["created_by"],
            version=row["version"],
        )

    def _build_filter_clause(
        self, filters: Optional[Dict[str, Any]]
    ) -> Tuple[str, List[Any]]:
        """Build WHERE clause from filters dict. Returns (clause_str, params).

        Supports:
            - Simple equality: {"layer": "hot"}
            - $in operator: {"layer": {"$in": ["hot", "warm"]}}
            - Tags containment: {"tags": ["tag1", "tag2"]} (uses @> operator)
        """
        if not filters:
            return "", []

        clauses: List[str] = []
        params: List[Any] = []
        idx = 1

        for key, value in filters.items():
            if isinstance(value, dict):
                # Operator-based filter
                if "$in" in value:
                    in_values = value["$in"]
                    placeholders = ", ".join(f"${idx + i}" for i in range(len(in_values)))
                    clauses.append(f"AND {key} IN ({placeholders})")
                    params.extend(in_values)
                    idx += len(in_values)
            elif key == "tags" and isinstance(value, list):
                # JSONB array containment
                clauses.append(f"AND tags @> ${idx}::jsonb")
                params.append(json.dumps(value))
                idx += 1
            else:
                # Simple equality
                clauses.append(f"AND {key} = ${idx}")
                params.append(value)
                idx += 1

        return " " + " ".join(clauses), params
