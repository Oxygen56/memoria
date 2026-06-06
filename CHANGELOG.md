# Changelog

## [0.2.0] - 2024-06-07

### Added
- GraphEngine with NetworkX (entity/relation extraction, multi-hop BFS)
- EngineOrchestrator (contradiction filtering, graph boost, decay protection)
- ContradictionProvider (LLM/CrossEncoder/Heuristic/Fallback modes)
- PgVectorAdapter with asyncpg connection pooling
- CachedEmbeddingProvider with LRU cache and exponential backoff
- PersistentUsageTracker with buffer/flush persistence
- Adaptive decay interval based on memory count
- Background decay scheduling with asyncio
- tiktoken-based precise token counting
- Configurable RRF fusion parameters
- CLI tool (memoria remember/recall/stats)
- Comprehensive test suite (89+ tests)

### Fixed
- Default storage backend changed from LanceDB to InMemory for zero-config startup
- OpenAI embedding provider now accepts optional dimensions parameter

## [0.1.0] - 2024-01-01

### Added
- Initial release with Awareness, Decay, and Feedback engines
- InMemory storage adapter
- Basic remember/recall API
