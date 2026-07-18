# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [2.1.0] - 2026-07-18

### Breaking changes

- Public node, edge, query, and vector reads now return detached snapshots.
  Mutate graph state through `PolyGraph` and `VectorIndex` methods instead of
  editing returned objects in place.
- Node and edge mutation inputs are structured-cloned. Data must therefore be
  compatible with the platform `structuredClone` implementation.
- Invalid IDs, timestamps, vectors, ranges, pagination values, traversal depths,
  and top-K values now throw instead of producing ambiguous behavior.
- `similarTo()` excludes nodes without vectors, and `count()` respects
  similarity, traversal, offset, and limit.
- Edge source IDs and edge types may not contain the reserved `::` separator.
- The default IndexedDB schema version is now 2 and adds a node-type index.

### Added

- Pluggable synchronous or asynchronous text embedding providers, graph helpers
  for embedding-backed node mutations and queries, and a default normalized
  384-dimensional feature-hash embedding requiring no model or dependency.
- Safe asynchronous mutation of evicted nodes through `updateNodeSafe()`,
  `removeNodeSafe()`, and `removeNodeVectorSafe()`.
- Explicit loaded/persisted state APIs: `loadedSize`, `hasLoadedNode()`, and
  `persistedSize()`.
- `PersistedGraphQuery` for detached asynchronous queries across the complete
  backing store, including filters, similarity, joins, collection, traversal,
  ordering, pagination, counts, and IDs.
- Optional adapter query, count, edge lookup, and atomic `PersistenceChanges`
  hooks, with optimized Memory and IndexedDB implementations.
- Configurable IndexedDB node-data indexes and early-stopping cursor pagination.
- Reliable sync delivery with acknowledgements, retry, operation deduplication,
  pending-operation inspection, reconnect support, server cursors, gap detection,
  operation snapshots, and cursor-based delta recovery.
- `removeNodeVector()` for explicitly clearing a vector while retaining its node.
- Optional React `useGraphQuery` error callback and mounted lifecycle/race tests.

### Fixed

- Owned cascade deletion and updates for nodes outside the loaded working set.
- Ambiguous persisted edge-key reconstruction.
- Stale React node-type filters and debounce delays.
- Vectorless similarity results, paginated counts, and non-numeric aggregation.
- Disconnected sync-client retention and duplicate sync operation application.
- Cross-store partial persistence through atomic Memory/IndexedDB commits and
  complete dirty-batch retry after failure.

### Migration notes

- Replace direct mutations such as `graph.getNode(id)!.data.x = value` with
  `graph.updateNode(id, { x: value })`.
- Replace direct vector clearing with `graph.removeNodeVector(id)`.
- Custom persistence adapters remain compatible. Implement `applyChanges`,
  `queryNodes`, `countNodes`, and indexed edge lookup hooks for stronger atomicity
  and persisted-query performance.
- Custom-named IndexedDB databases that add `nodeIndexes` must increment their
  configured schema version so the browser runs `onupgradeneeded`.

## [1.1.0] - 2026-07-17

### Added

- Property graph queries, vector similarity search, ownership-aware edges,
  pluggable persistence, React hooks, and transport-agnostic synchronization.
- Public API reference, release metadata, native Node.js ESM support, and npm
  subpath exports for core, React, and sync entry points.

### Fixed

- Persisted edge cleanup during node deletion.
- Concurrent flush and shutdown data-loss paths.
- Node/vector index consistency during replacement, restoration, and eviction.
- IndexedDB bulk-read and vector top-k performance.

[1.1.0]: https://github.com/imattau/polypack/releases/tag/v1.1.0
[2.1.0]: https://github.com/imattau/polypack/compare/v1.1.0...v2.1.0
