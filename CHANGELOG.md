# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Breaking changes

- `BinaryStoreAdapter` and `BinaryStoreConfig` are no longer exported from the
  core root (`@0xx0lostcause0xx0/polypack`). They now live behind platform
  persistence subpaths so the root entry stays free of `node:` built-ins:
  `@0xx0lostcause0xx0/polypack/persistence/node` and
  `@0xx0lostcause0xx0/polypack/persistence/opfs`.
- `VectorIndex.get()` / `entries()` and `HNSWIndex.get()` / `entries()` now
  return detached copies instead of sharing the internal `Float64Array`.
  Mutate vectors through the index methods only.

### Added

- `FileIO` abstraction with `NodeFileIO` (filesystem), `OPFSFileIO` (browser
  File System Access API), and `MemoryFileIO` (tests / custom storage). Pass a
  `fileIO` into `BinaryStoreAdapter` to plug in any backing store.
- `BinaryStoreAdapter` option `syncWrites` for fsync-guaranteed durability.
- `HNSWIndex.update()` for in-place vector replacement, plus physical node
  unlinking so removed ids can be re-added without stale topology.
- `npm run check:browser` and `npm run check:build:browser` verify that all
  root-reachable browser entry points are free of `node:` built-ins and bundle
  cleanly, while the node-only persistence subpath is rejected.
- `tests/browser-entry.test.ts` smoke-testing the browser-safe entry points
  under a browser-like environment.

### Fixed

- `BinaryStoreAdapter` now serialises startup, mutations, compaction, and
  shutdown through an internal queue, preventing lost writes during recovery
  or compaction and making `close()` idempotent.
- WAL recovery now persists a snapshot before deleting the replayed WAL, and
  tolerates a truncated WAL tail from a mid-append crash.

### Changed

- The stress suite now uses a fixed-seed PRNG for reproducible runs, asserts
  `recall@10` against the exact index (100K/4-dim and 5K/384-dim), reports
  p50/p95/p99 query latencies, and runs with `--expose-gc` so heap numbers are
  accurate.

### Added

- `crates/polypack-node` — NAPI-RS bindings exposing the Rust vector core
  (exact + HNSW) to Node, with a `packages/node-native` wrapper that provides
  drop-in `NativeVectorIndex`/`NativeHnswIndex` classes, a `createNativeVectorIndex`
  factory for `PolyGraph`'s non-breaking `createVectorIndex` hook, and
  `engineInfo()`/`isNativeAvailable()` diagnostics. Verified by
  `npm run test:native` against the same conformance fixtures.
- `crates/polypack-python` — PyO3 bindings (`polypack._core`) with NumPy
  input, GIL release around long operations, and the `errors.md` exception
  hierarchy. `python/polypack` ships a Python-native `PolyGraph` + fluent
  query API over the Rust vector core, passing the shared conformance fixtures
  (except hot-cache eviction, deferred) and benchmarked on the same seeded
  matrix.
- Rust persistence state machine (`crates/polypack-core/src/storage`):
  byte-compatible v1 snapshot/WAL codecs (verified byte-for-byte against the
  JS encoder), a host `Storage` byte-stream trait, and a `Store` mirroring
  `BinaryStoreAdapter` semantics (WAL replay then snapshot-before-delete,
  generation-boundary compaction, truncated-tail tolerance, durability modes,
  `format_version` checks). The shared recovery fixtures pass against it in
  Rust.
- Python filesystem persistence — `PolyGraph.open`/`save`/`close` over the
  Rust store via a `NativeStore` PyO3 binding (completing Python v1 scope),
  with WAL-recovery tests.
- Node native storage — a `NativeStore` NAPI-RS binding with a filesystem
  host adapter; cross-language byte round-trip tests prove the Rust store and
  the TypeScript `BinaryStoreAdapter` read each other's files.
- Rust query executor (`crates/polypack-core/src/query_exec.rs`): runs the
  shared query-plan IR over node/edge snapshots (filters, ranges, ordering,
  traversal, joins, pagination, exact cosine similarity, and opt-in HNSW via
  `similarity.engine`). The query-plan conformance fixtures pass against it in
  Rust.
- Query bindings — napi `executeQueryPlan`/`aggregateQueryPlan` and PyO3
  equivalents; Python `GraphQuery` delegates to the Rust executor.
- Deep TypeScript integration (gated): `GraphQuery.toArray` can route through
  the native executor via `installNativeQueryExecutor()` (opt-in; falls back
  for join predicates and without the binary). The measurement gate
  (`benchmarks/query-gate.md`) shows per-query FFI serialization makes native
  in-memory delegation ~46× slower at 50K nodes, so in-memory GraphQuery stays
  on TypeScript; the Rust executor serves Python and whole-store queries.
- Release hardening: per-platform native npm packages
  (`polypack-native-<triple>`) with `optionalDependencies`, an abi3 Python
  wheel build, a clean-venv 100K-vector example, cross-platform CI
  (macOS/Windows) for Rust, native, and Python, a `package` job running the
  release-candidate suite (clean Rust-free native install + clean-venv wheel
  install), and `RELEASING.md` documenting the coordinated versioning and
  release rules.

## [2.3.0] - 2026-07-20

### Breaking changes

- `IndexedDBAdapter` is replaced by `BinaryStoreAdapter` (MessagePack snapshot
  plus append-only WAL persistence). Existing browser IndexedDB data does not
  migrate automatically.

### Added

- `HNSWIndex` approximate nearest-neighbour vector index with `PolyGraph`
  factory integration, alongside the existing exact `VectorIndex`.
- `BinaryStoreAdapter` — durable MessagePack + WAL persistence across nodes,
  edges, and vectors.
- Default hot cache increased from 10K to 50K loaded nodes.
- Stress test suite measuring memory, vector search, HNSW, warm, and insert
  throughput, plus BinaryStoreAdapter WAL/snapshot/warm benchmarks at 10K/50K.

### Changed

- Performance: `Float64Array` sharing between `PolyNode` and `VectorIndex`,
  MemoryAdapter in-place flush and LRU eviction, edge-index `Map` refactor,
  partial warm (load only the hot cache), and a batched `getVectors` API.
- CI: stress tests run via `npm run test:stress` (with `npm run test:all` to
  include them); vitest uses a single fork with a raised timeout.
- Docs: release reference updated to v2.2.0.

### Fixed

- HNSW index type errors (`ArrayLike<number>` distance/search parameters).
- vitest RPC timeouts on stress tests via `singleFork` and larger timeouts.

## [2.2.0] - 2026-07-20

### Added

- `DataTransform` serialize/deserialize hooks for non-cloneable data (Blob,
  File, etc.) applied transparently across `addNode`, `updateNode`, `getNode`,
  and query results.
- Idempotent `warm()` — safe to call repeatedly.
- `defineEdges()`: typed frozen edge-constant utility.
- `buildEmbeddingText()`: weighted field repetition for feature-hash embeddings.
- `walkAncestors()` / `walkDescendants()`: linear traversal with cycle
  detection.
- `searchNodes()`: shorthand for `queryPersistedText` + `whereNodeType`.
- Node-type query helpers: `getNodesByType`, `getNodesByTypeOrdered`,
  `countNodesByType`, `deleteNodesByType`.

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
