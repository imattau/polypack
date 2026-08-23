# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [3.3.2] - 2026-08-23

### Fixed

- `Store::validate_pending_indexes` cloned the entire node map on every
  `apply()` call even when no unique secondary index was defined, an
  O(store size) cost per write. Added the same early-return guard its
  sibling `validate_pending_schema` already had for the no-op case.
- `Store::apply_with_identity`'s idempotency check re-read and re-parsed the
  entire on-disk mutation log on every identity-bearing `apply()` — which
  `Graph::transaction()` always triggers, since it synthesizes a
  transaction id — turning transaction cost into O(total historical
  mutation count) per call. Replaced with an in-memory identity cache kept
  in sync with the durable log.

## [3.3.1] - 2026-08-23

### Fixed

- `Graph.supersede()` wrote the `SUPERSEDED_BY` edge backwards (from the new
  node to the old one), so graph traversal read the relationship as "the new
  node is superseded by the old one" instead of the reverse. Fixed identically
  across the Rust, TypeScript, and Python implementations.

## [3.3.0] - 2026-08-23

### Added

- `HnswIndex` supports Euclidean distance across all three bindings
  (`HnswConfig.distance` in Rust, a `distance` option in the native Node
  binding, and a `distance` parameter in Python's `HnswIndex`), matching the
  TypeScript `HNSWIndex`'s existing pluggable `distanceFn` and closing a gap
  where the Rust core silently only supported cosine.
- `HnswIndex::query_with_ef_search` lets callers search with a different
  `ef_search` than the index was built with, without rebuilding — enables
  tracing a recall/speed curve against one built graph.
- New benchmarks: `bench:database-core:edge-flush:rust` (isolates edge-heavy
  flush cost), `bench:database-core:batches:napi` (exercises the real napi
  binding rather than a standalone Rust process), and a comparative HNSW
  benchmark against the public ann-benchmarks `sift-128-euclidean` dataset
  (`crates/polypack-core/examples/ann_sift_bench.rs`,
  `benchmarks/ann-sift-report.md`).

### Fixed

- `Graph::flush()` resolved each dirty edge id by scanning every source's
  edge map — O(dirty_edges × total_edges) per flush. Fixed with a persistent
  `edge_id_index`, also removing three other linear id-scans in
  `crates/polypack-graph/src/graph.rs`.
- `InMemoryStorage::append()` cloned the entire accumulated file on every
  append (O(n²) over n flushes). Now extends in place.
- `Store::validate_pending_schema` deep-cloned and rescanned every node and
  edge in the store on every single `apply()`/flush, even when no schema
  was registered. Now short-circuits when no node/edge type is registered —
  this was the dominant cost behind Rust's durable-write throughput trailing
  TypeScript's; Rust is now at parity (was 0.67×, now ~1.0×) and 1.1–2×
  faster on the in-memory mixed workload.
- `SyncServer::submit()` computed and discarded a whole-batch checksum, and
  cloned each accepted operation twice across two passes. Fixed both,
  ~15% faster sync throughput.
- `NativeStore::apply()` deserialized every JS object twice per call (napi's
  own JS→JSON walk, then `serde_json::from_value`). Replaced with typed
  `#[napi(object)]` mirror structs converted via `TryFrom`, 1.14×–2× faster
  across all batch sizes; at batch sizes ≥500 it now beats the TypeScript
  write path instead of trailing it.

## [3.2.0] - 2026-08-22

### Added

- Aligned the adaptive-memory layer across TypeScript, Rust, Python, and the
  native bindings.
- Added persisted/cold working-memory ranking, context fallback, built-in token
  estimation, cost validation, explainable score breakdowns, and learned-weight
  export/restore APIs.
- Preserved adaptive and provenance metadata when reading persisted nodes.
- Added native token-estimation and score-breakdown helpers, plus local native
  binary preference during monorepo development.

## [3.1.0] - 2026-08-22

### Added

- Adaptive memory layer, extended across TypeScript, Rust, and Python:
  - Per-context activation (`NodeActivation.context`): `reinforceNode`/
    `reinforce_node` gain an optional `context` argument that reinforces an
    independently-decaying, additional lens alongside the global score.
    `getContextActivation`/`get_context_activation` reads it back.
  - Inhibition (`NodeActivation.inhibition`/`lastInhibitedAt`): a new
    `suppressNode`/`suppress_node` primitive with its own decay half-life,
    subtracted from `score` only at the final read/ranking layer so a
    suppressed node stays re-evaluable rather than permanently invisible.
  - Budgeted, diversity-aware `ActivationEngine.workingMemory`/
    `working_memory`: an options form (`tokenBudget`, `costOf`,
    `diversityLambda`, `similarityOf`) implementing a memory-flavoured
    maximal-marginal-relevance selection, backward compatible with the
    existing `(limit, minScore)` call shape.
  - Memory classes (`episodic`/`semantic`/`procedural`/`entity`): a
    per-node `memoryClass` override falling back to a schema-level
    `NodeTypeDefinition.memoryClass` default, driving differentiated
    score/importance decay half-lives via `ActivationConfig.classHalfLives`.
  - Confidence and provenance fields on `PolyNode`: `confidence`, `source`,
    `observedAt`, `derivedFrom`, `supersedes`, `contradicts` — ordinary,
    optional node fields (not activation state), settable via `addNode`,
    `updateNode`, and `patchNode` (which now also supports these as
    top-level `set`/`unset`/`increment`/`compareAndSet` paths, not only
    `data.*`, in all three languages).
  - `supersede(id, supersededId, amount?, reason?)`: a contradiction
    primitive that records `supersedes`, adds a `SUPERSEDED_BY` edge, and
    suppresses the superseded node without deleting it.
  - `consolidate(node, sourceIds, options?)`: a consolidation primitive
    that writes a higher-level node with `derivedFrom` merged from its
    sources, adds `CONSOLIDATED_FROM` edges, and suppresses the sources.
  - `ActivationEngine.recordFeedback(id, wasUseful, learningRate?)`:
    learned scoring weights — nudges `pulse`'s composite
    semantic/graph/recency/usage weights from outcome feedback instead of
    requiring hand-tuning.
  - Fixed a latent double-decay bug in the TypeScript
    `ActivationEngine.effective`, found while adding class-resolved
    half-lives: it re-decayed the already-decayed result of
    `getActivationState` a second time, silently breaking any non-default
    half-life under real elapsed time.
  - See `specification/data-model.md` sections 1.5–1.10 and
    [docs/API.md](docs/API.md) for full details.

### Fixed

- Restored an edge-id integrity check in the Rust graph's `rebuild_edge_index`
  (rejecting empty ids) that had been dropped while adding support for
  independent edge IDs.
- Python `DirectoryStorage` now implements real `sync`/`sync_dir` (fsync),
  bridged from the native storage adapter, so its advertised `fsync`
  capability is honest.
- Rust `attribute_ranges` queries now resolve nested dot-path fields the same
  way `attributes` does, instead of only matching flat top-level fields.
- Rust's in-memory `GraphQuery` index selection now excludes a sparse index
  when the queried value is `null`, matching `PersistedGraphQuery`.
- Rust `count_nodes` now derives its `maxNodesVisited` candidate set the same
  way `query_nodes` does, instead of always falling back to the full store
  size.
- The TypeScript `BinaryStoreAdapter` now dedups retried batches by
  `transactionId` as well as `operationId`, matching the Rust/Python
  idempotency contract.
- Rust's `SyncServer::submit` adds a `clientId:seq` fallback dedup (evicted
  alongside the ops ring buffer), matching the TypeScript server.
- Python's `SyncServer.receive` stops evaluating `conflict()` for an
  operation `authorize()` already rejected, matching the TypeScript server.

## [3.0.0] - 2026-08-21

### Added

- Promoted the database-core contract across TypeScript, Rust, and Python with
  shared persistence, transaction, schema, index, migration, sync, and query
  conformance coverage.
- Added cross-binding database-core benchmarks for durable batches, queries,
  vector search, and mixed read/write workloads.
- Added Python `PersistedGraphQuery` with native storage-level filtering,
  ordering, pagination, and counting for directory-backed stores.

### Changed

- Python hot-cache queries route simple filters and pagination through the local
  pipeline, while native execution is reserved for similarity, joins, and
  traversal workloads where boundary conversion can be amortized.
- Aligned the npm, crates.io, and PyPI package versions at `3.0.0`.

## [2.5.0] - 2026-08-02

### Added

- **Activation model (adaptive memory)**. A new two-tier relevance layer turns
  Polypack from a passive data store into an adaptive memory system:
  - **Durable activation.** Optional `NodeActivation` (`score`, `importance`,
    `reinforcementCount`, `lastMeaningfulActivation`) on every node, persisted
    through the existing snapshot/WAL and adapters (MessagePack carries the new
    field, so stores remain forward/backward compatible).
  - **Core `PolyGraph` primitives.** `reinforceNode`/`reinforceNodeSafe`
    (durable reinforcement, decay-corrected, re-anchored, emits a new
    `activation_updated` change event with `delta`/`reason`),
    `getActivation`/`getActivationState` (lazy decay as a pure function of
    elapsed time from the anchor, so replicas converge), `topActivated`
    (working-memory primitive), and `decay` (materialize fresh values).
  - **`@0xx0lostcause0xx0/polypack/activation` subpath.** `ActivationEngine`
    composing the adaptive layer: transient (never-synced) attention
    (`bumpAttention`/`attentionOf`/`effective`), spreading activation
    (`spread` with per-hop attenuation and edge-type filtering), semantic
    region scoring (`pulse`, zero-similarity nodes never seed the region),
    self-maintaining reinforcement (`absorb`), and a live `workingMemory`
    view. `mergeActivation` merges total-state records (max, idempotent).
  - **Query integration.** `whereActivated` and `orderByActivation` on both
    `GraphQuery` and `PersistedGraphQuery`; `GraphQuery` falls back from the
    native executor when activation filters are present.
  - **React.** `useWorkingMemory` — a live view of the most-activated nodes;
    pass an optional `ActivationEngine` to rank by durable score *plus*
    transient attention instead of the graph's durable-only ranking.
  - **Sync semantics.** Activation is synced as *derived statistical state*,
    not ordinary CRUD: deltas accumulate via a new `activationUpdate` op
    (coalesced per node and gated by `SyncClientOptions.activationSyncThreshold`,
    default 0.05) while full-node payloads max-merge on arrival, and an absent
    incoming activation never wipes locally learned state. The `SyncClient`
    applies deltas synchronously for loaded nodes so the echo-suppression window
    stays intact.
  - `examples/activation.ts` — a Nostr-style adaptive-memory scenario.
  - **Rust port (`polypack-core` + `polypack-graph`).** The activation model is
    implemented natively in Rust, byte-for-byte compatible with the TypeScript
    reference: `NodeActivation` on the core `Node` (persisted through the
    snapshot/WAL — the Rust `node_to_msg`/`msg_to_node` codecs now always emit
    the `activation` key, matching TS's explicit-nil encoding, with updated
    byte-compat hex fixtures and a new activation round-trip test);
    `polypack_core::activation` (decay/reinforce/merge math); `Graph`
    primitives (`reinforce_node`/`reinforce_node_safe` emitting a new
    `ActivationUpdated` event, `get_activation`/`get_activation_state`,
    `top_activated`, `decay`); `where_activated`/`order_by_activation` on
    `GraphQuery` and `PersistedGraphQuery`; and an `ActivationEngine` with
    spread/pulse/absorb/working-memory.
  - **Python port (`polypack-db`).** The pure-Python `PolyGraph`/`GraphQuery`
    layer carries `activation` through node plumbing and the store, gains
    `reinforce_node`/`get_activation`/`top_activated`/`decay` and
    `where_activated`/`order_by_activation` (activation filters bypass the
    native query executor), and ships an `ActivationEngine`, `merge_activation`,
    and `decay_factor`. `reinforce_node_safe` is an alias — the Python graph
    has no hot-cache eviction.
  - **Native bindings (`polypack-node`/`polypack-native`).** Pure activation
    helpers exposed through NAPI and wrapped in `@0xx0lostcause0xx0/polypack-native`:
    `decayFactor`, `mergeActivation`, `reinforceActivation`, `activationScoreOf`,
    with a parity test. The TS activation engine itself stays pure TypeScript.

### Changed

- `PolyNode`/`SerializedNode` gain an optional `activation` field;
  `GraphChangeEvent` gains the `activation_updated` type (additive, non-breaking).
- `updateNode` accepts an optional fourth `activation` argument for sync-merge
  application.
- The Rust snapshot/WAL wire format now carries the `activation` node field
  (always emitted, nil when absent, matching TypeScript) — old stores without
  the field still decode.

### Fixed

- `SyncClient.disconnect()` now flushes pending operations (including
  coalesced `autoFlush: false` activation deltas) before tearing down,
  instead of silently discarding whatever hadn't been sent yet.
- `HNSWIndex` now validates `M`/`Mmax0`/`efConstruction`/`efSearch` are
  positive integers and throws `RangeError` on invalid config, instead of
  silently building a degenerate graph that returns bad query results.
- `HNSWIndex.update()` — not part of `VectorIndexLike` and functionally
  identical to `add()` — is now `@deprecated` and delegates to `add()`
  instead of duplicating its logic.
- `BinaryStoreAdapter.close()` now flips its closed state synchronously
  before enqueueing shutdown work, closing a race where a write issued
  concurrently with `close()` could pass its open-check and land after
  the compaction step completed.
- Python `PolyGraph.close_store()` (and the `with PolyGraph.open(...)`
  context manager, which calls it) now saves before closing the store,
  instead of silently discarding mutations made since the last explicit
  `save()` — matching Rust's `Graph::close`, which already flushes first.
- Rust `HnswIndex::new` now validates `m`/`mmax0`/`ef_construction`/
  `ef_search` are at least 1 and returns `Err(InvalidArgument)` instead of
  either building an index whose queries silently return no results
  (`ef_search == 0`) or panicking on the second insert (`ef_construction ==
  0`). Propagated through `Graph::open`, the N-API `NativeHnswIndex`
  constructor, and the PyO3 `HnswIndex` constructor (all now fallible).
- Removed two stale Rust doc comments in `polypack-graph::Graph` claiming
  `remove_node` and hot-cache eviction were unimplemented stubs that would
  panic; both are fully implemented and tested.
- `polypack-graph` now re-exports `polypack_core::query::Direction`, which
  `GraphQuery`/`PersistedGraphQuery`'s `traverse`/`join` require as a
  parameter but previously couldn't be named without an explicit
  `polypack-core` dependency.
- The Python wheel/sdist (`maturin`) no longer bundles `__pycache__`/`.pyc`
  build-environment leakage. (`bench.py`/`conformance.py` were briefly
  excluded too, but CI builds a real wheel and runs the full pytest suite
  — including `test_conformance.py` — against it, so they're back; both
  are genuine, if repo-relative, importable modules post-install.)
- `@0xx0lostcause0xx0/polypack-native`'s `files` field now excludes
  `dist/*.node`, so a locally-built platform binary left in `dist/` can't
  accidentally ship in the root npm package — only the per-platform
  `optionalDependencies` packages should carry a binary.

## [2.4.7] - 2026-08-01

### Added

- New `polypack-graph` crate: a pure-Rust port of `PolyGraph`'s public API
  on top of `polypack-core`, for applications that need the graph engine
  without a JS runtime. Covers node/edge CRUD with owned/shared
  cascade-delete semantics, the flush/warm/save/clear/dispose/prune
  persistence lifecycle, hot-cache eviction backed by an O(1) LRU list,
  the `GraphQuery`/`PersistedGraphQuery` fluent query builders (including
  joins, traversal, similarity ranking, and grouping/aggregation), and
  pluggable text embeddings (`EmbeddingProvider`, `FeatureHashEmbedding`,
  verified bit-for-bit identical to the TypeScript implementation) with
  `add_node_with_embedding`/`query_text`/`search_nodes` wired through.
  Published to crates.io alongside `polypack-core`.

### Fixed

- Resolved `cargo clippy --all-targets -D warnings` failures in the new
  `polypack-graph` crate that were blocking CI: type-complexity on several
  `dyn Fn` closure fields (factored into type aliases), a needless
  `.clone()` on the `Copy` `HnswConfig`, two `sort_by` calls better
  expressed as `sort_by_key`, an `if`/`else` with identical bodies in
  `remove_edges`'s edge-index cleanup, and a manual modulo check now
  expressed as `u64::is_multiple_of`. No behavior changes.

## [2.4.6] - 2026-08-01

### Fixed

- `PolyGraph`'s `createVectorIndex` constructor hook was typed as
  `(onChange) => VectorIndex`, so `HNSWIndex` and
  `@0xx0lostcause0xx0/polypack-native`'s `NativeVectorIndex`/`NativeHnswIndex`
  — the entire point of the hook — failed to type-check under `tsc --strict`
  despite working at runtime, because `VectorIndex`'s private fields give it
  a nominal type brand. Introduced a structural `VectorIndexLike` interface
  (now exported from the package root) that `VectorIndex` and `HNSWIndex`
  both implement, and retyped `PolyGraph.vectors`/`createVectorIndex`
  against it.
- `MemoryAdapter`'s `maxNodes` cap was not actually LRU: `applyChanges` and
  `bulkPutNodes` re-put an existing node via a raw `Map.set()`, which does
  not move an existing key to the end of a JS `Map`'s iteration order (only
  `putNode`'s delete-then-set did). Since `PolyGraph.flush()`/`save()`
  prefer `applyChanges` when available, a frequently-updated node could
  still be evicted as if untouched. Both paths now route through the same
  touch-then-set helper as `putNode`.

### Documentation

- README, `docs/API.md`, and per-package docs (`python/README.md`,
  `crates/polypack-core/README.md`, `packages/node-native/README.md`)
  updated to cover all three distribution channels (npm, PyPI's
  `polypack-db`, crates.io's `polypack-core`) and to describe recent
  additions (`addNodes`, adaptive compaction, secondary indexes,
  `createVectorIndex`, `HNSWIndex`, `markVectorDirty`, `VectorIndex.hydrate`,
  `MemoryAdapter`'s eviction behavior).
- Added missing JSDoc/rustdoc/docstrings across the TypeScript, Rust, and
  Python public API surfaces, and merged three duplicate/leftover doc
  comments found on `SyncClient`, `SyncServer`, and `MemoryTransport`.

## [2.4.5] - 2026-08-01

### Fixed

- `get_edges_by_sources`/`get_edges_by_targets` in the Rust `Store` tripped
  clippy's `unnecessary_map_or` lint (`edge_type.map_or(true, |t| ...)` →
  `edge_type.is_none_or(|t| ...)`), which CI enforces as a hard error. No
  behavior change.

## [2.4.4] - 2026-08-01

### Added

- `PolyGraph.addNodes(nodes)` batches node inserts: validation happens for the
  whole batch before any insert, change events coalesce into one flush, and
  the persistence debounce is scheduled once. Prefer it over a loop of
  `addNode` for large loads.
- Persisted-query fast paths in `MemoryAdapter` and `BinaryStoreAdapter`:
  `countNodes({})` returns the total without materialising ids, type-only
  queries use a secondary type index, and `getEdgesBySources`/`getEdgesByTargets`
  are backed by source/target edge indexes instead of scanning every edge.
- `polypack-core` `Store` mirrors the above: `node_count`, `query_nodes`,
  `count_nodes`, `get_edges_by_sources`, and `get_edges_by_targets`, exposed
  through the Python and Node native bindings.

### Changed

- WAL compaction now uses an adaptive threshold (`max(compactThreshold,
  records / 4)`) in both the TypeScript adapter and the Rust `Store`, keeping
  total compaction work linear in writes instead of quadratic. The
  `compactThreshold` config is now a lower bound rather than a fixed count.

### Fixed

- The new secondary type index (`MemoryAdapter`, `BinaryStoreAdapter`, and the
  Rust `Store`) left a stale entry under a node's previous type whenever the
  node was overwritten with a different type, corrupting `countNodes`/
  `queryNodes` results for that type indefinitely. `indexNode`/`index_node`
  now unindex the prior type first.

## [2.4.3] - 2026-08-01

Version-only re-release. `@0xx0lostcause0xx0/polypack-native@2.4.2` was
accidentally unpublished from npm during a manual recovery from an OIDC
trusted-publisher outage; npm permanently blocks reusing an unpublished
name+version, so the whole stack moves to `2.4.3` to stay in lockstep. No
code changes beyond `2.4.2`.

## [2.4.2] - 2026-08-01

### Fixed

- Python `PolyGraph.save()` only ever sent puts, so deleting a node, edge, or
  vector and saving could resurrect it on the next `open()`. Deletions are
  now tracked and flushed through `Store.apply()`.
- Several Python graph invariants found alongside the above: a duplicate-edge
  check against the wrong dict level, a set-based (not refcounted)
  incoming-edge index that could drop a still-live source, stale
  incoming-index entries after node removal, shallow node `data` copies,
  `update_node()` always writing `updatedAt = 0`, the context manager
  clearing state instead of closing the attached store, and an overly broad
  `except` swallowing genuine native query errors.
- `Store::apply()` in `polypack-core` mutated in-memory state before
  appending to the WAL; a failed append or fsync could leave memory ahead of
  disk. Mutation now only happens after the WAL write succeeds.
- Snapshot writes (Node `FsStorage`, Python `DirectoryStorage`) now go
  through a temp file, fsync, and atomic rename instead of a direct write;
  the Node adapter also implements directory fsync.
- Corrected the README's `@0xx0lostcause0xx0/polypack/native` subpath
  reference to the real, separately published `@0xx0lostcause0xx0/polypack-native`
  package.

### Changed

- The Rust crates, Python wheel, and native npm packages now share one
  version with the TypeScript package instead of tracking an independent
  `0.1.x` line. See `RELEASING.md`.

## [2.4.1] - 2026-07-31

Coordinated re-release of the multi-language stack after the initial 2.4.0
attempt. Version bump only (npm/crates.io/PyPI require a fresh version for a
re-publish); the Python wheel distribution is `polypack-db` to match the PyPI
project.

## [2.4.0] - 2026-07-31

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
