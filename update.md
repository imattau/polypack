# Polypack remaining work

## 1. Complete persisted graph queries

- [x] Add persisted edge filters, joins, collection, and traversal.
- [x] Push pagination into adapters and stop IndexedDB cursor scans early when
  query ordering and shape permit it.
- [x] Push constrained indexed ordering into IndexedDB adapters to avoid sorting
  every matching node.
- [x] Add configurable IndexedDB indexes for frequently queried attributes.
- [x] Add initial persisted-query benchmarks and 10K-node coverage.
- Add larger persisted traversal and IndexedDB-specific benchmarks.

## 2. Build reliable synchronization

- [x] Add acknowledgements and retain unacknowledged operations.
- [x] Add retry and explicit transport reconnection behavior.
- [x] Deduplicate operations on the server.
- [x] Recover through operation snapshots or cursor-based deltas after reconnect.
- Add compact state snapshots and operation-log retention/compaction policies.
- Define an explicit conflict-resolution policy.
- Add durable server operation storage.
- Provide authentication integration hooks while leaving authentication policy to applications.

## 3. Harden mutation boundaries

- [x] Prevent callers from silently mutating internal node, edge, data, and
  vector references by returning detached snapshots.
- [x] Add explicit vector removal support.
- [x] Validate IDs, timestamps, vectors, pagination values, and traversal depth.
- [x] Structured-clone mutation inputs so the graph owns its internal state.
- [x] Use detached mutable snapshots instead of readonly live node views.

## 4. Improve persistence consistency

- [x] Define an optional atomic node, edge, and vector transaction contract.
- [x] Test mixed-store failures, rollback, dirty-state restoration, and retry.
- [x] Add IndexedDB schema migration tests from version 1 to version 2.
- [x] Prefer adapter-native persisted counts over `allNodeIds()` fallbacks.

## 5. Expand React verification

- [x] Add mounted-hook tests for changing `nodeTypes` and `delay`.
- [x] Test asynchronous query races, unmounting, and rapid mutation bursts.
- [x] Add an optional query-error callback while preserving error logging.

## 6. Prepare the next release

- [x] Bump the package version to `2.1.0` for the breaking mutation-boundary,
  validation, similarity, and IndexedDB schema changes.
- [x] Update `CHANGELOG.md` with breaking changes and migration guidance.
- [x] Inspect the package contents with `npm pack --dry-run` and install the
  actual tarball in a clean consumer project.
- [x] Verify root, sync, and optional React exports from the packed artifact.
- [x] Verify the IndexedDB version-1 to version-2 migration in headless Firefox.

## 7. Add pluggable text embeddings

- [x] Define a sync/async provider contract for custom embedding models.
- [x] Include a deterministic model-free feature-hash provider by default.
- [x] Add graph helpers for embedded node creation, updates, and text queries.
- [x] Validate provider dimensions and finite vector output.

## Recommended sequence

1. Add compact sync snapshots and an operation-log retention policy.
2. Define sync conflict resolution, durable operation storage, and auth hooks.
3. Expand persisted traversal and IndexedDB benchmarks at larger scales.
4. Publish the prepared `2.1.0` release after final review.
