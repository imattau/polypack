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

- Prevent callers from silently mutating internal node and data references.
- Add explicit vector removal support.
- Validate IDs, timestamps, vectors, pagination values, and traversal depth.
- Decide whether mutations clone input or take ownership.
- Consider readonly node views.

## 4. Improve persistence consistency

- Define atomic node, edge, and vector transaction expectations.
- Test partial adapter failures across mixed operations.
- [x] Add IndexedDB schema migration tests from version 1 to version 2.
- Prefer adapter-native persisted counts over `allNodeIds()` fallbacks.

## 5. Expand React verification

- Add mounted-hook tests for changing `nodeTypes` and `delay`.
- Test asynchronous query races, unmounting, and rapid mutation bursts.
- Consider an explicit error result or callback instead of only logging failures.

## 6. Prepare the next release

- Bump the package version beyond `1.1.0`.
- Update `CHANGELOG.md`.
- Choose `1.2.0` or a larger release based on persisted-query stability.
- Inspect package contents with `npm pack --dry-run`.
- Verify the IndexedDB version-2 migration in a real browser.

## Recommended sequence

1. Persisted edge traversal and join support.
2. Persisted query performance and adapter pagination.
3. Reliable synchronization as a separate milestone.
4. Mutation-boundary and persistence hardening.
5. React verification and release preparation.
