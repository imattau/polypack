# Conformance fixtures

Language-neutral JSON fixtures consumed by the TypeScript conformance runner
(`tests/conformance/`). Rust and Python implementations must pass the same
fixtures. Fixture files are the cross-language contract; they do not reference
TypeScript implementation details.

The current behavioural specification is version 2. Existing schema-version 1
fixtures remain valid for compatibility; new fixtures should not rely on
separator-derived edge IDs.

The database-core contract is documented in
[`specification/database-core.md`](../specification/database-core.md).

## `conformance/` — graph behaviour

Each file is one fixture:

```json
{
  "schemaVersion": 1,
  "name": "unique name",
  "group": "one of the conformance groups",
  "orphanAware": true,
  "graphOptions": { "hotCacheMax": 3 },
  "setup": { "nodes": [...], "edges": [...] },
  "operations": [...],
  "expect": { ... }
}
```

### Setup nodes

Serialized node envelopes from the change-batch schema. `vector` may be `null`
or a finite number array. `insertedAt`/`updatedAt` are integer milliseconds.

### Setup edges

`{ "source", "type", "target", "data"?, "ownership"? }` where ownership is
`owned` | `shared` | `reference`.

### Operations

| op               | fields                                              | notes                              |
|------------------|-----------------------------------------------------|------------------------------------|
| `addNode`        | `node`                                              | optional `expectError`             |
| `updateNode`     | `id`, `data`, `vector`?                             | optional `expectError`             |
| `addEdge`        | `source`, `type`, `target`, `data`?, `ownership`?   | optional `expectError`             |
| `removeNode`     | `id`                                                |                                    |
| `removeEdges`    | `source`, `type`?, `target`?                        |                                    |
| `hnswAdd`        | `id`, `vector`                                      | requires a fixture with HNSW use   |
| `hnswRemove`     | `id`                                                |                                    |
| `hnswUpdate`     | `id`, `vector`                                      |                                    |
| `mutateDetached` | `id`                                                | mutates a returned snapshot copy   |

`expectError` uses the codes from `specification/errors.md`.

### Expectations

| key                 | meaning                                                        |
|---------------------|----------------------------------------------------------------|
| `presentNodeIds`    | ids that must exist                                            |
| `absentNodeIds`     | ids that must not exist                                        |
| `nodeCount`         | loaded node count (`graph.size`)                               |
| `loadedSize`        | loaded working-set size (`graph.loadedSize`)                   |
| `nodeData`          | spot-check `id -> field -> value`                              |
| `nodeVector`        | spot-check node vectors                                        |
| `edgeTargets`       | `[{source, type, targets}]` compared unordered                 |
| `orphanEvents`      | ordered ids fired through the orphan hook (`orphanAware`)      |
| `queries`           | `[{plan, resultIds}]` ordered ids from a query-plan IR         |
| `exactSearches`     | exact top-K ids, or `expectError`                              |
| `hnswSearches`      | approximate top-K with `minOverlap` tolerance                  |
| `hnswVector`        | HNSW index vector reads                                        |
| `hnswSize`          | HNSW index size                                                |
| `aggregate`         | `{field, op, value, count}` of `aggregate()` over a plan       |

Query plans use `specification/query-plan.schema.json`.

## `recovery/` — persistence recovery

Each fixture describes a store to build and the state expected after opening
it:

```json
{
  "store": {
    "snapshot": { "nodes": [...], "edges": [...], "vectors": [["id", [...]]] },
    "wal": [ entry objects ],
    "corruptTailHex": "hex bytes appended after the WAL"
  },
  "expect": {
    "presentNodeIds": [...],
    "absentNodeIds": [...],
    "vectors": { "id": [...] },
    "walRemovedAfterRecovery": true,
    "snapshotPresentAfterRecovery": true
  }
}
```

WAL entry objects follow the kinds in `specification/persistence.md`
(`putNode`, `deleteNode`, `putEdge`, `deleteEdge`, `putVector`,
`deleteVector`, `clearAll`). `corruptTailHex` simulates a crash mid-append; the
trailing partial frame must be tolerated. The runner encodes these into bytes
using its own format implementation — the fixture content is language-neutral.
