# polypack

Generic property graph engine with vector similarity search, edge ownership semantics, relational queries, and real-time sync. Runs in browser (IndexedDB) and Node.js (MemoryAdapter).

## Install

```sh
npm install @0xx0lostcause0xx0/polypack
```

React is an optional peer dependency and is only required when importing from `@0xx0lostcause0xx0/polypack/react`.

## Features

- **Property graph** — typed nodes and edges with arbitrary data payloads
- **LRU working set** — 10K loaded-node limit by default, with explicit restoration from persistence
- **Vector similarity** — cosine, euclidean, or pluggable distance functions
- **Fluent query builder** — filter by type/attribute/edge/range, BFS traversal, vector similarity
- **Relational extensions** — `pluck`, `aggregate`, `groupAggregate`, `join`, `groupByVector` (clustering)
- **Edge ownership** — `owned` (cascade delete), `shared` (orphan detection), `reference` (no-op)
- **Reactive** — RxJS change events, batching, React hooks
- **Pluggable persistence** — MemoryAdapter, IndexedDBAdapter, build your own
- **Persisted queries** — asynchronous filtering and similarity across the full backing store
- **Real-time sync** — acknowledgements, retry, deduplication, reconnect, and echo suppression

## Quick start

```typescript
import { PolyGraph, MemoryAdapter } from '@0xx0lostcause0xx0/polypack'

const graph = new PolyGraph()

// Add nodes with vectors
graph.addNode({
  id: 'doc_1',
  type: 'document',
  data: { title: 'Quantum Computing' },
  vector: new Float64Array([0.95, 0.20, 0.10]),
  insertedAt: Date.now(),
  updatedAt: Date.now(),
})

// Search by similarity
graph.query()
  .whereNodeType('document')
  .similarTo([0.90, 0.30, 0.10], 0.5, 5)
  .toArray()

// Traverse edges
graph.query()
  .where('title', 'Quantum Computing')
  .traverse('REFERENCES', 3, 'out')
  .toArray()

// Aggregate
graph.query()
  .whereNodeType('book')
  .aggregate('price', 'avg')

// Group by vector cluster
graph.query()
  .whereNodeType('product')
  .groupByVector(
    [{ key: 'electronics', centroid: [0.9, 0.1] }],
    'price', 'avg', 0.4,
  )
```

## Packages

| Subpath | Contents |
|---------|----------|
| `@0xx0lostcause0xx0/polypack` | Core: PolyGraph, VectorIndex, GraphQuery, persistence adapters |
| `@0xx0lostcause0xx0/polypack/react` | React hooks: `useGraphQuery`, `useLiveQuery` |
| `@0xx0lostcause0xx0/polypack/sync` | Sync layer: OpLog, SyncAdapter, SyncClient, SyncServer |

See the complete [API reference](docs/API.md), including persistence, React,
sync, lifecycle, ownership, and error contracts.

## Requirements

- Node.js 18 or newer when used in Node.js.
- A browser with IndexedDB when using `IndexedDBAdapter`.
- Equal vector dimensions for similarity operations; mismatches throw `RangeError`.

Polypack is distributed as native ES modules.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions. Please
report vulnerabilities according to [SECURITY.md](SECURITY.md).

## License

MIT
