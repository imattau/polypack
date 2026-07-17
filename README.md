# polypack

Generic property graph engine with vector similarity search, edge ownership semantics, relational queries, and real-time sync. Runs in browser (IndexedDB) and Node.js (MemoryAdapter).

## Install

```sh
npm install polypack
```

React is an optional peer dependency and is only required when importing from `polypack/react`.

## Features

- **Property graph** — typed nodes and edges with arbitrary data payloads
- **LRU hot cache** — 10K node limit by default, spills to pluggable persistence
- **Vector similarity** — cosine, euclidean, or pluggable distance functions
- **Fluent query builder** — filter by type/attribute/edge/range, BFS traversal, vector similarity
- **Relational extensions** — `pluck`, `aggregate`, `groupAggregate`, `join`, `groupByVector` (clustering)
- **Edge ownership** — `owned` (cascade delete), `shared` (orphan detection), `reference` (no-op)
- **Reactive** — RxJS change events, batching, React hooks
- **Pluggable persistence** — MemoryAdapter, IndexedDBAdapter, build your own
- **Real-time sync** — OpLog + SyncAdapter + SyncServer/SyncClient with echo suppression

## Quick start

```typescript
import { PolyGraph, MemoryAdapter } from 'polypack'

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
| `polypack` | Core: PolyGraph, VectorIndex, GraphQuery, persistence adapters |
| `polypack/react` | React hooks: `useGraphQuery`, `useLiveQuery` |
| `polypack/sync` | Sync layer: OpLog, SyncAdapter, SyncClient, SyncServer |

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
