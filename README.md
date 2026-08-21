# polypack

[![npm version](https://img.shields.io/npm/v/%400xx0lostcause0xx0%2Fpolypack?logo=npm)](https://www.npmjs.com/package/@0xx0lostcause0xx0/polypack)
[![GitHub release](https://img.shields.io/github/v/release/imattau/polypack?logo=github)](https://github.com/imattau/polypack/releases/latest)
[![CI](https://github.com/imattau/polypack/actions/workflows/ci.yml/badge.svg)](https://github.com/imattau/polypack/actions/workflows/ci.yml)

Polypack is an embedded graph and vector database for adaptive, local-first
applications. It combines property relationships, semantic search, persistent
activation, working-memory primitives, and real-time synchronisation across
TypeScript, Python, and Rust. It runs in Node.js and the browser with pluggable
persistence (in-memory, filesystem, or OPFS).

## Install

```sh
npm install @0xx0lostcause0xx0/polypack
```

React is an optional peer dependency and is only required when importing from `@0xx0lostcause0xx0/polypack/react`.

Python and Rust bindings over the same core are published separately:

```sh
pip install polypack-db
```

```sh
cargo add polypack-core
```

## Releases

- **npm package:** [`@0xx0lostcause0xx0/polypack`](https://www.npmjs.com/package/@0xx0lostcause0xx0/polypack)
  is the installable TypeScript/JavaScript distribution for Node.js and browser
  build tooling. [`@0xx0lostcause0xx0/polypack-native`](https://www.npmjs.com/package/@0xx0lostcause0xx0/polypack-native)
  provides NAPI-RS bindings to the native `VectorIndex`/`HNSWIndex`.
- **PyPI package:** [`polypack-db`](https://pypi.org/project/polypack-db/)
  is the Python distribution, built with maturin/PyO3 over the same Rust core.
- **crates.io package:** [`polypack-core`](https://crates.io/crates/polypack-core)
  is the portable Rust core (property graph, vector search, persistence)
  underlying the TypeScript native addon and the Python bindings.
- **GitHub releases:** [release notes, source archives, and tags](https://github.com/imattau/polypack/releases)
  are published from the repository. The latest GitHub release is
  [`v3.0.0`](https://github.com/imattau/polypack/releases/tag/v3.0.0); the
  current coordinated source version is `3.0.1`.

Stable GitHub releases run the complete test, build, export, and package checks
before the corresponding packages are submitted to npm, PyPI, and crates.io
with provenance/trusted publishing. All three ecosystems are version-locked
together. See the [changelog](CHANGELOG.md) for breaking changes and
migration notes.

## Features

- **Property graph** — typed nodes and edges with arbitrary data payloads
- **LRU working set** — 50K loaded-node limit by default, with explicit restoration from persistence
- **Vector similarity** — cosine, euclidean, or pluggable distance functions
- **Pluggable text embeddings** — supply any local or hosted model, with a
  dependency-free 384-dimensional feature-hash provider included by default
- **Fluent query builder** — filter by type/attribute/edge/range, BFS traversal, vector similarity
- **Relational extensions** — `pluck`, `aggregate`, `groupAggregate`, `join`, `groupByVector` (clustering)
- **Edge ownership** — `owned` (cascade delete), `shared` (orphan detection), `reference` (no-op)
- **Reactive** — RxJS change events, batching, React hooks
- **Pluggable persistence** — MemoryAdapter, BinaryStoreAdapter (MessagePack + WAL), build your own
- **Database core** — atomic transactions, revisions, conditional writes, patches, schema hooks, and resource limits
- **Capability checks** — inspect adapter guarantees and reject configurations that require unsupported durability, indexing, concurrency, or vector-search features
- **Operational durability** — checkpoint/backup/verification APIs, durable logical mutation logs, cursors, and idempotent operation IDs
- **Schema migrations** — contiguous, validated, retry-safe application migrations with dry runs, resume cursors, and progress reporting
- **Explainable queries** — persisted and hot-query plans with index selection, estimated cost, and operational metrics
- **Configurable indexes** — compound and unique node-data indexes with persisted Python index metadata
- **Cross-language contract** — shared conformance fixtures and compatibility levels for TypeScript, Rust, and Python
- **Persisted queries** — asynchronous filtering, ordering, pagination, and similarity across the full backing store without loading the result set into the hot cache
- **Adaptive memory** — activation model: durable, decayed relevance (`score`/`importance`) per node, spreading activation over edges, semantic pulses, and a live working-memory set — synced additively
- **Real-time sync** — acknowledgements, retry, deduplication, reconnect recovery, and echo suppression

For scale limits and reproducible 100K/1M/10M-node characterization, see the
[scale benchmark](benchmarks/scale-characterization.md). Its default vector
scope is capped at 100K for practical 10M-node storage and graph runs; full
vector indexing is an explicit opt-in.

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

// Or generate vectors from text with the default model-free provider
await graph.addNodeWithEmbedding({
  id: 'doc-2',
  type: 'document',
  data: { title: 'Graph search' },
  insertedAt: Date.now(),
  updatedAt: Date.now(),
}, 'Property graphs with vector similarity search')

const textMatches = await graph.queryText('semantic graph search', 0.1, 10)
textMatches.toArray()

// Query the complete persisted store without warming the hot cache
const recentPosts = await graph.queryPersisted()
  .whereNodeType('document')
  .orderBy('updatedAt', 'desc')
  .limit(20)
  .toArray()

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
| `@0xx0lostcause0xx0/polypack` | Core: PolyGraph, VectorIndex, GraphQuery, PersistedGraphQuery, MemoryAdapter |
| `@0xx0lostcause0xx0/polypack/persistence` | Platform-neutral persistence: adapters, `FileIO` types |
| `@0xx0lostcause0xx0/polypack/persistence/node` | BinaryStoreAdapter + `NodeFileIO` for the filesystem |
| `@0xx0lostcause0xx0/polypack/persistence/opfs` | BinaryStoreAdapter + `OPFSFileIO` for the browser |
| `@0xx0lostcause0xx0/polypack/react` | React hooks: `useGraphQuery`, `useLiveQuery`, `useWorkingMemory` |
| `@0xx0lostcause0xx0/polypack/activation` | Adaptive memory: `ActivationEngine`, `mergeActivation`, activation config types |
| `@0xx0lostcause0xx0/polypack/sync` | Sync layer: OpLog, SyncAdapter, SyncClient, SyncServer |
| `@0xx0lostcause0xx0/polypack-native` | Separate package: NAPI-RS bindings for native `VectorIndex`/`HNSWIndex` over the Rust core |
| `polypack-db` (PyPI) | Separate package: PyO3/maturin bindings exposing `PolyGraph`, `GraphQuery`, `PersistedGraphQuery`, and vector indexes to Python — see [python/README.md](python/README.md) |
| `polypack-core` (crates.io) | Separate package: the portable Rust core (property graph, vector search, persistence) shared by the TypeScript native addon and the Python bindings |

See the complete [API reference](docs/API.md), including persistence, React,
sync, lifecycle, ownership, and error contracts.

The Python binding has a separate, idiomatic API with `snake_case` method names.
See the [Python API guide](python/README.md) for `PolyGraph`, hot-cache
`GraphQuery`, native storage-level `query_persisted()`, vector indexes, and
binding-specific behavior.

## Requirements

- Node.js 18 or newer when used in Node.js.
- A browser with the File System Access API (OPFS) when using
  `@0xx0lostcause0xx0/polypack/persistence/opfs`.
- Equal vector dimensions for similarity operations; mismatches throw `RangeError`.

Polypack is distributed as native ES modules.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions. Please
report vulnerabilities according to [SECURITY.md](SECURITY.md).

## License

MIT
