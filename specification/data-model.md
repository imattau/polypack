# Data model specification

Version: 2 (draft)

This document is the cross-language behavioural contract for Polypack nodes,
edges, vectors, and the ownership rules that connect them. The TypeScript
implementation in `src/` is the reference for this document until the
conformance harness passes for every language.

## 1. Core types

### 1.1 Node

A node is the primary entity of the graph.

| Field       | Type                 | Required | Notes                                            |
|-------------|----------------------|----------|--------------------------------------------------|
| `id`        | UTF-8 string         | yes      | Non-empty.                                       |
| `type`      | UTF-8 string         | yes      | Non-empty.                                       |
| `data`      | object               | yes      | MessagePack-compatible values only.              |
| `vector`    | array of finite f64  | no       | Uniform dimension within an index.               |
| `insertedAt`| integer millis       | yes      | Finite, non-negative.                            |
| `updatedAt` | integer millis       | yes      | Finite, non-negative.                            |
| `revision`  | non-negative integer | no       | Defaults to `0`; increments on successful update. |
| `activation`| object               | no       | Durable activation state (see 1.5).              |

A node without a vector is distinct from a node whose vector is absent from the
index. Both are valid; similarity operations simply skip vectorless nodes.

### 1.5 Activation

A node may carry an optional `activation` object with four fields:

| Field                     | Type         | Constraints                                       |
|---------------------------|--------------|---------------------------------------------------|
| `score`                   | finite f64   | Clamped to `[0, 1]`.                              |
| `importance`              | finite f64   | Clamped to `[0, 1]`.                              |
| `reinforcementCount`      | integer      | Non-negative.                                     |
| `lastMeaningfulActivation`| integer millis | Finite, non-negative decay anchor.                |

Semantics (shared across TypeScript, Rust, and Python):

- **Decay** is a pure function of elapsed time anchored at
  `lastMeaningfulActivation`: `decay = 0.5 ** (elapsed / halfLife)` for the
  `score` curve (24 h default) and a slower `importance` curve (30 days).
  Because it depends only on stored state and the clock, replicas with the same
  state compute identical current scores. Reads decay lazily; nothing is
  re-written until reinforcement or an explicit `decay()` sweep.
- **Reinforcement** (`reinforceNode`/`reinforce_node`): decay-correct the prior
  state to now, add the delta to `score`, fold a fraction (`importanceGain`,
  0.05) into `importance`, increment `reinforcementCount`, and re-anchor
  `lastMeaningfulActivation` to now. `score`/`importance` clamp to `[0, 1]`.
- **Merge** for total-state payloads (e.g. sync snapshots) is a max-merge of the
  decay-corrected components, re-anchored to now — idempotent for re-delivered
  snapshots. Concurrent **deltas** accumulate additively instead; activation is
  accumulated knowledge, not last-write-wins data.

### 1.2 Edge

A directed typed edge between two node IDs.

| Field       | Type                 | Required | Notes                                              |
|-------------|----------------------|----------|----------------------------------------------------|
| `id`        | UTF-8 string         | yes      | Independent edge identity; must be non-empty.      |
| `source`    | UTF-8 string         | yes      | Non-empty node ID.                                  |
| `target`    | UTF-8 string         | yes      | Non-empty node ID.                                  |
| `type`      | UTF-8 string         | yes      | Non-empty edge type.                                |
| `data`      | object \| null       | no       | May carry the reserved ownership key (1.4).        |
| `createdAt` | integer millis       | yes      | Finite, non-negative.                              |
| `revision`  | non-negative integer | no       | Defaults to `0`; increments on successful update. |

### 1.3 Edge identity

Edge identity is independent from adjacency. Multiple edges may share the
same `(source, type, target)` triple. Implementations maintain adjacency
indexes from that triple to one or more edge IDs. The historical
`source::type::target` value remains a deterministic compatibility helper, but
is not required as the canonical edge ID.

### 1.4 Edge ownership

An edge may declare an ownership mode through a reserved `data` key
(`__ownership`). Absence of the key means `reference`.

| Mode        | Effect                                                                 |
|-------------|------------------------------------------------------------------------|
| `owned`     | Removing the source node, or removing this edge, removes the target — unless the target has another incoming `owned` edge from a different source. |
| `shared`    | Removing this edge notifies the orphan hook when the target becomes disconnected (no incoming edges remain). No cascade delete. |
| `reference` | No lifecycle effect.                                                   |

Cascade deletion is recursive and cycle-safe: each node is removed at most once
per cascade, so cyclic owned edges (`A -> B -> A`) terminate.

## 2. Validation rules

All validation happens before mutation; invalid input throws rather than being
silently coerced.

- Node and edge IDs, and edge source/type/target, must be non-empty strings.
- Edge source, target, and type are independent strings; the separator helper
  is not a validation requirement.
- Timestamps must be finite, non-negative integers (milliseconds).
- Revisions must be non-negative integers when present.
- Vectors must contain only finite values.
- A similarity query vector and every stored vector in the queried index must
  share the same dimension; a mismatch throws.
- Query `topK`, `offset`, `limit`, and traversal `depth` must be non-negative
  integers.
- Similarity `threshold` must be finite.

## 3. Detached reads

All public reads (node getters, query results, vector reads, edge reads) return
detached copies. Mutating a returned node, its `data`, or its vector must not
affect internal state, change tracking, or the persistence queue. Mutate graph
state only through graph and index methods.

## 4. Change semantics

- Adding a node with an existing ID replaces the node.
- Adding an edge with an existing ID is a no-op for the edge's existence (the
  first insertion wins).
- Removing a missing node or edge is a no-op.
- Removing a node also removes its vector and all incident edges, then applies
  ownership rules to connected nodes.
- A node vector can exist without the node being considered vectorless; vectors
  are managed through the vector index and the graph together.

## 5. Query semantics

Queries filter, traverse, rank, and paginate **loaded** nodes. Persisted
queries apply the same predicate language across the full backing store.

Pipeline order:

1. Source selection (node type, or edge source/target constraints).
2. Attribute equality and range filters.
3. Join filters.
4. Traversal expansion (BFS, cycle-safe, bounded by `depth`).
5. Ordering.
6. Similarity ranking (only nodes with vectors; score is the configured
   similarity function; results below `threshold` are dropped; `topK` limits).
7. `offset` then `limit`.

Ordering compares numeric field values; missing or non-numeric values sort as 0
for ascending, the negation for descending.

## 6. Vector search

- **Exact search** returns the true top-K by the configured similarity.
- **ANN search (HNSW)** returns approximate top-K. Recall and ranking are
  bounded by tolerances defined in the conformance harness, not required to be
  identical to exact results.
- Similarity functions: cosine (default) and Euclidean (`1 / (1 + d)`). Scores
  are in `[0, 1]` for both, larger is better.

## 7. Hot cache

The graph loads at most `hotCacheMax` nodes into memory (default 50,000).
Evicted dirty nodes are retained until persistence completes. Edges remain
indexed while nodes are evicted. `getNodeSafe`/`removeNodeSafe` restore an
evicted node before operating on it.

## 8. Persistence envelope

Persistence uses the serialisable envelopes defined in
[`change-batch.schema.json`](./change-batch.schema.json) and
[`query-plan.schema.json`](./query-plan.schema.json). See
[`persistence.md`](./persistence.md) for the on-disk format and durability
model, and [`errors.md`](./errors.md) for the error taxonomy.

## 9. MessagePack compatibility

All values crossing the language core must be MessagePack-safe: null, boolean,
integer, float, string, array, and map. Host-only values (`Blob`, `File`,
class instances, callbacks) are stored in host sidecars or require a
`DataTransform`; they never cross the core boundary.
