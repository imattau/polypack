# Data model specification

Version: 2 (stable)

This document is the cross-language behavioural contract for Polypack nodes,
edges, vectors, and the ownership rules that connect them. The TypeScript
implementation in `src/` was the initial reference; the shared conformance
fixtures are now the normative executable contract for TypeScript, Rust, and
Python.

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
| `memoryClass`| `episodic` \| `semantic` \| `procedural` \| `entity` | no | Overrides `NodeTypeDefinition.memoryClass` for this node only (see 1.6). |
| `confidence`| finite f64           | no       | Clamped to `[0, 1]`. Absent means "not tracked," not a stored default of 1. |
| `source`    | UTF-8 string         | no       | Non-empty if present. Free-form provenance label. |
| `observedAt`| integer millis       | no       | Finite, non-negative. May predate `insertedAt` for backfilled data. |
| `derivedFrom`| array of UTF-8 string | no     | Node ids this node was derived/consolidated from. Soft references — not validated for existence. |
| `supersedes`| UTF-8 string         | no       | Non-empty if present. Node id this node supersedes. Soft reference. |
| `contradicts`| array of UTF-8 string | no     | Node ids this node conflicts with. Soft references. |

A node without a vector is distinct from a node whose vector is absent from the
index. Both are valid; similarity operations simply skip vectorless nodes.

### 1.5 Activation

A node may carry an optional `activation` object:

| Field                     | Type         | Constraints                                       |
|---------------------------|--------------|---------------------------------------------------|
| `score`                   | finite f64   | Clamped to `[0, 1]`.                              |
| `importance`              | finite f64   | Clamped to `[0, 1]`.                              |
| `reinforcementCount`      | integer      | Non-negative.                                     |
| `lastMeaningfulActivation`| integer millis | Finite, non-negative decay anchor.                |
| `inhibition`              | finite f64   | Optional. Clamped to `[0, 1]`. Absent is equivalent to 0. |
| `lastInhibitedAt`         | integer millis | Required iff `inhibition` is present. Finite, non-negative decay anchor for `inhibition`, independent of `lastMeaningfulActivation`. |
| `context`                 | object       | Optional. Keyed by an application-defined context id; each entry is `{ score, lastMeaningfulActivation }` (same constraints as the top-level pair), an additional per-context lens on top of the global `score`. |

Semantics (shared across TypeScript, Rust, and Python):

- **Decay** is a pure function of elapsed time anchored at
  `lastMeaningfulActivation`: `decay = 0.5 ** (elapsed / halfLife)` for the
  `score` curve (24 h default) and a slower `importance` curve (30 days).
  `inhibition` decays independently against `lastInhibitedAt` (12 h default
  half-life, shorter than `score` so suppression fades unless reinforced), and
  each `context` entry decays independently against its own anchor (same curve
  as `score` by default). Because decay depends only on stored state and the
  clock, replicas with the same state compute identical current scores. Reads
  decay lazily; nothing is re-written until reinforcement/suppression or an
  explicit `decay()` sweep.
- **Reinforcement** (`reinforceNode`/`reinforce_node`): decay-correct the prior
  state to now, add the delta to `score`, fold a fraction (`importanceGain`,
  0.05) into `importance`, increment `reinforcementCount`, and re-anchor
  `lastMeaningfulActivation` to now. `score`/`importance` clamp to `[0, 1]`.
  An optional `context` argument additionally reinforces
  `activation.context[context]` with the same delta — an independent,
  additional lens, not a replacement for the global score; the global score is
  always reinforced too.
- **Suppression** (`suppressNode`/`suppress_node`) mirrors reinforcement but on
  the `inhibition` axis: decay-correct the prior inhibition to now, add the
  delta, clamp to `[0, 1]`, and re-anchor `lastInhibitedAt`. A negative delta
  releases suppression. Inhibition is subtracted from `score` only at the
  final read/ranking layer (e.g. `ActivationEngine.effective`), never inside
  relational spreading — a suppressed node stays re-evaluable, not permanently
  invisible.
- **Merge** for total-state payloads (e.g. sync snapshots) is a max-merge of the
  decay-corrected components, re-anchored to now — idempotent for re-delivered
  snapshots, including a per-key max-merge over `context` and a max-merge of
  `inhibition`. Concurrent **deltas** accumulate additively instead; activation
  is accumulated knowledge, not last-write-wins data.

### 1.6 Memory class

A node's memory class (`episodic`, `semantic`, `procedural`, or `entity`)
selects which score/importance half-lives its activation decays with, letting
different kinds of memory fade at different rates without a separate decay
mechanism. Resolution order: `node.memoryClass` (an explicit per-node
override) if set, else the owning type's `NodeTypeDefinition.memoryClass` (a
schema-level default registered once via `registerNodeType`/`register_node_type`),
else no class — the node uses the flat, un-differentiated
`scoreHalfLifeMs`/`importanceHalfLifeMs` defaults, unaffected by class.
`NodeTypeDefinition.memoryClass` is schema metadata, so it rides the existing
schema-definitions sync path alongside `dataTypes`/`requiredFields`; it is not
part of the per-node wire format. Only nodes actually reinforced/read through
an `ActivationEngine` use class-based half-lives — the engine resolves the
class and looks up its (possibly overridden) half-life pair before calling the
same decay functions described in 1.5; the underlying decay math itself
remains a pure function of `(activation, now, halfLifeMs)` with no notion of
class.

### 1.7 Confidence and provenance

`confidence`, `source`, `observedAt`, `derivedFrom`, `supersedes`, and
`contradicts` (1.1) are ordinary node fields, not activation state — they
carry no decay curve and are not part of the `activation` object. They are
static until explicitly revised (confidence does not erode on its own; an
application that wants confidence to fall over time should do so by writing a
new value, the same way any other node field is revised). They ride the
existing full-node write paths (`addNode`/`updateNode`/`patchNode` and their
sync operations) exactly like `data`; no additive-merge semantics apply to
them, unlike activation deltas.

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

## 10. Compatibility and conformance

Version 2 is stable for the graph data model. Implementations must preserve
the field meanings, validation order, detached-read behavior, edge identity,
ownership semantics, query ordering, and vector-score rules above. Legacy
records may omit `revision` and are interpreted as revision `0`; newly
materialized records expose the revision explicitly.

The shared fixtures covering node and edge CRUD, ownership and cascade
behavior, detached reads, revisions and patches, traversal, pagination,
aggregation, exact search, and ANN search are the compatibility gate. The
database-core fixtures extend that gate for transactions, independent edge
identity, snapshots, indexes, migrations, resource limits, and durable
mutation logs. Recovery behavior is specified separately in
[`persistence.md`](./persistence.md) and exercised by `fixtures/recovery`.

Changes to observable data-model behavior require a specification version
bump or an explicitly documented backwards-compatible extension, together
with updates to the language-neutral fixtures.
