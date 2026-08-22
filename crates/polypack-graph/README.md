# polypack-graph

[![crates.io](https://img.shields.io/crates/v/polypack-graph.svg)](https://crates.io/crates/polypack-graph)
[![docs.rs](https://docs.rs/polypack-graph/badge.svg)](https://docs.rs/polypack-graph)

A stateful property-graph engine for Rust applications, built on
[`polypack-core`](https://crates.io/crates/polypack-core). It has no
dependency on Node.js or Python and can be embedded directly in any Rust
application.

This crate is the Rust counterpart to the TypeScript `PolyGraph` class in
[polypack](https://github.com/imattau/polypack) — the same node/edge model,
persistence lifecycle, query builders, and embedding hooks, expressed as a
native Rust API rather than ported line-for-line. It depends only on
`polypack-core`; nothing else in the polypack workspace depends on it, so it
evolves independently of the published NAPI/PyO3 bindings.

## Install

```sh
cargo add polypack-graph
```

## What's included

- **`Graph`** — a hot in-memory node/edge working set backed by a
  `polypack_core::Store` for durability and an `HnswIndex` for vector
  search. Node/edge CRUD (`add_node`, `add_edge`, `remove_node`,
  `remove_edges`, ...) with `owned`/`shared`/`reference` edge-ownership
  semantics — an `owned` edge cascade-deletes its target when no other
  owning source remains; a `shared` edge fires an `on_orphan` callback
  instead.
- **Persistence lifecycle** — `flush`/`warm`/`save`/`clear`/`dispose`/`prune`,
  plus `get_node_safe`/`remove_node_safe`/`update_node_safe` variants that
  restore a node from the `Store` if it's fallen out of the hot working set.
  Hot-cache eviction is LRU-ordered via an O(1) intrusive linked-list
  (`touch`/`remove`/`pop_front`), not a linear scan.
- **`GraphQuery`** / **`PersistedGraphQuery`** — fluent, chainable query
  builders over the hot working set and over the full persisted store
  respectively: attribute/type/edge filters, `order_by`, `limit`/`offset`,
  BFS `traverse`, cosine `similar_to`, `join`, plus relational extensions
  (`pluck`, `aggregate`, `group_aggregate`, `group_by_vector`, `collect`,
  `unique_keys`).
- **`EmbeddingProvider`** / **`FeatureHashEmbedding`** — a pluggable
  text-embedding trait, plus a dependency-free deterministic hashing-trick
  provider (verified bit-for-bit identical to the TypeScript reference
  implementation) wired into `Graph::add_node_with_embedding`,
  `update_node_with_embedding`, `query_text`, `query_persisted_text`, and
  `search_nodes`.
- **Activation (adaptive memory)** — durable per-node relevance
  (`NodeActivation` with `score`/`importance`/`reinforcement_count`/
  `last_meaningful_activation`, plus an independently-decaying `inhibition`
  axis and per-context `context` entries) persisted through the core `Store`;
  graph primitives `reinforce_node`/`reinforce_node_safe`/
  `reinforce_node_in_context` (decay-correct, re-anchor, emit
  `ActivationUpdated`), `suppress_node`/`suppress_node_safe` (the inhibition
  axis, emits `InhibitionUpdated`), `get_activation`/`get_activation_state`/
  `get_context_activation`, `top_activated`, `decay`, `supersede` (records a
  contradiction and suppresses the superseded node), and `consolidate`
  (promotes a group of nodes into a higher-level one via `derived_from` +
  suppression); `where_activated`/`order_by_activation` on both query
  builders; nodes also carry an optional `memory_class`
  (`episodic`/`semantic`/`procedural`/`entity`, resolved from the node or its
  type's `NodeTypeDefinition.memory_class`) driving differentiated decay
  half-lives, and confidence/provenance fields (`confidence`, `source`,
  `observed_at`, `derived_from`, `supersedes`, `contradicts`); and an
  `ActivationEngine` composing spreading activation, semantic `pulse`/
  `absorb`, transient attention, class-aware `effective`/`resolve_score_half_life`,
  budgeted/diverse `working_memory_with_options`, and feedback-driven learned
  scoring weights via `record_feedback`, exportable/restorable through
  `weights`/`set_weights`; `working_memory_persisted` ranks cold-store nodes,
  `estimate_node_tokens` supplies a default budget cost, and
  `score_breakdown_of` exposes explainable signal contributions. Decay math and `merge_activation`
  live in `polypack_core::activation`, matching the TypeScript and Python
  implementations exactly.

## Quick start

```rust
use polypack_graph::{EdgeOwnership, Graph, GraphConfig};
use polypack_core::{InMemoryStorage, Node, StoreConfig};
use serde_json::json;

let mut graph = Graph::open(
    Box::new(InMemoryStorage::new()),
    StoreConfig::default(),
    GraphConfig::default(),
)?;

graph.add_node(Node {
    id: "doc_1".into(),
    node_type: "document".into(),
    data: json!({ "title": "Quantum Computing" }).as_object().unwrap().clone(),
    vector: Some(vec![0.95, 0.20, 0.10]),
    inserted_at: 0,
    updated_at: 0,
})?;
graph.add_node(Node {
    id: "doc_2".into(),
    node_type: "document".into(),
    data: json!({ "title": "Baking Sourdough" }).as_object().unwrap().clone(),
    vector: Some(vec![0.05, 0.90, 0.30]),
    inserted_at: 0,
    updated_at: 0,
})?;
graph.add_edge("doc_1", "CITES", "doc_2", None, EdgeOwnership::Reference)?;

let similar = graph.query().similar_to(vec![0.9, 0.2, 0.1], 0.5, Some(5)).to_array();

graph.flush()?; // persist to the Store

# Ok::<(), polypack_core::PolypackError>(())
```

Swap `InMemoryStorage` for a `Storage` implementation backed by the
filesystem (or any byte store) to get crash-safe, WAL-backed persistence —
call `graph.warm()` on startup to reload the hot working set from it.

## Status

This crate mirrors the TypeScript `PolyGraph`'s semantics but its own Rust
API (types, method names, error handling) is independent and not guaranteed
to match the TypeScript API 1:1 — notably, embedding text-transform/sidecar
serialization hooks and event-batching ergonomics differ where Rust's type
system and lack of a JS event loop call for a different shape. Treat
`docs/API.md` in the main repository as the source of truth for
cross-language behavior, and this crate's `docs.rs` documentation as the
source of truth for the Rust surface.

## License

MIT
