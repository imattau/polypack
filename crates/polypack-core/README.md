# polypack-core

[![crates.io](https://img.shields.io/crates/v/polypack-core.svg)](https://crates.io/crates/polypack-core)
[![docs.rs](https://docs.rs/polypack-core/badge.svg)](https://docs.rs/polypack-core)

Portable, embedded property-graph and vector-search core for
[polypack](https://github.com/imattau/polypack), written in Rust. It has no
dependency on Node.js or Python and can be embedded directly in any Rust
application.

This crate implements the data model, vector search, query execution, and
persistence engine shared across all polypack language bindings:

- `polypack-node` — NAPI-RS bindings, published to npm as
  [`@0xx0lostcause0xx0/polypack-native`](https://www.npmjs.com/package/@0xx0lostcause0xx0/polypack-native)
- `polypack-python` — PyO3 bindings, published to PyPI as
  [`polypack-db`](https://pypi.org/project/polypack-db/)

The three packages, plus the [TypeScript `@0xx0lostcause0xx0/polypack`](https://www.npmjs.com/package/@0xx0lostcause0xx0/polypack)
package (which reimplements the graph/query layer in TypeScript for
zero-native-dependency use in Node.js and the browser), are version-locked
and released together. See the main
[repository README](https://github.com/imattau/polypack#readme) for the
project overview and the [TypeScript API reference](https://github.com/imattau/polypack/blob/master/docs/API.md)
for the full cross-language semantics (ownership, persisted queries, sync,
etc.) that this crate's data model follows.

## Install

```sh
cargo add polypack-core
```

## What's included

- `model` — `Node`, `Edge`, `VectorEntry`, `ChangeBatch` envelopes with
  camelCase serde mappings matching `specification/data-model.md`.
- `vector` — `ExactIndex`, an exact cosine/euclidean vector index.
- `hnsw` — `HnswIndex`, an update-safe approximate nearest-neighbor index.
- `query` / `query_exec` — a serializable `QueryPlan`/`QueryResult` IR and an
  executor (`execute`, `aggregate`) over an in-memory `GraphSnapshot`.
- `storage` — `Store`, a directory-backed persistence engine (MessagePack
  snapshot + write-ahead log) with atomic batch `apply()`, adaptive
  compaction, and secondary indexes for type and edge-source/target lookups.

## Quick start

```rust
use polypack_core::{ChangeBatch, Node, Store, StoreConfig, InMemoryStorage};
use serde_json::json;

let mut store = Store::new(Box::new(InMemoryStorage::new()), StoreConfig::default());

let node = Node {
    id: "doc_1".into(),
    node_type: "document".into(),
    data: json!({ "title": "Quantum Computing" }).as_object().unwrap().clone(),
    vector: Some(vec![0.95, 0.20, 0.10]),
    inserted_at: 0,
    updated_at: 0,
};

store.apply(&ChangeBatch { put_nodes: vec![node], ..Default::default() })?;

let count = store.node_count()?;
# Ok::<(), polypack_core::PolypackError>(())
```

Swap `InMemoryStorage` for a `Storage` implementation backed by the
filesystem (or any byte store) to get crash-safe, WAL-backed persistence —
this is exactly what `polypack-node` and `polypack-python` do under a
directory-backed `Storage`.

## Status

This crate mirrors the TypeScript reference implementation's semantics but
its own Rust API (types, method names, error enum) is independent and not
guaranteed to match the TypeScript API 1:1. Treat `docs/API.md` in the main
repository as the source of truth for cross-language behavior, and this
crate's `docs.rs` documentation as the source of truth for the Rust surface.

## License

MIT
