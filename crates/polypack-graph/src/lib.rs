//! polypack-graph: stateful property-graph orchestration on top of
//! `polypack-core`.
//!
//! This crate is the Rust counterpart to the TypeScript `PolyGraph` class
//! (`src/graph.ts`) — it owns a hot in-memory node/edge working set, an
//! `HnswIndex` for vector search, and drives a `polypack_core::Store` for
//! durability. It depends on `polypack-core` but nothing in `polypack-core`,
//! `polypack-node`, or `polypack-python` depends on it, so it can evolve
//! independently of the published NAPI/PyO3 bindings.
//!
mod edge;
mod embedding;
mod event;
mod graph;
mod lru;
mod persisted_query;
mod query;

pub use edge::{EdgeEntry, EdgeOwnership};
pub use embedding::{
    build_embedding_text, create_embedding, EmbeddingProvider, FeatureHashEmbedding, FeatureHashEmbeddingOptions,
};
pub use event::GraphChangeEvent;
pub use graph::{Graph, GraphConfig};
pub use persisted_query::PersistedGraphQuery;
pub use query::{AggregateOp, AggregateResult, GraphQuery, GroupedRow, OrderDirection};
