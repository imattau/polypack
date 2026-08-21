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
mod activation;
mod edge;
mod embedding;
mod event;
mod graph;
mod lru;
mod migration;
mod persisted_query;
mod query;

pub use activation::{ActivationConfig, ActivationEngine, ActivationWeights, PulseOptions, SpreadOptions};
pub use edge::{EdgeEntry, EdgeOwnership};
pub use embedding::{
    build_embedding_text, create_embedding, EmbeddingProvider, FeatureHashEmbedding, FeatureHashEmbeddingOptions,
};
pub use event::GraphChangeEvent;
pub use graph::{EdgeTypeDefinition, Graph, GraphConfig, GraphResourceLimits, GraphStats, IndexDefinition, NodeTypeDefinition};
pub use migration::{MigrationDefinition, MigrationOptions, MigrationProgress, MigrationRegistry, MigrationReport};
pub use persisted_query::{PersistedGraphQuery, QueryExplain, QueryResourceLimits};
pub use query::{AggregateOp, AggregateResult, GraphQuery, GroupedRow, OrderDirection};
// `Direction` is a required parameter of `GraphQuery`/`PersistedGraphQuery`'s
// `traverse`/`join`, so it must be nameable without adding `polypack-core` as
// a separate direct dependency.
pub use polypack_core::query::Direction;
pub use polypack_core::{merge_activation, NodeActivation};
pub use polypack_core::query_exec::GraphSnapshot;
pub use polypack_core::storage::VerificationReport;
