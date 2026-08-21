//! polypack-core: a portable embedded property-graph and vector-search core.
//!
//! Node/edge/change-batch envelopes with validation ([`model`]), exact and
//! approximate ([`hnsw`]) vector search, a serialisable query plan and
//! executor ([`query`], [`query_exec`]), and a directory-backed persistence
//! engine with WAL replay, atomic batch commits, and adaptive compaction
//! ([`storage`]). This crate is the shared core behind `polypack-node`
//! (NAPI/npm) and `polypack-python` (PyO3/PyPI); see the
//! [repository README](https://github.com/imattau/polypack#readme) for the
//! project overview and `docs/API.md` for cross-language semantics.

pub mod activation;
pub mod error;
pub mod hnsw;
pub mod model;
pub mod query;
pub mod query_exec;
pub mod rng;
pub mod storage;
pub mod sync;
pub mod vector;

pub use activation::{
    activation_score_of, clamp01, decay_activation_state, decay_factor, merge_activation,
    reinforce_activation, suppress_activation, ActivationCurves, DEFAULT_ACTIVATION,
};
pub use error::{PolypackError, Result};
pub use hnsw::{HnswConfig, HnswIndex};
pub use model::{ChangeBatch, Edge, MemoryClass, Node, NodeActivation, VectorEntry};
pub use query::{QueryPlan, QueryResult};
pub use query_exec::{aggregate, execute, GraphSnapshot};
pub use storage::{Durability, InMemoryStorage, Storage, Store, StoreConfig, WalEntry};
pub use sync::SyncServer;
pub use vector::{ExactIndex, DistanceFn};
