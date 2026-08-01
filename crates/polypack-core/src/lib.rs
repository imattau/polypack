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

pub mod error;
pub mod hnsw;
pub mod model;
pub mod query;
pub mod query_exec;
pub mod rng;
pub mod storage;
pub mod vector;

pub use error::{PolypackError, Result};
pub use hnsw::{HnswConfig, HnswIndex};
pub use model::{ChangeBatch, Edge, Node, VectorEntry};
pub use query::{QueryPlan, QueryResult};
pub use query_exec::{aggregate, execute, GraphSnapshot};
pub use storage::{Durability, InMemoryStorage, Storage, Store, StoreConfig, WalEntry};
pub use vector::{ExactIndex, DistanceFn};
