//! polypack-core: a portable embedded property-graph and vector-search core.
//!
//! Phase-2 spike scope (see POLYPACK_RUST_PYTHON_PLAN): node/edge/change-batch
//! envelopes with validation, exact vector search, an update-safe HNSW index,
//! serialisable query/result envelopes, and a reproducible benchmark binary.
//! In-memory operation only; persistence, hosts, and transports are out of
//! scope for the spike.

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
