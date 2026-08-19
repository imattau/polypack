//! Persistence: snapshot/WAL codecs (byte-compatible with the TypeScript
//! reference) and the storage state machine. Hosts provide byte I/O through
//! the [`Storage`] trait.

pub mod format;
pub mod msgpack;
pub mod store;
pub mod wal;

pub use store::{
    AdapterCapabilities, Durability, InMemoryStorage, NodeQuery, OrderBy, RangeQuery, Storage, Store, StoreConfig,
    VerificationReport,
    VectorSearchCapability,
    DEFAULT_COMPACT_THRESHOLD, SNAPSHOT_FILE, WAL_FILE,
};
pub use wal::WalEntry;
