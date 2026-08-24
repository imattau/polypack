//! Persistence: snapshot/WAL codecs (byte-compatible with the TypeScript
//! reference) and the storage state machine. Hosts provide byte I/O through
//! the [`Storage`] trait.

pub mod format;
pub mod file;
pub mod migration;
pub mod msgpack;
pub mod store;
pub mod wal;

pub use store::{
    AdapterCapabilities, Durability, InMemoryStorage, MutationLogRetention, MutationOperation, MutationRecord, NodeQuery,
    OrderBy, RangeQuery,
    EdgeTypeDefinition, NodeTypeDefinition, SecondaryIndexDefinition, Storage, Store, StoreConfig, VerificationReport,
    VectorSearchCapability,
    DEFAULT_COMPACT_THRESHOLD, INDEXES_FILE, MUTATION_LOG_FILE, SCHEMAS_FILE, SNAPSHOT_FILE, WAL_FILE,
};
pub use file::FileStorage;
pub use migration::{migrate_storage, FormatArtifact, FormatMigrationRegistry, FormatMigrationReport};
pub use wal::WalEntry;
