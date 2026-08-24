//! The persistence state machine, mirroring the TypeScript `BinaryStoreAdapter`
//! semantics: serialised load/apply/compact/close, WAL replay then
//! snapshot-before-WAL-delete, generation-boundary compaction, truncated-tail
//! tolerance, and version checking. Hosts own byte I/O via the `Storage` trait.

use crate::error::{PolypackError, Result};
use crate::model::{validate_batch, ChangeBatch, Edge, MemoryClass, Node};
use crate::storage::format::{decode_snapshot, decode_wal, encode_snapshot, encode_wal};
use crate::storage::wal::WalEntry;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const SNAPSHOT_FILE: &str = "snapshot.msgpack";
pub const WAL_FILE: &str = "wal.msgpack";
pub const MUTATION_LOG_FILE: &str = "mutations.jsonl";
pub const INDEXES_FILE: &str = "indexes.json";
pub const SCHEMAS_FILE: &str = "schemas.json";
pub const DEFAULT_COMPACT_THRESHOLD: usize = 10_000;
/// Compact once the WAL holds at least this share of the store's record count.
const COMPACT_RATIO: usize = 4;

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MutationOperation {
    pub operation_type: String,
    pub payload: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MutationRecord {
    pub operation_id: String,
    pub sequence: u64,
    pub transaction_id: String,
    pub timestamp: i64,
    pub operations: Vec<MutationOperation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

fn mutation_operations(changes: &ChangeBatch) -> Vec<MutationOperation> {
    let mut operations = Vec::new();
    for node in &changes.put_nodes {
        operations.push(MutationOperation { operation_type: "putNode".into(), payload: serde_json::to_value(node).unwrap_or(serde_json::Value::Null) });
    }
    for id in &changes.delete_node_ids {
        operations.push(MutationOperation { operation_type: "deleteNode".into(), payload: serde_json::json!({ "id": id }) });
    }
    for edge in &changes.put_edges {
        operations.push(MutationOperation { operation_type: "putEdge".into(), payload: serde_json::to_value(edge).unwrap_or(serde_json::Value::Null) });
    }
    for id in &changes.delete_edge_ids {
        operations.push(MutationOperation { operation_type: "deleteEdge".into(), payload: serde_json::json!({ "id": id }) });
    }
    for vector in &changes.put_vectors {
        operations.push(MutationOperation { operation_type: "putVector".into(), payload: serde_json::to_value(vector).unwrap_or(serde_json::Value::Null) });
    }
    for id in &changes.delete_vector_ids {
        operations.push(MutationOperation { operation_type: "deleteVector".into(), payload: serde_json::json!({ "id": id }) });
    }
    operations
}

/// Numeric-range predicate used by [`NodeQuery`], matching the TypeScript
/// `PersistedNodeQuery.attributeRanges` semantics (exclusive bounds).
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RangeQuery {
    #[serde(default)]
    pub above: Option<f64>,
    #[serde(default)]
    pub below: Option<f64>,
}

/// Sort order used by [`NodeQuery`], mirroring `PersistedNodeQuery.orderBy`.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OrderBy {
    pub field: String,
    /// "asc" | "desc"
    pub direction: String,
}

/// Storage-level node predicate mirroring the TypeScript
/// `PersistedNodeQuery`. `attributes` match `node.data`, except the special
/// key `"type"` which matches `node.node_type`.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NodeQuery {
    /// Optional candidate ids selected by a higher-level persisted index.
    /// The store still applies every predicate, so this is an optimization,
    /// not a correctness shortcut.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_ids: Option<Vec<String>>,
    #[serde(default)]
    pub node_types: Option<Vec<String>>,
    #[serde(default)]
    pub attributes: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub attribute_ranges: Option<HashMap<String, RangeQuery>>,
    #[serde(default)]
    pub order_by: Option<OrderBy>,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
    /// Maximum candidate records examined by this query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_nodes_visited: Option<usize>,
    /// Maximum records returned before pagination is applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_result_size: Option<usize>,
}

/// A persisted index over node data fields. Compound fields are encoded in
/// declaration order and may be scoped to one node type.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SecondaryIndexDefinition {
    pub name: String,
    pub fields: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
    #[serde(default)]
    pub unique: bool,
    #[serde(default)]
    pub sparse: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NodeTypeDefinition {
    pub node_type: String,
    #[serde(default)]
    pub required_fields: Vec<String>,
    #[serde(default)]
    pub data_types: HashMap<String, String>,
    /// Default memory class for nodes of this type. Overridable per node via `Node.memory_class`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_class: Option<MemoryClass>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct EdgeTypeDefinition {
    pub edge_type: String,
    #[serde(default)]
    pub source_types: Vec<String>,
    #[serde(default)]
    pub target_types: Vec<String>,
    #[serde(default)]
    pub required_fields: Vec<String>,
    #[serde(default)]
    pub data_types: HashMap<String, String>,
    #[serde(default)]
    pub cardinality: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct SchemaMetadata {
    node_types: Vec<NodeTypeDefinition>,
    edge_types: Vec<EdgeTypeDefinition>,
}

fn node_field(node: &Node, field: &str) -> serde_json::Value {
    if field == "type" {
        serde_json::Value::String(node.node_type.clone())
    } else {
        field.split('.').try_fold(serde_json::Value::Object(node.data.clone()), |value, part| {
            value.get(part).cloned()
        }).unwrap_or(serde_json::Value::Null)
    }
}

fn matches_node(node: &Node, query: &NodeQuery) -> bool {
    if let Some(types) = &query.node_types {
        if !types.iter().any(|t| t == &node.node_type) {
            return false;
        }
    }
    if let Some(attributes) = &query.attributes {
        for (key, expected) in attributes {
            if &node_field(node, key) != expected {
                return false;
            }
        }
    }
    if let Some(ranges) = &query.attribute_ranges {
        for (key, range) in ranges {
            let Some(value) = node_field(node, key).as_f64() else {
                return false;
            };
            if let Some(above) = range.above {
                if value <= above {
                    return false;
                }
            }
            if let Some(below) = range.below {
                if value >= below {
                    return false;
                }
            }
        }
    }
    true
}

fn numeric_field(node: &Node, field: &str) -> f64 {
    node.data.get(field).and_then(|v| v.as_f64()).filter(|x| x.is_finite()).unwrap_or(0.0)
}

/// How far a write is guaranteed to survive. Only `Fsync` changes `Store`'s
/// own behavior (it calls `Storage::sync`/`sync_dir` after WAL appends and
/// snapshot writes); `Memory` and `Process` otherwise depend entirely on
/// what the attached `Storage` implementation actually does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    /// Not persisted at all (e.g. `InMemoryStorage`).
    Memory,
    /// Written to the OS but not fsynced — survives process crash, not power loss. Default.
    Process,
    /// fsynced to disk after every WAL append and snapshot write — survives power loss, at a throughput cost.
    Fsync,
}

/// Host-owned byte-stream storage. Implementations map names to files. Send +
/// Sync so the state machine can be used from bindings.
pub trait Storage: Send + Sync {
    fn read(&self, name: &str) -> Result<Option<Vec<u8>>>;
    fn write(&mut self, name: &str, data: &[u8]) -> Result<()>;
    fn append(&mut self, name: &str, data: &[u8]) -> Result<()>;
    fn delete(&mut self, name: &str) -> Result<()>;
    fn exists(&self, name: &str) -> Result<bool>;
    /// fsync a named file; no-op for in-memory/process stores.
    fn sync(&self, _name: &str) -> Result<()> {
        Ok(())
    }
    /// fsync the containing directory; best effort.
    fn sync_dir(&self) -> Result<()> {
        Ok(())
    }
    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorSearchCapability {
    None,
    Exact,
    Ann,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterCapabilities {
    pub atomic_batches: bool,
    pub transactions: bool,
    pub fsync: bool,
    pub secondary_indexes: bool,
    pub snapshots: bool,
    pub change_feed: bool,
    pub concurrent_writers: bool,
    pub vector_search: VectorSearchCapability,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerificationReport {
    pub ok: bool,
    pub errors: Vec<String>,
    pub node_count: usize,
    pub edge_count: usize,
    pub vector_count: usize,
    pub mutation_count: usize,
}

impl Default for AdapterCapabilities {
    fn default() -> Self {
        Self {
            atomic_batches: false,
            transactions: false,
            fsync: false,
            secondary_indexes: false,
            snapshots: false,
            change_feed: false,
            concurrent_writers: false,
            vector_search: VectorSearchCapability::None,
        }
    }
}

/// A `Storage` implementation that never touches disk. Useful for tests and
/// ephemeral graphs; wrap in `Arc<Mutex<_>>` (a `Storage` impl is provided)
/// to share it across threads.
#[derive(Default)]
pub struct InMemoryStorage {
    files: HashMap<String, Vec<u8>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        InMemoryStorage::default()
    }
}

impl<T: Storage + ?Sized> Storage for &mut T {
    fn read(&self, name: &str) -> Result<Option<Vec<u8>>> {
        (**self).read(name)
    }
    fn write(&mut self, name: &str, data: &[u8]) -> Result<()> {
        (**self).write(name, data)
    }
    fn append(&mut self, name: &str, data: &[u8]) -> Result<()> {
        (**self).append(name, data)
    }
    fn delete(&mut self, name: &str) -> Result<()> {
        (**self).delete(name)
    }
    fn exists(&self, name: &str) -> Result<bool> {
        (**self).exists(name)
    }
    fn sync(&self, name: &str) -> Result<()> {
        (**self).sync(name)
    }
    fn sync_dir(&self) -> Result<()> {
        (**self).sync_dir()
    }
    fn capabilities(&self) -> AdapterCapabilities {
        (**self).capabilities()
    }
}

impl Storage for std::sync::Arc<std::sync::Mutex<InMemoryStorage>> {
    fn read(&self, name: &str) -> Result<Option<Vec<u8>>> {
        self.lock().unwrap().read(name)
    }
    fn write(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.lock().unwrap().write(name, data)
    }
    fn append(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.lock().unwrap().append(name, data)
    }
    fn delete(&mut self, name: &str) -> Result<()> {
        self.lock().unwrap().delete(name)
    }
    fn exists(&self, name: &str) -> Result<bool> {
        self.lock().unwrap().exists(name)
    }
    fn sync(&self, name: &str) -> Result<()> {
        self.lock().unwrap().sync(name)
    }
    fn sync_dir(&self) -> Result<()> {
        self.lock().unwrap().sync_dir()
    }
    fn capabilities(&self) -> AdapterCapabilities {
        self.lock().unwrap().capabilities()
    }
}

impl Storage for InMemoryStorage {
    fn read(&self, name: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.files.get(name).cloned())
    }
    fn write(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.files.insert(name.to_string(), data.to_vec());
        Ok(())
    }
    fn append(&mut self, name: &str, data: &[u8]) -> Result<()> {
        // `entry(..).or_default()` extends the buffer in place instead of
        // cloning the full accumulated file on every append (that clone made
        // append — and so every `flush()` — O(total appended bytes so far),
        // i.e. O(n^2) over n flushes; see
        // benchmarks/database-core-edge-flush-report.md).
        self.files.entry(name.to_string()).or_default().extend_from_slice(data);
        Ok(())
    }
    fn delete(&mut self, name: &str) -> Result<()> {
        self.files.remove(name);
        Ok(())
    }
    fn exists(&self, name: &str) -> Result<bool> {
        Ok(self.files.contains_key(name))
    }
    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            atomic_batches: true,
            transactions: true,
            secondary_indexes: true,
            snapshots: true,
            vector_search: VectorSearchCapability::Exact,
            ..Default::default()
        }
    }
}

pub struct StoreConfig {
    /// Minimum WAL-entry count at which compaction is scheduled. Acts as a
    /// lower bound — the effective threshold also grows with the store
    /// (`max(compact_threshold, records / COMPACT_RATIO)`), so a large store
    /// doesn't rewrite its snapshot on every batch. Default 10,000.
    pub compact_threshold: usize,
    pub durability: Durability,
    /// Retention policy applied to `mutations.jsonl` each time `compact()`
    /// runs (i.e. on the same cadence as WAL compaction, and on `close()`).
    /// `None` (the default) keeps the full mutation history forever,
    /// matching prior behavior. Trimming drops records from the front of
    /// the log, so a sync consumer resuming from a sequence older than the
    /// retained window will miss mutations — only enable this when nothing
    /// depends on `mutation_log_since`/`mutation_log_page` reaching back
    /// further than the configured window (e.g. replicas re-snapshot
    /// instead of tailing arbitrarily far behind).
    pub mutation_log_retention: Option<MutationLogRetention>,
}

/// Retention policy for `mutations.jsonl`. When both bounds are set, a
/// record is retained only if it satisfies both (the intersection) — i.e. a
/// record is dropped as soon as it falls outside either bound.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MutationLogRetention {
    /// Keep at most this many of the newest records.
    pub max_entries: Option<usize>,
    /// Keep only records newer than this age, in milliseconds.
    pub max_age_ms: Option<i64>,
}

impl Default for StoreConfig {
    fn default() -> Self {
        StoreConfig {
            compact_threshold: DEFAULT_COMPACT_THRESHOLD,
            durability: Durability::Process,
            mutation_log_retention: None,
        }
    }
}

/// The persistence state machine: an in-memory graph backed by a snapshot +
/// WAL on a `Storage` implementation. See the module docs for the
/// load/apply/compact/close lifecycle.
pub struct Store {
    nodes: HashMap<String, Node>,
    edges: HashMap<String, Edge>,
    vectors: HashMap<String, Vec<f64>>,
    by_type: HashMap<String, HashSet<String>>,
    index_definitions: HashMap<String, SecondaryIndexDefinition>,
    secondary_indexes: HashMap<String, HashMap<String, HashSet<String>>>,
    node_type_definitions: HashMap<String, NodeTypeDefinition>,
    edge_type_definitions: HashMap<String, EdgeTypeDefinition>,
    edges_by_source: HashMap<String, HashSet<String>>,
    edges_by_target: HashMap<String, HashSet<String>>,
    wal_entry_count: usize,
    config: StoreConfig,
    storage: Box<dyn Storage>,
    closed: bool,
    loaded: bool,
    mutation_sequence: u64,
    seen_operation_ids: HashSet<String>,
    seen_transaction_ids: HashSet<String>,
}

impl Store {
    pub fn new(storage: Box<dyn Storage>, config: StoreConfig) -> Self {
        Store {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            vectors: HashMap::new(),
            by_type: HashMap::new(),
            index_definitions: HashMap::new(),
            secondary_indexes: HashMap::new(),
            node_type_definitions: HashMap::new(),
            edge_type_definitions: HashMap::new(),
            edges_by_source: HashMap::new(),
            edges_by_target: HashMap::new(),
            wal_entry_count: 0,
            config,
            storage,
            closed: false,
            loaded: false,
            mutation_sequence: 0,
            seen_operation_ids: HashSet::new(),
            seen_transaction_ids: HashSet::new(),
        }
    }

    pub fn capabilities(&self) -> AdapterCapabilities {
        self.storage.capabilities()
    }

    pub fn read_auxiliary(&mut self, name: &str) -> Result<Option<Vec<u8>>> {
        self.assert_open()?;
        self.storage.read(name)
    }

    pub fn write_auxiliary(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.assert_open()?;
        self.storage.write(name, data)?;
        if self.config.durability == Durability::Fsync {
            self.storage.sync(name)?;
            self.storage.sync_dir()?;
        }
        Ok(())
    }

    pub fn delete_auxiliary(&mut self, name: &str) -> Result<()> {
        self.assert_open()?;
        self.storage.delete(name)
    }

    pub fn mutation_log(&mut self) -> Result<Vec<MutationRecord>> {
        self.assert_open()?;
        let Some(data) = self.storage.read(MUTATION_LOG_FILE)? else { return Ok(Vec::new()) };
        let lines: Vec<&[u8]> = data.split(|byte| *byte == b'\n').filter(|line| !line.is_empty()).collect();
        let last_index = lines.len().checked_sub(1);
        let mut records = Vec::with_capacity(lines.len());
        for (index, line) in lines.into_iter().enumerate() {
            match serde_json::from_slice(line) {
                Ok(record) => records.push(record),
                // A crash mid-append can leave a truncated final line; tolerate that
                // one case so an otherwise-intact log still loads. Any other line
                // failing to parse is genuine corruption and must still error.
                Err(_) if Some(index) == last_index => break,
                Err(error) => return Err(PolypackError::CorruptData(format!("mutation log: {error}"))),
            }
        }
        Ok(records)
    }

    pub fn mutation_log_since(&mut self, sequence: u64) -> Result<Vec<MutationRecord>> {
        Ok(self.mutation_log()?.into_iter().filter(|record| record.sequence > sequence).collect())
    }

    pub fn mutation_log_page(&mut self, sequence: u64, limit: usize) -> Result<Vec<MutationRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        Ok(self.mutation_log()?.into_iter().filter(|record| record.sequence > sequence).take(limit).collect())
    }

    /// Return the highest acknowledged logical mutation sequence, or zero for
    /// a store with no durable mutations.
    pub fn latest_mutation_sequence(&mut self) -> Result<u64> {
        Ok(self.mutation_log()?.last().map(|record| record.sequence).unwrap_or(0))
    }

    /// Create a consistent administrative backup in another byte storage.
    /// The recovery WAL is compacted first; logical mutations and auxiliary
    /// metadata are copied alongside the snapshot.
    pub fn backup(&mut self, destination: &mut dyn Storage) -> Result<()> {
        self.compact()?;
        for name in [SNAPSHOT_FILE, WAL_FILE, MUTATION_LOG_FILE, INDEXES_FILE, SCHEMAS_FILE] {
            if let Some(data) = self.storage.read(name)? {
                destination.write(name, &data)?;
            } else {
                destination.delete(name)?;
            }
        }
        Ok(())
    }

    /// Restore a store from an adapter-neutral backup and validate that it
    /// can be loaded before returning it to the caller.
    pub fn restore(source: &dyn Storage, destination: Box<dyn Storage>, config: StoreConfig) -> Result<Self> {
        let mut store = Store::new(destination, config);
        for name in [SNAPSHOT_FILE, WAL_FILE, MUTATION_LOG_FILE, INDEXES_FILE, SCHEMAS_FILE] {
            if let Some(data) = source.read(name)? {
                store.storage.write(name, &data)?;
            }
        }
        store.ensure_loaded()?;
        Ok(store)
    }

    pub fn verify(&mut self) -> Result<VerificationReport> {
        self.assert_open()?;
        let mut errors = Vec::new();
        let mut node_count = 0;
        let mut edge_count = 0;
        let mut vector_count = 0;
        if let Some(data) = self.storage.read(SNAPSHOT_FILE)? {
            match decode_snapshot(&data) {
                Ok(snapshot) => {
                    node_count = snapshot.nodes.len();
                    edge_count = snapshot.edges.len();
                    vector_count = snapshot.vectors.len();
                    let node_ids: HashSet<String> = snapshot.nodes.iter().map(|(id, _)| id.clone()).collect();
                    for (id, node) in &snapshot.nodes {
                        if let Err(error) = crate::model::validate_node(node) {
                            errors.push(format!("node {id}: {error}"));
                        }
                    }
                    for (id, edge) in &snapshot.edges {
                        if let Err(error) = crate::model::validate_edge(edge) {
                            errors.push(format!("edge {id}: {error}"));
                        }
                        if !node_ids.contains(&edge.source) {
                            errors.push(format!("edge {id}: missing source {}", edge.source));
                        }
                        if !node_ids.contains(&edge.target) {
                            errors.push(format!("edge {id}: missing target {}", edge.target));
                        }
                    }
                    for (id, vector) in &snapshot.vectors {
                        if !vector.iter().all(|value| value.is_finite()) {
                            errors.push(format!("vector {id}: contains non-finite values"));
                        }
                    }
                }
                Err(error) => errors.push(format!("snapshot: {error}")),
            }
        }
        if let Some(data) = self.storage.read(INDEXES_FILE)? {
            match serde_json::from_slice::<Vec<serde_json::Value>>(&data) {
                Ok(definitions) => {
                    let mut names = HashSet::new();
                    for (position, definition) in definitions.iter().enumerate() {
                        let Some(name) = definition.get("name").and_then(|value| value.as_str()) else {
                            errors.push(format!("index metadata: index {position}: missing name"));
                            continue;
                        };
                        if name.is_empty() || !names.insert(name.to_string()) {
                            errors.push(format!("index metadata: index {name}: duplicate or empty name"));
                        }
                        if definition.get("fields").and_then(|value| value.as_array()).is_none_or(|fields| {
                            fields.is_empty() || fields.iter().filter_map(|field| field.as_str()).collect::<HashSet<_>>().len() != fields.len()
                        }) {
                            errors.push(format!("index metadata: index {name}: fields must not be empty"));
                        }
                    }
                }
                Err(error) => errors.push(format!("index metadata: {error}")),
            }
        }
        let mutation_count = self.mutation_log()?.len();
        Ok(VerificationReport {
            ok: errors.is_empty(),
            errors,
            node_count,
            edge_count,
            vector_count,
            mutation_count,
        })
    }

    // ── secondary indexes ──

    fn data_value(data: Option<&serde_json::Map<String, serde_json::Value>>, field: &str) -> Option<serde_json::Value> {
        let mut value = serde_json::Value::Object(data.cloned().unwrap_or_default());
        for part in field.split('.') {
            value = value.get(part)?.clone();
        }
        Some(value)
    }

    fn matches_data_type(value: &serde_json::Value, expected: &str) -> bool {
        match expected {
            "string" => value.is_string(),
            "number" => value.as_f64().is_some(),
            "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
            "boolean" => value.is_boolean(),
            "object" => value.is_object(),
            "array" => value.is_array(),
            _ => false,
        }
    }

    fn validate_node_schema(&self, node: &Node) -> Result<()> {
        let Some(definition) = self.node_type_definitions.get(&node.node_type) else { return Ok(()) };
        for field in &definition.required_fields {
            if Self::data_value(Some(&node.data), field).is_none() {
                return Err(PolypackError::InvalidArgument(format!("required field {field} is missing for node type {}", node.node_type)));
            }
        }
        for (field, expected) in &definition.data_types {
            if let Some(value) = Self::data_value(Some(&node.data), field) {
                if !Self::matches_data_type(&value, expected) {
                    return Err(PolypackError::InvalidArgument(format!("node field {field} must be {expected}")));
                }
            }
        }
        Ok(())
    }

    fn validate_edge_schema(&self, edge: &Edge, nodes: &HashMap<String, Node>, edges: &HashMap<String, Edge>) -> Result<()> {
        let Some(definition) = self.edge_type_definitions.get(&edge.edge_type) else { return Ok(()) };
        let source = nodes.get(&edge.source).ok_or_else(|| PolypackError::InvalidArgument(format!("edge {} source node is missing", edge.id)))?;
        let target = nodes.get(&edge.target).ok_or_else(|| PolypackError::InvalidArgument(format!("edge {} target node is missing", edge.id)))?;
        if !definition.source_types.is_empty() && !definition.source_types.iter().any(|kind| kind == &source.node_type) {
            return Err(PolypackError::InvalidArgument(format!("edge type {} does not permit source type {}", edge.edge_type, source.node_type)));
        }
        if !definition.target_types.is_empty() && !definition.target_types.iter().any(|kind| kind == &target.node_type) {
            return Err(PolypackError::InvalidArgument(format!("edge type {} does not permit target type {}", edge.edge_type, target.node_type)));
        }
        for field in &definition.required_fields {
            if Self::data_value(edge.data.as_ref(), field).is_none() {
                return Err(PolypackError::InvalidArgument(format!("required field {field} is missing for edge type {}", edge.edge_type)));
            }
        }
        for (field, expected) in &definition.data_types {
            if let Some(value) = Self::data_value(edge.data.as_ref(), field) {
                if !Self::matches_data_type(&value, expected) {
                    return Err(PolypackError::InvalidArgument(format!("edge field {field} must be {expected}")));
                }
            }
        }
        if let Some(cardinality) = &definition.cardinality {
            let outgoing = edges.values().filter(|candidate| candidate.id != edge.id && candidate.edge_type == edge.edge_type && candidate.source == edge.source).count();
            let incoming = edges.values().filter(|candidate| candidate.id != edge.id && candidate.edge_type == edge.edge_type && candidate.target == edge.target).count();
            let violates = match cardinality.as_str() {
                "one-to-one" => outgoing > 0 || incoming > 0,
                "one-to-many" => incoming > 0,
                "many-to-one" => outgoing > 0,
                "many-to-many" => false,
                _ => return Err(PolypackError::InvalidArgument(format!("unsupported cardinality {cardinality}"))),
            };
            if violates { return Err(PolypackError::InvalidArgument(format!("edge cardinality {cardinality} would be exceeded"))); }
        }
        Ok(())
    }

    fn validate_pending_schema(&self, changes: &ChangeBatch) -> Result<()> {
        // With no registered node/edge type, every per-node/per-edge check
        // below is a guaranteed no-op (`validate_node_schema`/
        // `validate_edge_schema` early-return when there's no matching
        // definition) — but reaching that no-op still costs a full clone of
        // *every* node and edge in the store on *every* apply call, which
        // dominates flush() cost regardless of batch size (see
        // benchmarks/database-core-edge-flush-report.md). Skip entirely in
        // the common no-schema case; this changes no observable behavior.
        if self.node_type_definitions.is_empty() && self.edge_type_definitions.is_empty() {
            return Ok(());
        }
        let mut nodes = self.nodes.clone();
        let mut edges = self.edges.clone();
        for id in &changes.delete_edge_ids { edges.remove(id); }
        for id in &changes.delete_node_ids { nodes.remove(id); }
        for node in &changes.put_nodes { nodes.insert(node.id.clone(), node.clone()); }
        for edge in &changes.put_edges { edges.insert(edge.id.clone(), edge.clone()); }
        for node in nodes.values() { self.validate_node_schema(node)?; }
        for edge in &changes.put_edges { self.validate_edge_schema(edge, &nodes, &edges)?; }
        Ok(())
    }

    fn write_schema_metadata(&mut self) -> Result<()> {
        let metadata = SchemaMetadata {
            node_types: self.node_type_definitions.values().cloned().collect(),
            edge_types: self.edge_type_definitions.values().cloned().collect(),
        };
        let data = serde_json::to_vec(&metadata).map_err(|error| PolypackError::CorruptData(error.to_string()))?;
        self.storage.write(SCHEMAS_FILE, &data)?;
        if self.config.durability == Durability::Fsync {
            self.storage.sync(SCHEMAS_FILE)?;
            self.storage.sync_dir()?;
        }
        Ok(())
    }

    fn load_schema_metadata(&mut self) -> Result<()> {
        let Some(data) = self.storage.read(SCHEMAS_FILE)? else { return Ok(()) };
        let metadata: SchemaMetadata = serde_json::from_slice(&data)
            .map_err(|error| PolypackError::CorruptData(format!("schema metadata: {error}")))?;
        let mut node_types = HashMap::new();
        for definition in metadata.node_types {
            Self::validate_node_type_definition(&definition)?;
            if node_types.insert(definition.node_type.clone(), definition).is_some() {
                return Err(PolypackError::CorruptData("schema metadata: duplicate node type".into()));
            }
        }
        let mut edge_types = HashMap::new();
        for definition in metadata.edge_types {
            Self::validate_edge_type_definition(&definition)?;
            if edge_types.insert(definition.edge_type.clone(), definition).is_some() {
                return Err(PolypackError::CorruptData("schema metadata: duplicate edge type".into()));
            }
        }
        self.node_type_definitions = node_types;
        self.edge_type_definitions = edge_types;
        Ok(())
    }

    fn validate_node_type_definition(definition: &NodeTypeDefinition) -> Result<()> {
        if definition.node_type.is_empty()
            || definition.required_fields.iter().any(String::is_empty)
            || definition.required_fields.iter().collect::<HashSet<_>>().len() != definition.required_fields.len()
            || definition.data_types.keys().any(|field| field.is_empty())
            || definition.data_types.values().any(|kind| !matches!(kind.as_str(), "string" | "number" | "integer" | "boolean" | "object" | "array"))
        {
            return Err(PolypackError::InvalidArgument(format!("invalid schema for node type {}", definition.node_type)));
        }
        Ok(())
    }

    fn validate_edge_type_definition(definition: &EdgeTypeDefinition) -> Result<()> {
        if definition.edge_type.is_empty()
            || definition.source_types.iter().any(String::is_empty)
            || definition.target_types.iter().any(String::is_empty)
            || definition.required_fields.iter().any(String::is_empty)
            || definition.required_fields.iter().collect::<HashSet<_>>().len() != definition.required_fields.len()
            || definition.data_types.keys().any(|field| field.is_empty())
            || definition.data_types.values().any(|kind| !matches!(kind.as_str(), "string" | "number" | "integer" | "boolean" | "object" | "array"))
            || definition.cardinality.as_deref().is_some_and(|cardinality| !matches!(cardinality, "one-to-one" | "one-to-many" | "many-to-one" | "many-to-many"))
        {
            return Err(PolypackError::InvalidArgument(format!("invalid schema for edge type {}", definition.edge_type)));
        }
        Ok(())
    }

    pub fn register_node_type(&mut self, definition: NodeTypeDefinition) -> Result<()> {
        self.ensure_loaded()?;
        Self::validate_node_type_definition(&definition)?;
        let name = definition.node_type.clone();
        let previous = self.node_type_definitions.insert(name.clone(), definition);
        if let Err(error) = self.nodes.values().try_for_each(|node| self.validate_node_schema(node)) {
            if let Some(previous) = previous.clone() { self.node_type_definitions.insert(previous.node_type.clone(), previous); }
            else { self.node_type_definitions.remove(&name); }
            return Err(error);
        }
        if let Err(error) = self.write_schema_metadata() {
            if let Some(previous) = previous { self.node_type_definitions.insert(previous.node_type.clone(), previous); }
            else { self.node_type_definitions.remove(&name); }
            return Err(error);
        }
        Ok(())
    }

    pub fn register_edge_type(&mut self, definition: EdgeTypeDefinition) -> Result<()> {
        self.ensure_loaded()?;
        Self::validate_edge_type_definition(&definition)?;
        let name = definition.edge_type.clone();
        let previous = self.edge_type_definitions.insert(name.clone(), definition);
        let validation = self.edges.values().try_for_each(|edge| self.validate_edge_schema(edge, &self.nodes, &self.edges));
        if let Err(error) = validation {
            if let Some(previous) = previous.clone() { self.edge_type_definitions.insert(previous.edge_type.clone(), previous); }
            else { self.edge_type_definitions.remove(&name); }
            return Err(error);
        }
        if let Err(error) = self.write_schema_metadata() {
            if let Some(previous) = previous { self.edge_type_definitions.insert(previous.edge_type.clone(), previous); }
            else { self.edge_type_definitions.remove(&name); }
            return Err(error);
        }
        Ok(())
    }

    fn secondary_key(node: &Node, definition: &SecondaryIndexDefinition) -> Option<String> {
        if definition.node_type.as_ref().is_some_and(|node_type| node_type != &node.node_type) {
            return None;
        }
        let values: Vec<serde_json::Value> = definition
            .fields
            .iter()
            .map(|field| node_field(node, field))
            .collect();
        if definition.sparse && values.iter().any(serde_json::Value::is_null) {
            return None;
        }
        serde_json::to_string(&values).ok()
    }

    fn validate_index_definitions(definitions: &HashMap<String, SecondaryIndexDefinition>) -> Result<()> {
        for (name, definition) in definitions {
            if name.is_empty() || definition.name != *name || definition.fields.is_empty() {
                return Err(PolypackError::InvalidArgument(format!("invalid secondary index {name}")));
            }
            if definition.fields.iter().any(String::is_empty)
                || definition.fields.iter().collect::<HashSet<_>>().len() != definition.fields.len()
            {
                return Err(PolypackError::InvalidArgument(format!("secondary index {name} fields must be unique and non-empty")));
            }
        }
        Ok(())
    }

    fn rebuild_secondary_indexes(&mut self) -> Result<()> {
        let mut rebuilt: HashMap<String, HashMap<String, HashSet<String>>> = self
            .index_definitions
            .keys()
            .map(|name| (name.clone(), HashMap::new()))
            .collect();
        for node in self.nodes.values() {
            for (name, definition) in &self.index_definitions {
                let Some(key) = Self::secondary_key(node, definition) else { continue };
                rebuilt.entry(name.clone()).or_default().entry(key).or_default().insert(node.id.clone());
            }
        }
        for (name, definition) in &self.index_definitions {
            if definition.unique {
                if let Some(buckets) = rebuilt.get(name) {
                    if let Some((key, _ids)) = buckets.iter().find(|(_, ids)| ids.len() > 1) {
                        return Err(PolypackError::InvalidArgument(format!("unique index {name} conflicts for key {key}")));
                    }
                }
            }
        }
        self.secondary_indexes = rebuilt;
        Ok(())
    }

    fn write_index_metadata(&mut self) -> Result<()> {
        let definitions: Vec<&SecondaryIndexDefinition> = self.index_definitions.values().collect();
        let data = serde_json::to_vec(&definitions).map_err(|error| PolypackError::CorruptData(error.to_string()))?;
        self.storage.write(INDEXES_FILE, &data)?;
        if self.config.durability == Durability::Fsync {
            self.storage.sync(INDEXES_FILE)?;
            self.storage.sync_dir()?;
        }
        Ok(())
    }

    fn load_index_metadata(&mut self) -> Result<()> {
        let Some(data) = self.storage.read(INDEXES_FILE)? else { return Ok(()) };
        let definitions: Vec<SecondaryIndexDefinition> = serde_json::from_slice(&data)
            .map_err(|error| PolypackError::CorruptData(format!("index metadata: {error}")))?;
        let mut parsed = HashMap::new();
        for definition in definitions {
            parsed.insert(definition.name.clone(), definition);
        }
        Self::validate_index_definitions(&parsed)?;
        self.index_definitions = parsed;
        Ok(())
    }

    /// Define or replace a persisted node-data index.
    pub fn define_index(&mut self, definition: SecondaryIndexDefinition) -> Result<()> {
        self.ensure_loaded()?;
        let name = definition.name.clone();
        let previous = self.index_definitions.insert(name.clone(), definition);
        if let Err(error) = Self::validate_index_definitions(&self.index_definitions).and_then(|_| self.rebuild_secondary_indexes()) {
            if let Some(previous) = previous { self.index_definitions.insert(name, previous); }
            else { self.index_definitions.remove(&name); }
            self.rebuild_secondary_indexes()?;
            return Err(error);
        }
        self.write_index_metadata()
    }

    /// Drop a persisted node-data index.
    pub fn drop_index(&mut self, name: &str) -> Result<bool> {
        self.ensure_loaded()?;
        let Some(previous) = self.index_definitions.remove(name) else { return Ok(false) };
        self.rebuild_secondary_indexes()?;
        if let Err(error) = self.write_index_metadata() {
            self.index_definitions.insert(name.to_string(), previous);
            self.rebuild_secondary_indexes()?;
            return Err(error);
        }
        Ok(true)
    }

    pub fn index_definitions(&mut self) -> Result<Vec<SecondaryIndexDefinition>> {
        self.ensure_loaded()?;
        Ok(self.index_definitions.values().cloned().collect())
    }

    fn indexed_ids(&self, query: &NodeQuery) -> Option<HashSet<String>> {
        let attributes = query.attributes.as_ref()?;
        self.index_definitions.iter().find_map(|(name, definition)| {
            if definition.fields.iter().all(|field| attributes.contains_key(field)) {
                let values: Vec<serde_json::Value> = definition.fields.iter().map(|field| attributes.get(field).cloned().unwrap_or(serde_json::Value::Null)).collect();
                let key = serde_json::to_string(&values).ok()?;
                self.secondary_indexes.get(name)?.get(&key).cloned()
            } else { None }
        })
    }

    fn validate_pending_indexes(&self, changes: &ChangeBatch) -> Result<()> {
        if self.index_definitions.values().all(|definition| !definition.unique) {
            return Ok(());
        }
        let mut nodes = self.nodes.clone();
        for id in &changes.delete_node_ids { nodes.remove(id); }
        for node in &changes.put_nodes { nodes.insert(node.id.clone(), node.clone()); }
        for (name, definition) in &self.index_definitions {
            if !definition.unique { continue; }
            let mut seen: HashMap<String, String> = HashMap::new();
            for node in nodes.values() {
                let Some(key) = Self::secondary_key(node, definition) else { continue };
                if let Some(previous) = seen.insert(key, node.id.clone()) {
                    if previous != node.id {
                        return Err(PolypackError::InvalidArgument(format!("unique index {name} conflicts with node {previous}")));
                    }
                }
            }
        }
        Ok(())
    }

    fn index_node(&mut self, node: &Node) {
        if let Some(existing) = self.nodes.get(&node.id) {
            if existing.node_type != node.node_type {
                self.unindex_node(&node.id);
            }
        }
        self.by_type
            .entry(node.node_type.clone())
            .or_default()
            .insert(node.id.clone());
        for (name, definition) in &self.index_definitions {
            if let Some(key) = Self::secondary_key(node, definition) {
                self.secondary_indexes.entry(name.clone()).or_default().entry(key).or_default().insert(node.id.clone());
            }
        }
    }

    fn unindex_node(&mut self, id: &str) {
        let type_key = self.nodes.get(id).map(|n| n.node_type.clone());
        if let Some(node) = self.nodes.get(id) {
            for (name, definition) in &self.index_definitions {
                if let Some(key) = Self::secondary_key(node, definition) {
                    if let Some(bucket) = self.secondary_indexes.get_mut(name).and_then(|indexes| indexes.get_mut(&key)) {
                        bucket.remove(id);
                        if bucket.is_empty() { self.secondary_indexes.get_mut(name).unwrap().remove(&key); }
                    }
                }
            }
        }
        if let Some(t) = type_key {
            if let Some(ids) = self.by_type.get_mut(&t) {
                ids.remove(id);
                if ids.is_empty() {
                    self.by_type.remove(&t);
                }
            }
        }
    }

    fn index_edge(&mut self, edge: &Edge) {
        self.edges_by_source
            .entry(edge.source.clone())
            .or_default()
            .insert(edge.id.clone());
        self.edges_by_target
            .entry(edge.target.clone())
            .or_default()
            .insert(edge.id.clone());
    }

    fn unindex_edge(&mut self, id: &str) {
        let edge = self.edges.get(id);
        let Some(edge) = edge else { return };
        if let Some(ids) = self.edges_by_source.get_mut(&edge.source) {
            ids.remove(id);
            if ids.is_empty() {
                self.edges_by_source.remove(&edge.source);
            }
        }
        if let Some(ids) = self.edges_by_target.get_mut(&edge.target) {
            ids.remove(id);
            if ids.is_empty() {
                self.edges_by_target.remove(&edge.target);
            }
        }
    }

    fn rebuild_indexes(&mut self) {
        self.by_type.clear();
        self.secondary_indexes = self.index_definitions.keys().map(|name| (name.clone(), HashMap::new())).collect();
        self.edges_by_source.clear();
        self.edges_by_target.clear();
        let nodes: Vec<Node> = self.nodes.values().cloned().collect();
        for node in &nodes {
            self.index_node(node);
        }
        let edges: Vec<Edge> = self.edges.values().cloned().collect();
        for edge in &edges {
            self.index_edge(edge);
        }
    }

    /// Effective WAL compaction threshold, grown with the store to keep total
    /// compaction work linear in writes instead of quadratic.
    fn effective_compact_threshold(&self) -> usize {
        let records = self.nodes.len() + self.edges.len() + self.vectors.len();
        self.config.compact_threshold.max(records / COMPACT_RATIO)
    }

    /// Candidate node ids for a query when only the type index is needed.
    fn type_only_ids(&self, query: &NodeQuery) -> Option<HashSet<String>> {
        let types = query.node_types.as_ref()?;
        if query.attributes.is_some() || query.attribute_ranges.is_some() {
            return None;
        }
        let mut ids = HashSet::new();
        for t in types {
            if let Some(set) = self.by_type.get(t) {
                ids.extend(set.iter().cloned());
            }
        }
        Some(ids)
    }

    pub fn assert_open(&self) -> Result<()> {
        if self.closed {
            return Err(PolypackError::Closed);
        }
        Ok(())
    }

    fn do_load(&mut self) -> Result<()> {
        self.load_index_metadata()?;
        self.load_schema_metadata()?;
        if let Some(data) = self.storage.read(SNAPSHOT_FILE)? {
            let snapshot = decode_snapshot(&data)?;
            self.nodes = snapshot.nodes.into_iter().collect();
            self.edges = snapshot.edges.into_iter().collect();
            self.vectors = snapshot.vectors.into_iter().collect();
        }
        if let Some(wal_data) = self.storage.read(WAL_FILE)? {
            if !wal_data.is_empty() {
                for entry in decode_wal(&wal_data) {
                    self.replay(entry);
                }
                // Persist the snapshot BEFORE deleting the WAL so a crash
                // between the two only re-replays an idempotent WAL.
                self.write_snapshot()?;
                self.storage.delete(WAL_FILE)?;
                self.wal_entry_count = 0;
            }
        }
        self.rebuild_indexes();
        Ok(())
    }

    fn ensure_loaded(&mut self) -> Result<()> {
        self.assert_open()?;
        if !self.loaded {
            self.do_load()?;
            let log = self.mutation_log()?;
            self.mutation_sequence = log.last().map(|record| record.sequence).unwrap_or(0);
            self.seen_operation_ids = log.iter().map(|record| record.operation_id.clone()).collect();
            self.seen_transaction_ids = log.iter().map(|record| record.transaction_id.clone()).collect();
            self.loaded = true;
        }
        Ok(())
    }

    fn replay(&mut self, entry: WalEntry) {
        match entry {
            WalEntry::PutNode(node) => {
                self.index_node(&node);
                self.nodes.insert(node.id.clone(), node.clone());
            }
            WalEntry::DeleteNode(id) => {
                self.unindex_node(&id);
                self.nodes.remove(&id);
            }
            WalEntry::PutEdge(edge) => {
                self.edges.insert(edge.id.clone(), edge.clone());
                self.index_edge(&edge);
            }
            WalEntry::DeleteEdge(id) => {
                self.unindex_edge(&id);
                self.edges.remove(&id);
            }
            WalEntry::PutVector { id, vector } => {
                self.vectors.insert(id, vector);
            }
            WalEntry::DeleteVector(id) => {
                self.vectors.remove(&id);
            }
            WalEntry::ClearAll => {
                self.nodes.clear();
                self.edges.clear();
                self.vectors.clear();
                self.by_type.clear();
                self.edges_by_source.clear();
                self.edges_by_target.clear();
            }
        }
    }

    fn write_snapshot(&mut self) -> Result<()> {
        let nodes: Vec<(String, Node)> = self.nodes.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let edges: Vec<(String, Edge)> = self.edges.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let vectors: Vec<(String, Vec<f64>)> = self.vectors.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        self.storage.write(SNAPSHOT_FILE, &encode_snapshot(&nodes, &edges, &vectors))?;
        if self.config.durability == Durability::Fsync {
            self.storage.sync(SNAPSHOT_FILE)?;
            self.storage.sync_dir()?;
        }
        Ok(())
    }

    /// Compact the WAL into a snapshot. The generation captured at entry is
    /// the number of WAL entries reflected in the snapshot; entries that
    /// arrived mid-write (defensively) are retained in the WAL.
    pub fn compact(&mut self) -> Result<()> {
        self.ensure_loaded()?;
        let generation = self.wal_entry_count;
        if generation == 0 {
            return Ok(());
        }
        self.write_snapshot()?;
        let mut remaining = 0;
        if let Some(wal_data) = self.storage.read(WAL_FILE)? {
            if !wal_data.is_empty() {
                let entries = decode_wal(&wal_data);
                remaining = entries.len().saturating_sub(generation);
                if remaining > 0 {
                    self.storage.write(WAL_FILE, &encode_wal(&entries[generation..]))?;
                } else {
                    self.storage.write(WAL_FILE, &[])?;
                }
            }
        }
        self.wal_entry_count = remaining;
        self.trim_mutation_log()?;
        Ok(())
    }

    /// Apply `config.mutation_log_retention` to `mutations.jsonl`, dropping
    /// records that fall outside the configured window. A no-op when no
    /// retention policy is configured, or when nothing would be dropped.
    /// Called automatically by `compact()`; also public so a host can trim
    /// on its own schedule independent of WAL-driven compaction.
    pub fn trim_mutation_log(&mut self) -> Result<()> {
        let Some(retention) = self.config.mutation_log_retention else { return Ok(()) };
        if retention.max_entries.is_none() && retention.max_age_ms.is_none() {
            return Ok(());
        }
        let records = self.mutation_log()?;
        let len = records.len();
        if len == 0 {
            return Ok(());
        }
        let now = now_millis();
        let retained: Vec<&MutationRecord> = records
            .iter()
            .enumerate()
            .filter(|(index, record)| {
                let within_count = retention.max_entries.is_none_or(|max| *index >= len.saturating_sub(max));
                let within_age = retention.max_age_ms.is_none_or(|max_age| now.saturating_sub(record.timestamp) <= max_age);
                within_count && within_age
            })
            .map(|(_, record)| record)
            .collect();
        if retained.len() == len {
            return Ok(());
        }
        let mut encoded = Vec::new();
        for record in retained {
            encoded.extend_from_slice(&serde_json::to_vec(record).map_err(|error| PolypackError::CorruptData(format!("mutation log: {error}")))?);
            encoded.push(b'\n');
        }
        self.storage.write(MUTATION_LOG_FILE, &encoded)?;
        if self.config.durability == Durability::Fsync {
            self.storage.sync(MUTATION_LOG_FILE)?;
        }
        Ok(())
    }

    /// Apply a change batch: deletions first, then insertions, appended to the
    /// WAL in that order.
    ///
    /// The WAL append (and, under `Fsync` durability, the fsync) must
    /// succeed before in-memory state is mutated. Otherwise a failed append
    /// would leave memory ahead of disk: the caller sees success reflected
    /// nowhere on disk, and a crash right after would silently lose it.
    pub fn apply(&mut self, changes: &ChangeBatch) -> Result<()> {
        self.apply_with_transaction(changes, None)
    }

    pub fn apply_with_transaction(&mut self, changes: &ChangeBatch, transaction_id: Option<&str>) -> Result<()> {
        self.apply_with_metadata(changes, transaction_id, None, None, None)
    }

    pub fn apply_with_metadata(
        &mut self,
        changes: &ChangeBatch,
        transaction_id: Option<&str>,
        actor: Option<&str>,
        base_revision: Option<u64>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        self.apply_with_identity(changes, None, transaction_id, actor, base_revision, metadata)
    }

    pub fn apply_with_identity(
        &mut self,
        changes: &ChangeBatch,
        operation_id: Option<&str>,
        transaction_id: Option<&str>,
        actor: Option<&str>,
        base_revision: Option<u64>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        self.ensure_loaded()?;
        // Check idempotency before schema/index validation: a retry of an
        // already-committed operation_id/transaction_id must return Ok even if
        // the schema has since changed in a way that would reject the same
        // payload today. Validating first would break the idempotent-retry
        // guarantee this dedup check exists to provide.
        if operation_id.is_some_and(|id| self.seen_operation_ids.contains(id))
            || transaction_id.is_some_and(|id| self.seen_transaction_ids.contains(id))
        {
            return Ok(());
        }
        validate_batch(changes)?;
        self.validate_pending_schema(changes)?;
        self.validate_pending_indexes(changes)?;
        let mut entries: Vec<WalEntry> = Vec::new();
        for id in &changes.delete_node_ids {
            entries.push(WalEntry::DeleteNode(id.clone()));
        }
        for id in &changes.delete_edge_ids {
            entries.push(WalEntry::DeleteEdge(id.clone()));
        }
        for id in &changes.delete_vector_ids {
            entries.push(WalEntry::DeleteVector(id.clone()));
        }
        for node in &changes.put_nodes {
            entries.push(WalEntry::PutNode(node.clone()));
        }
        for edge in &changes.put_edges {
            entries.push(WalEntry::PutEdge(edge.clone()));
        }
        for v in &changes.put_vectors {
            entries.push(WalEntry::PutVector { id: v.id.clone(), vector: v.vector.clone() });
        }
        if entries.is_empty() {
            return Ok(());
        }
        let encoded = encode_wal(&entries);
        self.storage.append(WAL_FILE, &encoded)?;
        if self.config.durability == Durability::Fsync {
            self.storage.sync(WAL_FILE)?;
        }
        let sequence = self.mutation_sequence.saturating_add(1);
        let generated_operation_id = format!("mutation-{sequence}");
        let record = MutationRecord {
            operation_id: operation_id.unwrap_or(&generated_operation_id).to_string(),
            sequence,
            transaction_id: transaction_id.unwrap_or(&generated_operation_id).to_string(),
            timestamp: now_millis(),
            operations: mutation_operations(changes),
            actor: actor.map(str::to_string),
            base_revision,
            metadata,
        };
        let mut encoded_record = serde_json::to_vec(&record).map_err(|error| PolypackError::CorruptData(error.to_string()))?;
        encoded_record.push(b'\n');
        self.storage.append(MUTATION_LOG_FILE, &encoded_record)?;
        if self.config.durability == Durability::Fsync {
            self.storage.sync(MUTATION_LOG_FILE)?;
        }
        self.mutation_sequence = sequence;
        self.seen_operation_ids.insert(record.operation_id.clone());
        self.seen_transaction_ids.insert(record.transaction_id.clone());
        // WAL entries are durable (or at least accepted by the host); it is
        // now safe to mutate in-memory state to match.
        for id in &changes.delete_node_ids {
            self.unindex_node(id);
            self.nodes.remove(id);
        }
        for id in &changes.delete_edge_ids {
            self.unindex_edge(id);
            self.edges.remove(id);
        }
        for id in &changes.delete_vector_ids {
            self.vectors.remove(id);
        }
        for node in &changes.put_nodes {
            self.index_node(node);
            self.nodes.insert(node.id.clone(), node.clone());
        }
        for edge in &changes.put_edges {
            self.edges.insert(edge.id.clone(), edge.clone());
            self.index_edge(edge);
        }
        for v in &changes.put_vectors {
            self.vectors.insert(v.id.clone(), v.vector.clone());
        }
        self.wal_entry_count += entries.len();
        if self.wal_entry_count >= self.effective_compact_threshold() {
            self.compact()?;
        }
        Ok(())
    }

    /// Delete all nodes, edges, and vectors, writing an empty snapshot and truncating the WAL.
    pub fn clear_all(&mut self) -> Result<()> {
        self.ensure_loaded()?;
        self.nodes.clear();
        self.edges.clear();
        self.vectors.clear();
        self.by_type.clear();
        self.secondary_indexes.clear();
        self.edges_by_source.clear();
        self.edges_by_target.clear();
        self.wal_entry_count = 0;
        self.write_snapshot()?;
        self.storage.write(WAL_FILE, &[])?;
        Ok(())
    }

    /// Compact and mark the store closed. Idempotent — safe to call repeatedly.
    pub fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        if self.loaded {
            self.compact()?;
        }
        self.closed = true;
        Ok(())
    }

    // ── reads ──

    pub fn node_ids(&mut self) -> Result<Vec<String>> {
        self.ensure_loaded()?;
        Ok(self.nodes.keys().cloned().collect())
    }

    /// Number of nodes in the store, without materialising ids.
    pub fn node_count(&mut self) -> Result<usize> {
        self.ensure_loaded()?;
        Ok(self.nodes.len())
    }

    /// Query nodes against the store's secondary type index and full-scan
    /// fallback, mirroring the TypeScript adapter's `queryNodes`.
    pub fn query_nodes(&mut self, query: &NodeQuery) -> Result<Vec<Node>> {
        self.ensure_loaded()?;
        let candidate_ids: Vec<String> = match &query.candidate_ids {
            Some(ids) => ids.clone(),
            None => match self.indexed_ids(query).or_else(|| self.type_only_ids(query)) {
            Some(ids) => ids.into_iter().collect(),
            None => self.nodes.keys().cloned().collect(),
            },
        };
        if let Some(limit) = query.max_nodes_visited {
            if candidate_ids.len() > limit {
                return Err(PolypackError::ResourceLimit { name: "maxNodesVisited".into(), limit });
            }
        }
        let mut results: Vec<Node> = candidate_ids
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .filter(|n| matches_node(n, query))
            .cloned()
            .collect();
        if let Some(limit) = query.max_result_size {
            if results.len() > limit {
                return Err(PolypackError::ResourceLimit { name: "maxResultSize".into(), limit });
            }
        }
        if let Some(order) = &query.order_by {
            results.sort_by(|a, b| {
                let av = numeric_field(a, &order.field);
                let bv = numeric_field(b, &order.field);
                let ord = av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal);
                if order.direction == "desc" {
                    ord.reverse()
                } else {
                    ord
                }
            });
        }
        if let Some(offset) = query.offset {
            results = results.into_iter().skip(offset).collect();
        }
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    /// Count nodes matching `query`, with type-index and empty-query fast
    /// paths mirroring the TypeScript adapter's `countNodes`.
    pub fn count_nodes(&mut self, query: &NodeQuery) -> Result<usize> {
        self.ensure_loaded()?;
        let candidate_ids: Option<Vec<String>> = match &query.candidate_ids {
            Some(ids) => Some(ids.clone()),
            None => self.indexed_ids(query).or_else(|| self.type_only_ids(query)).map(|ids| ids.into_iter().collect()),
        };
        let candidate_count = candidate_ids.as_ref().map_or(self.nodes.len(), Vec::len);
        if let Some(limit) = query.max_nodes_visited {
            if candidate_count > limit {
                return Err(PolypackError::ResourceLimit { name: "maxNodesVisited".into(), limit });
            }
        }
        let count = match &candidate_ids {
            Some(ids) => ids.iter().filter_map(|id| self.nodes.get(id)).filter(|n| matches_node(n, query)).count(),
            None => self.nodes.values().filter(|n| matches_node(n, query)).count(),
        };
        if let Some(limit) = query.max_result_size {
            if count > limit {
                return Err(PolypackError::ResourceLimit { name: "maxResultSize".into(), limit });
            }
        }
        let after_offset = match query.offset {
            None => count,
            Some(offset) => count.saturating_sub(offset),
        };
        Ok(match query.limit {
            None => after_offset,
            Some(limit) => after_offset.min(limit),
        })
    }

    /// Edges from the given sources, using the source index.
    pub fn get_edges_by_sources(&mut self, sources: &[String], edge_type: Option<&str>) -> Result<Vec<Edge>> {
        self.ensure_loaded()?;
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for source in sources {
            if let Some(ids) = self.edges_by_source.get(source) {
                for id in ids {
                    if let Some(edge) = self.edges.get(id) {
                        if edge_type.is_none_or(|t| t == edge.edge_type) && seen.insert(edge.id.clone()) {
                            out.push(edge.clone());
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Edges targeting the given nodes, using the target index.
    pub fn get_edges_by_targets(&mut self, targets: &[String], edge_type: Option<&str>) -> Result<Vec<Edge>> {
        self.ensure_loaded()?;
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for target in targets {
            if let Some(ids) = self.edges_by_target.get(target) {
                for id in ids {
                    if let Some(edge) = self.edges.get(id) {
                        if edge_type.is_none_or(|t| t == edge.edge_type) && seen.insert(edge.id.clone()) {
                            out.push(edge.clone());
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn get_node(&mut self, id: &str) -> Result<Option<Node>> {
        self.ensure_loaded()?;
        Ok(self.nodes.get(id).cloned())
    }

    pub fn get_vector(&mut self, id: &str) -> Result<Option<Vec<f64>>> {
        self.ensure_loaded()?;
        Ok(self.vectors.get(id).cloned())
    }

    pub fn vectors_snapshot(&mut self) -> Result<Vec<(String, Vec<f64>)>> {
        self.ensure_loaded()?;
        Ok(self.vectors.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }

    pub fn edges_snapshot(&mut self) -> Result<Vec<(String, Edge)>> {
        self.ensure_loaded()?;
        Ok(self.edges.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn node(id: &str) -> Node {
        Node {
            id: id.into(),
            node_type: "doc".into(),
            data: serde_json::Map::new(),
            vector: None,
            inserted_at: 1,
            updated_at: 1,
            revision: 0,
            activation: None,
            ..Default::default()
        }
    }

    fn shared() -> Arc<Mutex<InMemoryStorage>> {
        Arc::new(Mutex::new(InMemoryStorage::new()))
    }

    fn batch(nodes: &[Node]) -> ChangeBatch {
        ChangeBatch {
            put_nodes: nodes.to_vec(),
            ..Default::default()
        }
    }

    #[test]
    fn persists_and_loads_across_stores() {
        let storage = shared();
        {
            let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
            s.apply(&batch(&[node("a"), node("b")])).unwrap();
            s.close().unwrap();
        }
        let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
        let mut ids = s.node_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
        s.close().unwrap();
    }

    #[test]
    fn recovers_when_only_wal_exists() {
        let storage = shared();
        {
            let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
            s.apply(&batch(&[node("w1"), node("w2")])).unwrap();
            drop(s);
            // Simulate a crash before compaction: no snapshot, WAL only.
            storage.lock().unwrap().delete(SNAPSHOT_FILE).unwrap();
        }
        let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
        let mut ids = s.node_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["w1".to_string(), "w2".to_string()]);
        s.close().unwrap();
        assert!(storage.lock().unwrap().exists(SNAPSHOT_FILE).unwrap());
    }

    #[test]
    fn recovers_from_truncated_wal_tail() {
        let storage = shared();
        let config = StoreConfig {
            compact_threshold: 100,
            ..Default::default()
        };
        {
            let mut s = Store::new(Box::new(storage.clone()), config);
            s.apply(&batch(&[node("a"), node("b"), node("c")])).unwrap();
            // Crash before close/compaction: the WAL holds all three frames.
        }
        let wal = storage.lock().unwrap().read(WAL_FILE).unwrap().unwrap();
        assert!(!wal.is_empty());
        // Cut into the third frame; the in-flight frame is lost, the two
        // acknowledged frames before it must survive.
        let truncated = &wal[..wal.len() - 3];
        storage.lock().unwrap().write(WAL_FILE, truncated).unwrap();
        let mut s2 = Store::new(Box::new(storage.clone()), StoreConfig::default());
        let mut ids = s2.node_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
        s2.close().unwrap();
    }

    #[test]
    fn replay_delete_is_idempotent() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
        s.apply(&batch(&[node("a")])).unwrap();
        s.apply(&ChangeBatch {
            delete_node_ids: vec!["a".into()],
            ..Default::default()
        })
        .unwrap();
        s.close().unwrap();
        let mut s2 = Store::new(Box::new(storage.clone()), StoreConfig::default());
        assert!(s2.node_ids().unwrap().is_empty());
    }

    #[test]
    fn compacts_at_threshold() {
        let storage = shared();
        let config = StoreConfig {
            compact_threshold: 2,
            ..Default::default()
        };
        let mut s = Store::new(Box::new(storage.clone()), config);
        s.apply(&batch(&[node("n1"), node("n2")])).unwrap();
        assert!(storage.lock().unwrap().exists(SNAPSHOT_FILE).unwrap());
        let wal = storage.lock().unwrap().read(WAL_FILE).unwrap().unwrap();
        assert!(wal.is_empty());
        s.close().unwrap();
    }

    #[test]
    fn mutation_log_retention_none_keeps_full_history() {
        let storage = shared();
        let config = StoreConfig { compact_threshold: 1, ..Default::default() };
        let mut s = Store::new(Box::new(storage), config);
        for id in ["a", "b", "c"] {
            s.apply(&batch(&[node(id)])).unwrap();
        }
        assert_eq!(s.mutation_log().unwrap().len(), 3);
        s.close().unwrap();
    }

    #[test]
    fn mutation_log_retention_by_count_trims_oldest_on_compact() {
        let storage = shared();
        let config = StoreConfig {
            compact_threshold: 1,
            mutation_log_retention: Some(MutationLogRetention { max_entries: Some(2), max_age_ms: None }),
            ..Default::default()
        };
        let mut s = Store::new(Box::new(storage), config);
        for id in ["a", "b", "c", "d"] {
            // compact_threshold: 1 means every apply() triggers compact(),
            // which in turn trims the mutation log.
            s.apply(&batch(&[node(id)])).unwrap();
        }
        let log = s.mutation_log().unwrap();
        assert_eq!(log.len(), 2);
        let ids: Vec<&str> = log
            .iter()
            .flat_map(|record| record.operations.iter())
            .map(|op| op.payload["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["c", "d"]);
        // Trimming the durable log must not affect node data.
        let mut ids = s.node_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()]);
        s.close().unwrap();
    }

    #[test]
    fn mutation_log_retention_by_age_trims_old_records_on_compact() {
        let storage = shared();
        let config = StoreConfig {
            compact_threshold: 1,
            mutation_log_retention: Some(MutationLogRetention { max_entries: None, max_age_ms: Some(50) }),
            ..Default::default()
        };
        let mut s = Store::new(Box::new(storage), config);
        s.apply(&batch(&[node("old")])).unwrap();
        // The "old" record is now well past the 50ms window by the time the
        // second apply()'s compact() runs, while "new" (just written) is
        // comfortably inside it — avoids racing on a zero-width window.
        std::thread::sleep(std::time::Duration::from_millis(100));
        s.apply(&batch(&[node("new")])).unwrap();
        let log = s.mutation_log().unwrap();
        let ids: Vec<&str> =
            log.iter().flat_map(|record| record.operations.iter()).map(|op| op.payload["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["new"]);
        s.close().unwrap();
    }

    #[test]
    fn trim_mutation_log_is_a_noop_without_retention_config() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage), StoreConfig::default());
        s.apply(&batch(&[node("a")])).unwrap();
        s.trim_mutation_log().unwrap();
        assert_eq!(s.mutation_log().unwrap().len(), 1);
    }

    #[test]
    fn rejects_use_after_close() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
        s.apply(&batch(&[node("a")])).unwrap();
        s.close().unwrap();
        assert!(matches!(s.node_ids(), Err(PolypackError::Closed)));
        assert!(matches!(s.apply(&batch(&[node("late")])), Err(PolypackError::Closed)));
    }

    #[test]
    fn close_is_idempotent() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
        s.apply(&batch(&[node("a")])).unwrap();
        s.close().unwrap();
        s.close().unwrap();
    }

    #[test]
    fn clear_all_empties_store() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
        s.apply(&batch(&[node("a"), node("b")])).unwrap();
        s.clear_all().unwrap();
        assert!(s.node_ids().unwrap().is_empty());
        s.close().unwrap();
        let mut s2 = Store::new(Box::new(storage.clone()), StoreConfig::default());
        assert!(s2.node_ids().unwrap().is_empty());
    }

    fn book(id: &str, genre: &str, price: f64) -> Node {
        Node {
            id: id.into(),
            node_type: "book".into(),
            data: serde_json::json!({ "genre": genre, "price": price })
                .as_object()
                .cloned()
                .unwrap(),
            vector: None,
            inserted_at: 1,
            updated_at: 1,
            revision: 0,
            activation: None,
            ..Default::default()
        }
    }

    #[test]
    fn counts_and_queries_nodes_via_type_index() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
        s.apply(&batch(&[
            book("a", "sci-fi", 20.0),
            book("b", "sci-fi", 10.0),
            book("c", "fantasy", 15.0),
        ]))
        .unwrap();
        s.apply(&batch(&[node("u")])).unwrap();
        // u is type "doc" (node() helper); add a genuine user type via put
        s.apply(&ChangeBatch {
            put_nodes: vec![Node {
                id: "u2".into(),
                node_type: "user".into(),
                data: Default::default(),
                vector: None,
                inserted_at: 1,
                updated_at: 1,
                revision: 0,
                activation: None, ..Default::default()
}],
            ..Default::default()
        })
        .unwrap();

        assert_eq!(s.node_count().unwrap(), 5);
        assert_eq!(s.count_nodes(&NodeQuery::default()).unwrap(), 5);
        assert_eq!(
            s.count_nodes(&NodeQuery { node_types: Some(vec!["book".into()]), ..Default::default() })
                .unwrap(),
            3
        );

        // attribute + range + order + pagination
        let q = NodeQuery {
            node_types: Some(vec!["book".into()]),
            attributes: Some(serde_json::json!({ "genre": "sci-fi" }).as_object().cloned().unwrap()),
            attribute_ranges: Some(HashMap::from([(
                "price".into(),
                RangeQuery { above: Some(5.0), below: Some(25.0) },
            )])),
            order_by: Some(OrderBy { field: "price".into(), direction: "asc".into() }),
            ..Default::default()
        };
        let nodes = s.query_nodes(&q).unwrap();
        assert_eq!(nodes.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(), vec!["b", "a"]);
        assert_eq!(s.count_nodes(&q).unwrap(), 2);

        // offset/limit
        let page = s
            .query_nodes(&NodeQuery {
                node_types: Some(vec!["book".into()]),
                offset: Some(1),
                limit: Some(1),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(s.count_nodes(&NodeQuery { node_types: Some(vec!["book".into()]), limit: Some(1), ..Default::default() }).unwrap(), 1);

        // type index survives delete/re-put
        s.apply(&ChangeBatch { delete_node_ids: vec!["a".into()], ..Default::default() }).unwrap();
        assert_eq!(s.count_nodes(&NodeQuery { node_types: Some(vec!["book".into()]), ..Default::default() }).unwrap(), 2);
        s.apply(&batch(&[book("a", "sci-fi", 20.0)])).unwrap();
        assert_eq!(s.count_nodes(&NodeQuery { node_types: Some(vec!["book".into()]), ..Default::default() }).unwrap(), 3);

        // indexes are rebuilt after reopen from the persisted store
        s.close().unwrap();
        let mut s2 = Store::new(Box::new(storage.clone()), StoreConfig::default());
        assert_eq!(s2.count_nodes(&NodeQuery { node_types: Some(vec!["book".into()]), ..Default::default() }).unwrap(), 3);
        assert_eq!(s2.count_nodes(&NodeQuery { node_types: Some(vec!["user".into()]), ..Default::default() }).unwrap(), 1);
    }

    #[test]
    fn query_resource_limits_reject_expensive_results() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage), StoreConfig::default());
        s.apply(&batch(&[node("a"), node("b")])).unwrap();
        let visited = NodeQuery { max_nodes_visited: Some(1), ..Default::default() };
        assert!(matches!(s.query_nodes(&visited), Err(PolypackError::ResourceLimit { name, limit }) if name == "maxNodesVisited" && limit == 1));
        let results = NodeQuery { max_result_size: Some(1), ..Default::default() };
        assert!(matches!(s.query_nodes(&results), Err(PolypackError::ResourceLimit { name, limit }) if name == "maxResultSize" && limit == 1));
        assert!(matches!(s.count_nodes(&results), Err(PolypackError::ResourceLimit { name, .. }) if name == "maxResultSize"));
    }

    #[test]
    fn configurable_indexes_filter_and_enforce_uniqueness() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
        s.define_index(SecondaryIndexDefinition {
            name: "book-isbn".into(),
            fields: vec!["isbn".into()],
            node_type: Some("book".into()),
            unique: true,
            sparse: false,
        }).unwrap();
        s.apply(&ChangeBatch { put_nodes: vec![
            Node { data: serde_json::json!({"isbn": "111"}).as_object().unwrap().clone(), ..book("a", "fiction", 1.0)
},
            Node { data: serde_json::json!({"isbn": "222"}).as_object().unwrap().clone(), ..book("b", "history", 2.0)
},
        ], ..Default::default() }).unwrap();
        let query = NodeQuery {
            node_types: Some(vec!["book".into()]),
            attributes: Some(serde_json::json!({"isbn": "222"}).as_object().unwrap().clone()),
            ..Default::default()
        };
        assert_eq!(s.query_nodes(&query).unwrap().iter().map(|node| node.id.as_str()).collect::<Vec<_>>(), vec!["b"]);
        let duplicate = ChangeBatch { put_nodes: vec![Node { data: serde_json::json!({"isbn": "111"}).as_object().unwrap().clone(), ..book("c", "science", 3.0)
}], ..Default::default() };
        assert!(matches!(s.apply(&duplicate), Err(PolypackError::InvalidArgument(message)) if message.contains("unique index")));
        assert_eq!(s.node_count().unwrap(), 2);

        s.close().unwrap();
        let mut reopened = Store::new(Box::new(storage), StoreConfig::default());
        assert_eq!(reopened.index_definitions().unwrap().len(), 1);
        assert_eq!(reopened.query_nodes(&query).unwrap().len(), 1);
    }

    #[test]
    fn schema_hooks_validate_nodes_edges_before_wal_write() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
        s.register_node_type(NodeTypeDefinition {
            node_type: "person".into(),
            required_fields: vec!["name".into()],
            data_types: HashMap::from([(String::from("age"), String::from("integer"))]),
            memory_class: None,
        }).unwrap();
        s.register_edge_type(EdgeTypeDefinition {
            edge_type: "PARENT_OF".into(),
            source_types: vec!["person".into()],
            target_types: vec!["person".into()],
            required_fields: Vec::new(),
            data_types: HashMap::new(),
            cardinality: Some("many-to-many".into()),
        }).unwrap();
        let invalid = ChangeBatch { put_nodes: vec![Node { id: "p1".into(), node_type: "person".into(), data: serde_json::json!({"age": 1}).as_object().unwrap().clone(), ..book("p1", "person", 0.0)
}], ..Default::default() };
        assert!(matches!(s.apply(&invalid), Err(PolypackError::InvalidArgument(message)) if message.contains("required field name")));
        assert!(storage.lock().unwrap().read(WAL_FILE).unwrap().is_none());
        let valid = ChangeBatch { put_nodes: vec![Node { id: "p1".into(), node_type: "person".into(), data: serde_json::json!({"name": "A", "age": 1}).as_object().unwrap().clone(), ..book("p1", "person", 0.0)
}], ..Default::default() };
        s.apply(&valid).unwrap();
        let invalid_edge = ChangeBatch { put_edges: vec![Edge { id: "e1".into(), source: "p1".into(), target: "missing".into(), edge_type: "PARENT_OF".into(), data: None, created_at: 1, revision: 0 }], ..Default::default() };
        assert!(matches!(s.apply(&invalid_edge), Err(PolypackError::InvalidArgument(message)) if message.contains("target node is missing")));
        s.close().unwrap();
        let mut reopened = Store::new(Box::new(storage), StoreConfig::default());
        let invalid_after_reload = ChangeBatch { put_nodes: vec![Node { id: "p2".into(), node_type: "person".into(), data: serde_json::json!({"name": "B", "age": "not-an-integer"}).as_object().unwrap().clone(), ..book("p2", "person", 0.0)
}], ..Default::default() };
        assert!(matches!(reopened.apply(&invalid_after_reload), Err(PolypackError::InvalidArgument(message)) if message.contains("must be integer")));
    }

    #[test]
    fn node_type_memory_class_round_trips_through_schema_persistence() {
        let storage = shared();
        {
            let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
            s.register_node_type(NodeTypeDefinition {
                node_type: "episode".into(),
                required_fields: Vec::new(),
                data_types: HashMap::new(),
                memory_class: Some(MemoryClass::Episodic),
            }).unwrap();
            s.close().unwrap();
        }
        let mut reopened = Store::new(Box::new(storage), StoreConfig::default());
        reopened.ensure_loaded().unwrap();
        assert_eq!(
            reopened.node_type_definitions.get("episode").unwrap().memory_class,
            Some(MemoryClass::Episodic),
        );
    }

    #[test]
    fn schema_definitions_reject_invalid_metadata() {
        let storage = shared();
        let mut store = Store::new(Box::new(storage.clone()), StoreConfig::default());
        assert!(matches!(store.register_node_type(NodeTypeDefinition {
            node_type: "".into(),
            required_fields: Vec::new(),
            data_types: HashMap::new(),
            memory_class: None,
        }), Err(PolypackError::InvalidArgument(message)) if message.contains("invalid schema")));
        assert!(matches!(store.register_node_type(NodeTypeDefinition {
            node_type: "person".into(),
            required_fields: vec!["name".into(), "name".into()],
            data_types: HashMap::new(),
            memory_class: None,
        }), Err(PolypackError::InvalidArgument(message)) if message.contains("invalid schema")));
        assert!(matches!(store.register_edge_type(EdgeTypeDefinition {
            edge_type: "RELATED".into(),
            source_types: vec!["person".into()],
            target_types: vec!["person".into()],
            required_fields: Vec::new(),
            data_types: HashMap::from([(String::from("weight"), String::from("date"))]),
            cardinality: Some("invalid".into()),
        }), Err(PolypackError::InvalidArgument(message)) if message.contains("invalid schema")));
        assert!(!storage.lock().unwrap().exists(SCHEMAS_FILE).unwrap());
    }

    #[test]
    fn schema_metadata_rejects_duplicate_definitions_on_reload() {
        let storage = shared();
        let duplicate = serde_json::json!({
            "nodeTypes": [
                { "nodeType": "person", "requiredFields": [], "dataTypes": {} },
                { "nodeType": "person", "requiredFields": [], "dataTypes": {} }
            ],
            "edgeTypes": []
        });
        storage.lock().unwrap().write(SCHEMAS_FILE, &serde_json::to_vec(&duplicate).unwrap()).unwrap();
        let mut store = Store::new(Box::new(storage), StoreConfig::default());
        assert!(matches!(store.node_count(), Err(PolypackError::CorruptData(message)) if message.contains("duplicate node type")));
    }

    #[test]
    fn edges_by_source_and_target_indexes() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage.clone()), StoreConfig::default());
        let edge = |id: &str, source: &str, target: &str, etype: &str| Edge {
            id: id.into(),
            source: source.into(),
            target: target.into(),
            edge_type: etype.into(),
            data: None,
            created_at: 1,
            revision: 0,
        };
        s.apply(&ChangeBatch {
            put_edges: vec![
                edge("a::R::b", "a", "b", "R"),
                edge("a::S::c", "a", "c", "S"),
                edge("d::R::b", "d", "b", "R"),
            ],
            ..Default::default()
        })
        .unwrap();

        let by_source = s.get_edges_by_sources(&["a".into()], Some("R")).unwrap();
        assert_eq!(by_source.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), vec!["a::R::b"]);
        let mut by_target = s.get_edges_by_targets(&["b".into()], Some("R")).unwrap();
        by_target.sort_by(|a, b| a.id.cmp(&b.id));
        assert_eq!(
            by_target.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
            vec!["a::R::b", "d::R::b"]
        );

        // delete keeps the indexes consistent
        s.apply(&ChangeBatch { delete_edge_ids: vec!["a::R::b".into()], ..Default::default() })
            .unwrap();
        assert_eq!(s.get_edges_by_sources(&["a".into()], None).unwrap().len(), 1);
        assert_eq!(s.get_edges_by_targets(&["b".into()], None).unwrap().len(), 1);
        assert_eq!(
            s.get_edges_by_targets(&["b".into()], Some("R")).unwrap()[0].id,
            "d::R::b"
        );

        s.close().unwrap();
        let mut s2 = Store::new(Box::new(storage.clone()), StoreConfig::default());
        assert_eq!(s2.get_edges_by_sources(&["a".into()], None).unwrap().len(), 1);
    }

    #[test]
    fn verify_reports_snapshot_contents_and_wal_mutations() {
        let storage = shared();
        let mut store = Store::new(Box::new(storage), StoreConfig::default());
        store.apply(&batch(&[node("a")])).unwrap();

        let before_compaction = store.verify().unwrap();
        assert!(before_compaction.ok);
        assert_eq!(before_compaction.mutation_count, 1);
        let page = store.mutation_log_page(0, 1).unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].operations[0].operation_type, "putNode");
        assert_eq!(store.latest_mutation_sequence().unwrap(), page[0].sequence);
        assert!(store.mutation_log_since(page[0].sequence).unwrap().is_empty());

        store.compact().unwrap();
        let after_compaction = store.verify().unwrap();
        assert!(after_compaction.ok);
        assert_eq!(after_compaction.node_count, 1);
        assert_eq!(after_compaction.edge_count, 0);
        assert_eq!(after_compaction.vector_count, 0);
    }

    #[test]
    fn verify_reports_invalid_index_metadata() {
        let storage = shared();
        storage.lock().unwrap().write(INDEXES_FILE, br#"[{"name":"dup","fields":["a"]},{"name":"dup","fields":[]}] "#).unwrap();
        let mut store = Store::new(Box::new(storage), StoreConfig::default());
        let report = store.verify().unwrap();
        assert!(!report.ok);
        assert!(report.errors.iter().any(|error| error.contains("index metadata")));
    }

    #[test]
    fn backup_and_restore_copy_all_durable_files() {
        let mut source = Store::new(Box::new(InMemoryStorage::new()), StoreConfig::default());
        source.apply(&batch(&[node("a")])).unwrap();
        source.write_auxiliary(INDEXES_FILE, br"[]").unwrap();
        let mut backup = InMemoryStorage::new();
        source.backup(&mut backup).unwrap();

        let mut restored = Store::restore(&backup, Box::new(InMemoryStorage::new()), StoreConfig::default()).unwrap();
        assert_eq!(restored.get_node("a").unwrap().unwrap().id, "a");
        assert_eq!(restored.latest_mutation_sequence().unwrap(), 1);
        assert!(restored.verify().unwrap().ok);
    }

    #[test]
    fn mutation_log_preserves_commit_metadata() {
        let storage = shared();
        let mut store = Store::new(Box::new(storage), StoreConfig::default());
        store
            .apply_with_metadata(
                &batch(&[node("metadata")]),
                Some("tx-import"),
                Some("actor-1"),
                Some(12),
                Some(serde_json::json!({"source": "test"})),
            )
            .unwrap();
        let record = store.mutation_log().unwrap().pop().unwrap();
        assert_eq!(record.transaction_id, "tx-import");
        assert_eq!(record.actor.as_deref(), Some("actor-1"));
        assert_eq!(record.base_revision, Some(12));
        assert_eq!(record.metadata, Some(serde_json::json!({"source": "test"})));
    }

    #[test]
    fn repeated_transaction_id_is_idempotent() {
        let storage = shared();
        let mut store = Store::new(Box::new(storage), StoreConfig::default());
        let changes = batch(&[node("once")]);
        store.apply_with_transaction(&changes, Some("tx-retry")).unwrap();
        store.apply_with_transaction(&batch(&[node("different")]), Some("tx-retry")).unwrap();
        assert_eq!(store.node_count().unwrap(), 1);
        assert!(store.get_node("once").unwrap().is_some());
        assert!(store.get_node("different").unwrap().is_none());
        assert_eq!(store.mutation_log().unwrap().len(), 1);
    }

    #[test]
    fn repeated_transaction_id_is_idempotent_after_reopen() {
        let storage = shared();
        let mut store = Store::new(Box::new(storage.clone()), StoreConfig::default());
        store.apply_with_transaction(&batch(&[node("once")]), Some("tx-retry")).unwrap();
        store.close().unwrap();

        // The identity cache is rebuilt from the on-disk mutation log when a
        // store is (re)loaded, so a retried transaction_id must still be
        // recognized as already-applied after a fresh Store instance loads it.
        let mut reopened = Store::new(Box::new(storage), StoreConfig::default());
        reopened.apply_with_transaction(&batch(&[node("different")]), Some("tx-retry")).unwrap();
        assert_eq!(reopened.node_count().unwrap(), 1);
        assert!(reopened.get_node("once").unwrap().is_some());
        assert!(reopened.get_node("different").unwrap().is_none());
        assert_eq!(reopened.mutation_log().unwrap().len(), 1);
    }

    #[test]
    fn writes_succeed_without_cloning_nodes_when_no_unique_index_exists() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage), StoreConfig::default());
        // A non-unique secondary index must not force validate_pending_indexes
        // to walk the fast (guarded) path; writes should still apply and stay
        // queryable by the index.
        s.define_index(SecondaryIndexDefinition {
            name: "book-genre".into(),
            fields: vec!["genre".into()],
            node_type: Some("book".into()),
            unique: false,
            sparse: false,
        }).unwrap();
        s.apply(&ChangeBatch { put_nodes: vec![
            Node { data: serde_json::json!({"genre": "fiction"}).as_object().unwrap().clone(), ..book("a", "fiction", 1.0) },
            Node { data: serde_json::json!({"genre": "fiction"}).as_object().unwrap().clone(), ..book("b", "fiction", 2.0) },
        ], ..Default::default() }).unwrap();
        let query = NodeQuery {
            node_types: Some(vec!["book".into()]),
            attributes: Some(serde_json::json!({"genre": "fiction"}).as_object().unwrap().clone()),
            ..Default::default()
        };
        assert_eq!(s.query_nodes(&query).unwrap().len(), 2);
    }

    #[test]
    fn identity_cache_not_updated_when_wal_append_fails_and_retry_still_commits() {
        struct FailOnceOnAppend {
            inner: InMemoryStorage,
            target: &'static str,
            armed: bool,
        }
        impl Storage for FailOnceOnAppend {
            fn read(&self, name: &str) -> Result<Option<Vec<u8>>> { self.inner.read(name) }
            fn write(&mut self, name: &str, data: &[u8]) -> Result<()> { self.inner.write(name, data) }
            fn append(&mut self, name: &str, data: &[u8]) -> Result<()> {
                if self.armed && name == self.target {
                    self.armed = false;
                    return Err(PolypackError::Storage("injected failure".into()));
                }
                self.inner.append(name, data)
            }
            fn delete(&mut self, name: &str) -> Result<()> { self.inner.delete(name) }
            fn exists(&self, name: &str) -> Result<bool> { self.inner.exists(name) }
        }

        let mut store = Store::new(Box::new(FailOnceOnAppend { inner: InMemoryStorage::new(), target: WAL_FILE, armed: true }), StoreConfig::default());
        let changes = batch(&[node("once")]);
        assert!(store.apply_with_transaction(&changes, Some("tx-fail")).is_err());
        assert_eq!(store.node_count().unwrap(), 0);
        assert_eq!(store.mutation_log().unwrap().len(), 0);

        // The WAL append failed before the identity was ever recorded, so the
        // retry must actually commit rather than being swallowed as a no-op.
        store.apply_with_transaction(&changes, Some("tx-fail")).unwrap();
        assert_eq!(store.node_count().unwrap(), 1);
        assert!(store.get_node("once").unwrap().is_some());
        assert_eq!(store.mutation_log().unwrap().len(), 1);
    }

    #[test]
    fn identity_cache_not_updated_when_mutation_log_append_fails_and_retry_still_commits() {
        struct FailOnceOnAppend {
            inner: InMemoryStorage,
            target: &'static str,
            armed: bool,
        }
        impl Storage for FailOnceOnAppend {
            fn read(&self, name: &str) -> Result<Option<Vec<u8>>> { self.inner.read(name) }
            fn write(&mut self, name: &str, data: &[u8]) -> Result<()> { self.inner.write(name, data) }
            fn append(&mut self, name: &str, data: &[u8]) -> Result<()> {
                if self.armed && name == self.target {
                    self.armed = false;
                    return Err(PolypackError::Storage("injected failure".into()));
                }
                self.inner.append(name, data)
            }
            fn delete(&mut self, name: &str) -> Result<()> { self.inner.delete(name) }
            fn exists(&self, name: &str) -> Result<bool> { self.inner.exists(name) }
        }

        let mut store = Store::new(Box::new(FailOnceOnAppend { inner: InMemoryStorage::new(), target: MUTATION_LOG_FILE, armed: true }), StoreConfig::default());
        let changes = batch(&[node("once")]);
        assert!(store.apply_with_transaction(&changes, Some("tx-fail")).is_err());
        assert_eq!(store.mutation_log().unwrap().len(), 0);

        // The WAL entry was appended but the mutation-log append (where the
        // identity cache is updated) failed, so a retry with the same
        // transaction_id must still commit, not be treated as already applied.
        store.apply_with_transaction(&changes, Some("tx-fail")).unwrap();
        assert!(store.get_node("once").unwrap().is_some());
        assert_eq!(store.mutation_log().unwrap().len(), 1);
    }

    #[test]
    fn identity_cache_is_not_left_half_built_when_mutation_log_is_corrupt() {
        let storage = shared();
        // Two lines: the log-parser tolerates a corrupt *final* line (a crash
        // mid-append), so put the bad line first to force a genuine parse error.
        storage.lock().unwrap().write(MUTATION_LOG_FILE, b"{ not json\n{ not json either\n").unwrap();

        let mut store = Store::new(Box::new(storage.clone()), StoreConfig::default());
        // ensure_loaded fails while parsing the corrupt log; `loaded` and the
        // identity caches must stay at their untouched defaults rather than a
        // partially-populated state, so a later valid load starts clean.
        assert!(store.apply(&batch(&[node("a")])).is_err());

        storage.lock().unwrap().write(MUTATION_LOG_FILE, b"").unwrap();
        let mut recovered = Store::new(Box::new(storage), StoreConfig::default());
        recovered.apply_with_transaction(&batch(&[node("a")]), Some("tx-1")).unwrap();
        assert!(recovered.get_node("a").unwrap().is_some());
        // A fresh Store's cache is rebuilt from persisted records, so a
        // retried transaction_id is still recognized after this reload.
        recovered.apply_with_transaction(&batch(&[node("b")]), Some("tx-1")).unwrap();
        assert!(recovered.get_node("b").unwrap().is_none());
    }

    #[test]
    fn grows_compaction_threshold_with_store() {
        let snapshots = Arc::new(Mutex::new(0usize));

        struct CountingStorage {
            inner: InMemoryStorage,
            snapshots: Arc<Mutex<usize>>,
        }
        impl Storage for CountingStorage {
            fn read(&self, name: &str) -> Result<Option<Vec<u8>>> {
                self.inner.read(name)
            }
            fn write(&mut self, name: &str, data: &[u8]) -> Result<()> {
                if name == SNAPSHOT_FILE {
                    *self.snapshots.lock().unwrap() += 1;
                }
                self.inner.write(name, data)
            }
            fn append(&mut self, name: &str, data: &[u8]) -> Result<()> {
                self.inner.append(name, data)
            }
            fn delete(&mut self, name: &str) -> Result<()> {
                self.inner.delete(name)
            }
            fn exists(&self, name: &str) -> Result<bool> {
                self.inner.exists(name)
            }
        }

        let storage = CountingStorage {
            inner: InMemoryStorage::new(),
            snapshots: snapshots.clone(),
        };
        let config = StoreConfig { compact_threshold: 2, ..Default::default() };
        let mut s = Store::new(Box::new(storage), StoreConfig { compact_threshold: 2, ..Default::default() });
        // A naive fixed threshold of 2 would rewrite the snapshot ~21 times for
        // 40 single-node applies; the adaptive max(2, records/4) keeps it lower.
        for i in 0..40 {
            s.apply(&batch(&[node(&format!("n{i}"))])).unwrap();
        }
        s.close().unwrap();
        let writes = *snapshots.lock().unwrap();
        assert!(writes > 1, "compaction must still run");
        assert!(writes < 15, "adaptive threshold must slow compaction: got {writes}");

        // Verify the adaptive path keeps data intact within a store lifetime.
        let mut s3 = Store::new(
            Box::new(CountingStorage { inner: InMemoryStorage::new(), snapshots: snapshots.clone() }),
            config,
        );
        for i in 0..40 {
            s3.apply(&batch(&[node(&format!("n{i}"))])).unwrap();
        }
        assert_eq!(s3.node_count().unwrap(), 40);
        s3.close().unwrap();
    }

    #[test]
    fn type_index_drops_stale_entry_when_a_node_changes_type() {
        let storage = shared();
        let mut s = Store::new(Box::new(storage), StoreConfig::default());
        let n1 = Node { id: "a".into(), node_type: "draft".into(), data: Default::default(), vector: None, inserted_at: 1, updated_at: 1, revision: 0, activation: None, ..Default::default()
};
        let n2 = Node { id: "a".into(), node_type: "published".into(), data: Default::default(), vector: None, inserted_at: 2, updated_at: 2, revision: 0, activation: None, ..Default::default()
};
        s.apply(&ChangeBatch { put_nodes: vec![n1], ..Default::default() }).unwrap();
        s.apply(&ChangeBatch { put_nodes: vec![n2], ..Default::default() }).unwrap();
        assert_eq!(
            s.count_nodes(&NodeQuery { node_types: Some(vec!["draft".into()]), ..Default::default() }).unwrap(),
            0
        );
        assert_eq!(
            s.count_nodes(&NodeQuery { node_types: Some(vec!["published".into()]), ..Default::default() }).unwrap(),
            1
        );
    }
}
