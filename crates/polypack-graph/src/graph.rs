//! [`Graph`]: the Rust counterpart to `PolyGraph` (`src/graph.ts`).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use polypack_core::activation::{
    activation_score_of, decay_activation_state, reinforce_activation, DEFAULT_ACTIVATION,
};
use polypack_core::model::{edge_id, validate_activation, validate_node, NodeActivation};
use polypack_core::storage::NodeQuery;
use polypack_core::{
    aggregate as core_aggregate, execute as core_execute, ChangeBatch, Edge, GraphSnapshot, HnswConfig,
    HnswIndex, Node, PolypackError, QueryPlan, Result, Storage, Store, StoreConfig,
};
use polypack_core::storage::{AdapterCapabilities, MutationRecord, VerificationReport, INDEXES_FILE};

use crate::edge::{decode_ownership, encode_ownership, EdgeEntry, EdgeOwnership};
use crate::embedding::{create_embedding, EmbeddingProvider, FeatureHashEmbedding};
use crate::event::GraphChangeEvent;
use crate::lru::LruList;
use crate::migration::{MigrationOptions, MigrationRegistry, MigrationReport};
use crate::persisted_query::{PersistedGraphQuery, QueryResourceLimits};
use crate::query::GraphQuery;

fn set_data_path(root: &mut serde_json::Value, path: &[&str], value: serde_json::Value) -> Result<()> {
    let Some(object) = root.as_object_mut() else {
        return Err(PolypackError::InvalidArgument("node data must be an object".into()));
    };
    if path.len() == 1 {
        object.insert(path[0].to_string(), value);
        return Ok(());
    }
    let child = object
        .entry(path[0].to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !child.is_object() {
        *child = serde_json::Value::Object(serde_json::Map::new());
    }
    set_data_path(child, &path[1..], value)
}

fn unset_data_path(root: &mut serde_json::Value, path: &[&str]) {
    let Some(object) = root.as_object_mut() else { return };
    if path.len() == 1 {
        object.remove(path[0]);
        return;
    }
    if let Some(child) = object.get_mut(path[0]) {
        unset_data_path(child, &path[1..]);
    }
}

fn get_data_path<'a>(root: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    let object = root.as_object()?;
    if path.len() == 1 {
        return object.get(path[0]);
    }
    get_data_path(object.get(path[0])?, &path[1..])
}

fn patch_parts(path: &str) -> Result<Vec<&str>> {
    let mut parts: Vec<&str> = path.split('.').collect();
    if parts.first() == Some(&"data") {
        parts.remove(0);
    }
    if parts.is_empty() || parts.iter().any(|part| part.is_empty()) {
        return Err(PolypackError::InvalidArgument("patch paths must target node data".into()));
    }
    Ok(parts)
}

pub(crate) fn indexed_value_key(node: &Node, index: &IndexDefinition) -> Option<String> {
    if index.node_type.as_deref().is_some_and(|node_type| node_type != node.node_type) {
        return None;
    }
    let data = serde_json::Value::Object(node.data.clone());
    let values: Vec<serde_json::Value> = index
        .fields
        .iter()
        .map(|field| get_data_path(&data, &field.split('.').collect::<Vec<_>>()).cloned().unwrap_or(serde_json::Value::Null))
        .collect();
    if index.sparse && values.iter().any(serde_json::Value::is_null) { return None; }
    Some(serde_json::to_string(&values).expect("JSON index key serialization cannot fail"))
}

fn matches_json_type(value: &serde_json::Value, expected: &str) -> bool {
    match expected {
        "string" => value.is_string(),
        "number" => value.as_f64().is_some(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

type OnChangeCallback = Box<dyn FnMut(GraphChangeEvent)>;
type OnOrphanCallback = Box<dyn FnMut(&str)>;

/// Tuning knobs, mirrors the `PolyGraph` constructor's `hotCacheMax`, the
/// `HnswIndex` config threaded through `createVectorIndex`, and the
/// `embedding` provider — all three are TS constructor arguments, bundled
/// here since Rust has no optional-positional-argument constructors.
pub struct GraphConfig {
    pub hot_cache_max: usize,
    pub hnsw: HnswConfig,
    pub embedding: Box<dyn EmbeddingProvider>,
    pub resource_limits: GraphResourceLimits,
}

/// Write-side safety limits for the hot graph. Query limits live on
/// [`QueryResourceLimits`] because they are scoped to individual queries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphResourceLimits {
    pub max_vector_dimensions: Option<usize>,
    pub max_node_payload_bytes: Option<usize>,
    pub max_batch_size: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphStats {
    pub loaded_node_count: usize,
    pub persisted_node_count: usize,
    pub edge_count: usize,
    pub vector_count: usize,
    pub dirty_record_count: usize,
    pub pending_persistence: bool,
    pub index_count: usize,
    pub query_count: usize,
    pub query_duration_ms: f64,
    pub query_scanned_records: usize,
    pub query_index_usage: HashMap<String, usize>,
}

#[derive(Default)]
pub(crate) struct QueryMetrics {
    pub count: usize,
    pub duration_ms: f64,
    pub scanned_records: usize,
    pub index_usage: HashMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeTypeDefinition {
    pub required_fields: Vec<String>,
    pub data_types: HashMap<String, String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EdgeTypeDefinition {
    pub source_types: Vec<String>,
    pub target_types: Vec<String>,
    pub cardinality: Option<String>,
    pub required_fields: Vec<String>,
    pub data_types: HashMap<String, String>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexDefinition {
    pub name: String,
    pub node_type: Option<String>,
    pub fields: Vec<String>,
    pub unique: bool,
    pub sparse: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self { hot_cache_max: 50_000, hnsw: HnswConfig::default(), embedding: Box::new(FeatureHashEmbedding::default()), resource_limits: GraphResourceLimits::default() }
    }
}

/// Stateful property graph: hot node/edge working set + vector index, backed
/// by a [`Store`] for durability. See the module docs in `lib.rs` for how
/// this relates to `PolyGraph`.
pub struct Graph {
    store: Store,
    hnsw: HnswIndex,
    config: GraphConfig,

    // hot working set
    nodes: HashMap<String, Node>,
    edges: HashMap<String, HashMap<String, EdgeEntry>>,
    node_to_edge: HashMap<String, HashSet<String>>,
    by_type: HashMap<String, HashSet<String>>,
    /// O(1) LRU order for hot-cache eviction (front = least-recently-used).
    hot_cache_order: LruList,
    /// Snapshots of dirty nodes evicted before their edits were flushed,
    /// mirrors `PolyGraph.evictedDirtyNodes`. Without this, evicting a node
    /// that hasn't been persisted yet would silently lose its edits — both
    /// `flush` and `get_node_safe` check here before falling back to `Store`.
    evicted_dirty_nodes: HashMap<String, Node>,
    /// Mirrors `PolyGraph.evictionSkipCounter`: `evict_oldest_if_over_cap`
    /// only runs every 10th `touch_hot_cache`, since `flush` already runs it
    /// unconditionally after every successful apply.
    eviction_skip_counter: u64,

    // dirty tracking, mirrors `dirtyNodes`/`dirtyEdges`/`dirtyVectors`/etc.
    dirty_nodes: HashSet<String>,
    dirty_edges: HashSet<String>,
    dirty_vectors: HashSet<String>,
    removed_node_ids: HashSet<String>,
    removed_edge_ids: HashSet<String>,
    removed_vector_ids: HashSet<String>,

    on_change: Option<OnChangeCallback>,
    on_orphan: Option<OnOrphanCallback>,

    /// Mirrors `PolyGraph.batchDepth`/`pendingBatchEvents`: while > 0,
    /// `emit` queues events instead of dispatching them — see `start_batch`.
    batch_depth: u32,
    pending_batch_events: Vec<GraphChangeEvent>,

    warmed: bool,
    transaction_active: bool,
    transaction_mutation_count: usize,
    transaction_sequence: u64,
    current_transaction_id: Option<String>,
    current_operation_id: Option<String>,
    node_type_definitions: HashMap<String, NodeTypeDefinition>,
    edge_type_definitions: HashMap<String, EdgeTypeDefinition>,
    indexes: HashMap<String, IndexDefinition>,
    secondary_indexes: HashMap<String, HashMap<String, HashSet<String>>>,
    migrations: MigrationRegistry,
    pub(crate) query_metrics: RefCell<QueryMetrics>,
}

struct GraphCheckpoint {
    nodes: HashMap<String, Node>,
    edges: HashMap<String, HashMap<String, EdgeEntry>>,
    node_to_edge: HashMap<String, HashSet<String>>,
    by_type: HashMap<String, HashSet<String>>,
    hot_cache_order: LruList,
    evicted_dirty_nodes: HashMap<String, Node>,
    hnsw: HnswIndex,
    dirty_nodes: HashSet<String>,
    dirty_edges: HashSet<String>,
    dirty_vectors: HashSet<String>,
    removed_node_ids: HashSet<String>,
    removed_edge_ids: HashSet<String>,
    removed_vector_ids: HashSet<String>,
    batch_depth: u32,
    pending_batch_events: Vec<GraphChangeEvent>,
    secondary_indexes: HashMap<String, HashMap<String, HashSet<String>>>,
    warmed: bool,
    transaction_mutation_count: usize,
}

impl GraphCheckpoint {
    fn capture(graph: &Graph) -> Self {
        Self {
            nodes: graph.nodes.clone(),
            edges: graph.edges.clone(),
            node_to_edge: graph.node_to_edge.clone(),
            by_type: graph.by_type.clone(),
            hot_cache_order: graph.hot_cache_order.clone(),
            evicted_dirty_nodes: graph.evicted_dirty_nodes.clone(),
            hnsw: graph.hnsw.clone(),
            dirty_nodes: graph.dirty_nodes.clone(),
            dirty_edges: graph.dirty_edges.clone(),
            dirty_vectors: graph.dirty_vectors.clone(),
            removed_node_ids: graph.removed_node_ids.clone(),
            removed_edge_ids: graph.removed_edge_ids.clone(),
            removed_vector_ids: graph.removed_vector_ids.clone(),
            batch_depth: graph.batch_depth,
            pending_batch_events: graph.pending_batch_events.clone(),
            secondary_indexes: graph.secondary_indexes.clone(),
            warmed: graph.warmed,
            transaction_mutation_count: graph.transaction_mutation_count,
        }
    }

    fn restore(self, graph: &mut Graph) {
        graph.nodes = self.nodes;
        graph.edges = self.edges;
        graph.node_to_edge = self.node_to_edge;
        graph.by_type = self.by_type;
        graph.hot_cache_order = self.hot_cache_order;
        graph.evicted_dirty_nodes = self.evicted_dirty_nodes;
        graph.hnsw = self.hnsw;
        graph.dirty_nodes = self.dirty_nodes;
        graph.dirty_edges = self.dirty_edges;
        graph.dirty_vectors = self.dirty_vectors;
        graph.removed_node_ids = self.removed_node_ids;
        graph.removed_edge_ids = self.removed_edge_ids;
        graph.removed_vector_ids = self.removed_vector_ids;
        graph.batch_depth = self.batch_depth;
        graph.pending_batch_events = self.pending_batch_events;
        graph.secondary_indexes = self.secondary_indexes;
        graph.warmed = self.warmed;
        graph.transaction_mutation_count = self.transaction_mutation_count;
    }
}

impl Graph {
    /// Open a graph over `storage`, creating the backing [`Store`].
    pub fn open(storage: Box<dyn Storage>, store_config: StoreConfig, config: GraphConfig) -> Result<Self> {
        let store = Store::new(storage, store_config);
        let hnsw = HnswIndex::new(config.hnsw, 0)?;
        let mut graph = Self {
            store,
            hnsw,
            config,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            node_to_edge: HashMap::new(),
            by_type: HashMap::new(),
            hot_cache_order: LruList::new(),
            evicted_dirty_nodes: HashMap::new(),
            eviction_skip_counter: 0,
            dirty_nodes: HashSet::new(),
            dirty_edges: HashSet::new(),
            dirty_vectors: HashSet::new(),
            removed_node_ids: HashSet::new(),
            removed_edge_ids: HashSet::new(),
            removed_vector_ids: HashSet::new(),
            on_change: None,
            on_orphan: None,
            batch_depth: 0,
            pending_batch_events: Vec::new(),
            warmed: false,
            transaction_active: false,
            transaction_mutation_count: 0,
            transaction_sequence: 0,
            current_transaction_id: None,
            current_operation_id: None,
            node_type_definitions: HashMap::new(),
            edge_type_definitions: HashMap::new(),
            indexes: HashMap::new(),
            secondary_indexes: HashMap::new(),
            migrations: MigrationRegistry::default(),
            query_metrics: RefCell::new(QueryMetrics::default()),
        };
        graph.load_index_metadata()?;
        Ok(graph)
    }

    /// Report the guarantees declared by the underlying storage adapter.
    pub fn capabilities(&self) -> AdapterCapabilities {
        self.store.capabilities()
    }

    /// Replace the write-side resource limits for subsequent mutations.
    /// Limits are intentionally validated at configuration time as positive
    /// values, matching the TypeScript and Python APIs.
    pub fn set_resource_limits(&mut self, limits: GraphResourceLimits) -> Result<()> {
        for (name, value) in [
            ("maxVectorDimensions", limits.max_vector_dimensions),
            ("maxNodePayloadBytes", limits.max_node_payload_bytes),
            ("maxBatchSize", limits.max_batch_size),
        ] {
            if value == Some(0) {
                return Err(PolypackError::InvalidArgument(format!("{name} must be a positive integer")));
            }
        }
        self.config.resource_limits = limits;
        Ok(())
    }

    /// Return the active write-side resource limits.
    pub fn resource_limit_config(&self) -> GraphResourceLimits {
        self.config.resource_limits.clone()
    }

    /// Reject a graph configuration unless the backing adapter declares every
    /// requested guarantee. `vector_search` is treated as a minimum: ANN
    /// satisfies exact-search requirements, while `None` imposes no vector
    /// requirement.
    pub fn require_capabilities(&self, required: AdapterCapabilities) -> Result<()> {
        let actual = self.capabilities();
        let checks = [
            (required.atomic_batches, actual.atomic_batches, "atomic_batches"),
            (required.transactions, actual.transactions, "transactions"),
            (required.fsync, actual.fsync, "fsync"),
            (required.secondary_indexes, actual.secondary_indexes, "secondary_indexes"),
            (required.snapshots, actual.snapshots, "snapshots"),
            (required.change_feed, actual.change_feed, "change_feed"),
            (required.concurrent_writers, actual.concurrent_writers, "concurrent_writers"),
        ];
        if let Some((_, _, name)) = checks.into_iter().find(|(needed, supported, _)| *needed && !*supported) {
            return Err(PolypackError::InvalidArgument(format!("persistence adapter does not support capability: {name}")));
        }
        let vector_ok = match required.vector_search {
            polypack_core::storage::VectorSearchCapability::None => true,
            polypack_core::storage::VectorSearchCapability::Exact => actual.vector_search != polypack_core::storage::VectorSearchCapability::None,
            polypack_core::storage::VectorSearchCapability::Ann => actual.vector_search == polypack_core::storage::VectorSearchCapability::Ann,
        };
        if !vector_ok {
            return Err(PolypackError::InvalidArgument("persistence adapter does not support capability: vector_search".into()));
        }
        Ok(())
    }

    /// Define a node-data index. Unique indexes are enforced before every
    /// mutation; equality and single-field numeric range queries use the
    /// maintained candidate buckets with full predicate validation retained.
    pub fn define_index(&mut self, definition: IndexDefinition) -> Result<()> {
        if definition.name.is_empty() || definition.fields.is_empty() || definition.fields.iter().any(|field| field.is_empty()) || definition.fields.iter().collect::<HashSet<_>>().len() != definition.fields.len() {
            return Err(PolypackError::InvalidArgument("index name and fields must not be empty".into()));
        }
        if self.indexes.contains_key(&definition.name) {
            return Err(PolypackError::InvalidArgument(format!("index {} is already defined", definition.name)));
        }
        self.indexes.insert(definition.name.clone(), definition.clone());
        self.secondary_indexes.insert(definition.name.clone(), HashMap::new());
        let mut indexed_nodes: HashMap<String, Node> = self.store.query_nodes(&NodeQuery::default())?.into_iter().map(|node| (node.id.clone(), node)).collect();
        indexed_nodes.extend(self.nodes.iter().map(|(id, node)| (id.clone(), node.clone())));
        let validation = indexed_nodes.values().try_for_each(|node| self.validate_node_indexes(node, Some(&node.id)));
        if let Err(error) = validation {
            self.indexes.remove(&definition.name);
            self.secondary_indexes.remove(&definition.name);
            return Err(error);
        }
        let nodes: Vec<Node> = indexed_nodes.into_values().collect();
        for node in &nodes { self.add_secondary_index_entry(node); }
        if let Err(error) = self.persist_index_metadata() {
            self.drop_index_memory(&definition.name);
            return Err(error);
        }
        Ok(())
    }

    pub fn drop_index(&mut self, name: &str) -> Result<bool> {
        if !self.indexes.contains_key(name) { return Ok(false); }
        let definition = self.indexes.remove(name).expect("index existence checked");
        self.secondary_indexes.remove(name);
        if let Err(error) = self.persist_index_metadata() {
            self.secondary_indexes.insert(name.to_string(), HashMap::new());
            self.indexes.insert(name.to_string(), definition);
            let mut indexed_nodes: HashMap<String, Node> = self.store.query_nodes(&NodeQuery::default())?.into_iter().map(|node| (node.id.clone(), node)).collect();
            indexed_nodes.extend(self.nodes.iter().map(|(id, node)| (id.clone(), node.clone())));
            let nodes: Vec<Node> = indexed_nodes.into_values().collect();
            for node in &nodes { self.add_secondary_index_entry(node); }
            return Err(error);
        }
        Ok(true)
    }

    pub fn indexes(&self) -> Vec<IndexDefinition> {
        let mut indexes: Vec<_> = self.indexes.values().cloned().collect();
        indexes.sort_by(|a, b| a.name.cmp(&b.name));
        indexes
    }

    fn drop_index_memory(&mut self, name: &str) {
        self.secondary_indexes.remove(name);
        self.indexes.remove(name);
    }

    fn load_index_metadata(&mut self) -> Result<()> {
        let Some(data) = self.store.read_auxiliary(INDEXES_FILE)? else { return Ok(()); };
        let definitions: Vec<IndexDefinition> = serde_json::from_slice(&data)
            .map_err(|error| PolypackError::CorruptData(format!("index metadata: {error}")))?;
        for definition in definitions {
            if definition.name.is_empty() || definition.fields.is_empty() || definition.fields.iter().any(|field| field.is_empty()) || definition.fields.iter().collect::<HashSet<_>>().len() != definition.fields.len() || self.indexes.contains_key(&definition.name) {
                return Err(PolypackError::CorruptData("invalid index metadata".into()));
            }
            self.secondary_indexes.insert(definition.name.clone(), HashMap::new());
            self.indexes.insert(definition.name.clone(), definition);
        }
        // `Store` materializes its persisted records independently of the
        // graph's hot cache. Build the lookup buckets from that full view so
        // persisted queries remain indexed after reopen and cache eviction.
        let nodes = self.store.query_nodes(&NodeQuery::default())?;
        for node in &nodes {
            self.add_secondary_index_entry(node);
        }
        Ok(())
    }

    fn persist_index_metadata(&mut self) -> Result<()> {
        let data = serde_json::to_vec(&self.indexes())
            .map_err(|error| PolypackError::InvalidArgument(format!("index metadata: {error}")))?;
        self.store.write_auxiliary(INDEXES_FILE, &data)
    }

    /// Return the active transaction identifier while inside a transaction.
    pub fn transaction_id(&self) -> Option<&str> {
        self.current_transaction_id.as_deref()
    }

    /// Return the active logical operation identifier while in a transaction.
    pub fn operation_id(&self) -> Option<&str> {
        self.current_operation_id.as_deref()
    }

    /// Verify the persisted snapshot, WAL, records, endpoints, and vectors.
    pub fn verify(&mut self) -> Result<VerificationReport> {
        self.store.verify()
    }

    /// Return the durable logical mutation history for replication and audit.
    pub fn mutation_log(&mut self) -> Result<Vec<MutationRecord>> {
        self.store.mutation_log()
    }

    /// Return durable mutations after a global sequence cursor.
    pub fn mutation_log_since(&mut self, sequence: u64) -> Result<Vec<MutationRecord>> {
        self.store.mutation_log_since(sequence)
    }

    /// Return at most `limit` durable mutations after a sequence cursor.
    pub fn mutation_log_page(&mut self, sequence: u64, limit: usize) -> Result<Vec<MutationRecord>> {
        self.store.mutation_log_page(sequence, limit)
    }

    /// Return the latest acknowledged logical mutation cursor.
    pub fn latest_mutation_sequence(&mut self) -> Result<u64> {
        self.store.latest_mutation_sequence()
    }

    pub fn register_node_type(&mut self, node_type: impl Into<String>, definition: NodeTypeDefinition) -> Result<()> {
        let node_type = node_type.into();
        let previous = self.node_type_definitions.insert(node_type.clone(), definition);
        let validation = self.nodes.values().filter(|node| node.node_type == node_type).try_for_each(|node| self.validate_node_schema(node));
        if validation.is_err() {
            if let Some(previous) = previous {
                self.node_type_definitions.insert(node_type, previous);
            } else {
                self.node_type_definitions.remove(&node_type);
            }
        }
        validation
    }

    pub fn register_edge_type(&mut self, edge_type: impl Into<String>, definition: EdgeTypeDefinition) -> Result<()> {
        let edge_type = edge_type.into();
        let previous = self.edge_type_definitions.insert(edge_type.clone(), definition);
        let validation = self.edges.iter().try_for_each(|(source, edges)| {
            edges.values().filter(|edge| edge.edge_type == edge_type).try_for_each(|edge| {
                self.validate_edge_schema(source, &edge_type, &edge.target, Some(&edge.id))?;
                self.validate_edge_data_schema(&edge_type, edge.data.as_ref())
            })
        });
        if validation.is_err() {
            if let Some(previous) = previous {
                self.edge_type_definitions.insert(edge_type.clone(), previous);
            } else {
                self.edge_type_definitions.remove(&edge_type);
            }
        }
        validation
    }

    /// Register an application-schema migration step.
    pub fn register_migration(&mut self, definition: crate::migration::MigrationDefinition) -> Result<()> {
        self.migrations.register(definition)
    }

    /// Migrate records from one application schema version to another.
    ///
    /// All transformed records are validated before the transaction starts.
    /// A migration may change record data, but not node IDs/types or edge
    /// identity/endpoints; topology migrations should use explicit graph CRUD.
    pub fn migrate(&mut self, from: u32, to: u32, options: MigrationOptions) -> Result<MigrationReport> {
        if options.batch_size == 0 {
            return Err(PolypackError::InvalidArgument("migration batch_size must be positive".into()));
        }
        self.warm()?;
        let mut node_ids: Vec<String> = self.nodes.keys().cloned().collect();
        node_ids.sort();
        if let Some(resume) = &options.resume_after_node { node_ids.retain(|id| id > resume); }
        let mut edge_records: Vec<Edge> = self.edges.iter().flat_map(|(source, edges)| edges.values().map(move |entry| Edge {
            id: entry.id.clone(), source: source.clone(), target: entry.target.clone(), edge_type: entry.edge_type.clone(),
            data: encode_ownership(entry.data.clone(), entry.ownership), created_at: now_millis(), revision: entry.revision,
        })).collect();
        edge_records.sort_by(|a, b| a.id.cmp(&b.id));
        if let Some(resume) = &options.resume_after_edge { edge_records.retain(|edge| edge.id > *resume); }

        let mut nodes = Vec::with_capacity(node_ids.len());
        let mut migrated_nodes = 0;
        let mut processed_nodes_so_far = 0;
        for batch in node_ids.chunks(options.batch_size) {
            for id in batch {
                let original = self.nodes.get(id).expect("node ID came from graph").clone();
                let migrated = self.migrations.migrate_node(original.clone(), from, to)?;
                if migrated.id != *id || migrated.node_type != self.nodes[id].node_type {
                    return Err(PolypackError::InvalidArgument(format!("migration changed identity or type of node {id}")));
                }
                validate_node(&migrated)?;
                self.validate_node_schema(&migrated)?;
                if migrated != original { migrated_nodes += 1; nodes.push(migrated); }
            }
            processed_nodes_so_far += batch.len();
            if let Some(on_progress) = &options.on_progress {
                on_progress(crate::migration::MigrationProgress { from, to, processed_nodes: processed_nodes_so_far, processed_edges: 0, migrated_nodes, migrated_edges: 0, dry_run: options.dry_run });
            }
        }
        let mut edges = Vec::with_capacity(edge_records.len());
        let mut migrated_edges = 0;
        let mut processed_edges_so_far = 0;
        for batch in edge_records.chunks(options.batch_size) {
            for edge in batch {
                let migrated = self.migrations.migrate_edge(edge.clone(), from, to)?;
                if migrated.id != edge.id || migrated.source != edge.source || migrated.target != edge.target || migrated.edge_type != edge.edge_type {
                    return Err(PolypackError::InvalidArgument(format!("migration changed identity or endpoints of edge {}", edge.id)));
                }
                self.validate_edge_data_schema(&migrated.edge_type, migrated.data.as_ref())?;
                if migrated != *edge { migrated_edges += 1; edges.push(migrated); }
            }
            processed_edges_so_far += batch.len();
            if let Some(on_progress) = &options.on_progress {
                on_progress(crate::migration::MigrationProgress { from, to, processed_nodes: processed_nodes_so_far, processed_edges: processed_edges_so_far, migrated_nodes, migrated_edges, dry_run: options.dry_run });
            }
        }
        let processed_nodes = node_ids.len();
        let processed_edges = self.edges.values().map(HashMap::len).sum();
        let report = MigrationReport { from, to, processed_nodes, processed_edges, migrated_nodes, migrated_edges, dry_run: options.dry_run };
        if options.dry_run || from == to { return Ok(report); }
        self.transaction(|graph| {
            for node in nodes { graph.add_node(node)?; }
            for edge in edges {
                let (ownership, data) = decode_ownership(edge.data.clone());
                graph.update_edge_if_revision(&edge.id, data, Some(ownership), edge.revision)?;
            }
            Ok(())
        })?;
        // Keep the report fields explicit for callers even if future batching
        // changes the internal processing counters.
        Ok(report)
    }

    fn validate_node_schema(&self, node: &Node) -> Result<()> {
        let Some(definition) = self.node_type_definitions.get(&node.node_type) else { return Ok(()) };
        let data = serde_json::Value::Object(node.data.clone());
        for field in &definition.required_fields {
            let parts: Vec<&str> = field.split('.').collect();
            if parts.iter().any(|part| part.is_empty()) || get_data_path(&data, &parts).is_none() {
                return Err(PolypackError::InvalidArgument(format!("node {} is missing required field {field}", node.id)));
            }
        }
        for (field, expected) in &definition.data_types {
            let parts: Vec<&str> = field.split('.').collect();
            if let Some(value) = get_data_path(&data, &parts) {
                if !matches_json_type(value, expected) {
                    return Err(PolypackError::InvalidArgument(format!("node {} field {field} must be {expected}", node.id)));
                }
            }
        }
        Ok(())
    }

    fn validate_node_resource_limits(&self, node: &Node) -> Result<()> {
        if let Some(limit) = self.config.resource_limits.max_vector_dimensions {
            if node.vector.as_ref().is_some_and(|vector| vector.len() > limit) {
                return Err(PolypackError::ResourceLimit { name: "maxVectorDimensions".into(), limit });
            }
        }
        if let Some(limit) = self.config.resource_limits.max_node_payload_bytes {
            let payload = serde_json::to_vec(&node.data).map_err(|error| PolypackError::InvalidArgument(format!("node payload is not serializable: {error}")))?;
            if payload.len() > limit {
                return Err(PolypackError::ResourceLimit { name: "maxNodePayloadBytes".into(), limit });
            }
        }
        Ok(())
    }

    fn validate_node_indexes(&self, node: &Node, exclude_id: Option<&str>) -> Result<()> {
        for index in self.indexes.values().filter(|index| index.unique && index.node_type.as_deref().is_none_or(|node_type| node_type == node.node_type)) {
            let values: Vec<_> = index.fields.iter().map(|field| get_data_path(&serde_json::Value::Object(node.data.clone()), &field.split('.').collect::<Vec<_>>()).cloned()).collect();
            if index.sparse && values.iter().any(Option::is_none) { continue; }
            for other in self.nodes.values().filter(|other| Some(other.id.as_str()) != exclude_id && index.node_type.as_deref().is_none_or(|node_type| node_type == other.node_type)) {
                let other_values: Vec<_> = index.fields.iter().map(|field| get_data_path(&serde_json::Value::Object(other.data.clone()), &field.split('.').collect::<Vec<_>>()).cloned()).collect();
                if values == other_values {
                    return Err(PolypackError::InvalidArgument(format!("unique index {} would be violated by node {}", index.name, node.id)));
                }
            }
        }
        Ok(())
    }

    fn validate_edge_data_schema(&self, edge_type: &str, data: Option<&serde_json::Map<String, serde_json::Value>>) -> Result<()> {
        let Some(definition) = self.edge_type_definitions.get(edge_type) else { return Ok(()) };
        let value = serde_json::Value::Object(data.cloned().unwrap_or_default());
        for field in &definition.required_fields {
            let parts: Vec<&str> = field.split('.').collect();
            if parts.iter().any(|part| part.is_empty()) || get_data_path(&value, &parts).is_none() {
                return Err(PolypackError::InvalidArgument(format!("edge {edge_type} is missing required field {field}")));
            }
        }
        for (field, expected) in &definition.data_types {
            let parts: Vec<&str> = field.split('.').collect();
            if let Some(field_value) = get_data_path(&value, &parts) {
                if !matches_json_type(field_value, expected) {
                    return Err(PolypackError::InvalidArgument(format!("edge {edge_type} field {field} must be {expected}")));
                }
            }
        }
        Ok(())
    }

    fn validate_edge_schema(&self, source: &str, edge_type: &str, target: &str, exclude_id: Option<&str>) -> Result<()> {
        let Some(definition) = self.edge_type_definitions.get(edge_type) else { return Ok(()) };
        if !self.nodes.contains_key(source) || !self.nodes.contains_key(target) {
            return Err(PolypackError::InvalidArgument(format!("edge {edge_type} references a missing endpoint")));
        }
        if !definition.source_types.is_empty() {
            let source_type = self.nodes.get(source).map(|node| node.node_type.as_str());
            if source_type.is_none() || !definition.source_types.iter().any(|allowed| Some(allowed.as_str()) == source_type) {
                return Err(PolypackError::InvalidArgument(format!("edge source type is not permitted for {edge_type}")));
            }
        }
        if !definition.target_types.is_empty() {
            let target_type = self.nodes.get(target).map(|node| node.node_type.as_str());
            if target_type.is_none() || !definition.target_types.iter().any(|allowed| Some(allowed.as_str()) == target_type) {
                return Err(PolypackError::InvalidArgument(format!("edge target type is not permitted for {edge_type}")));
            }
        }
        if let Some(cardinality) = &definition.cardinality {
            if cardinality != "many-to-many" {
                let mut outgoing = 0;
                let mut incoming = 0;
                for (candidate_source, edges) in &self.edges {
                    for candidate in edges.values() {
                        if exclude_id == Some(candidate.id.as_str()) || candidate.edge_type != *edge_type {
                            continue;
                        }
                        outgoing += usize::from(candidate_source == source);
                        incoming += usize::from(candidate.target == target);
                    }
                }
                let violates = (cardinality == "one-to-one" && (outgoing > 0 || incoming > 0))
                    || (cardinality == "one-to-many" && incoming > 0)
                    || (cardinality == "many-to-one" && outgoing > 0);
                if violates {
                    return Err(PolypackError::InvalidArgument(format!("edge cardinality {cardinality} would be exceeded")));
                }
            }
        }
        Ok(())
    }

    /// Return operational counters without coupling the graph to a metrics
    /// package.
    pub fn stats(&mut self) -> Result<GraphStats> {
        Ok(GraphStats {
            loaded_node_count: self.nodes.len(),
            persisted_node_count: self.store.node_count()?,
            edge_count: self.store.edges_snapshot()?.len(),
            vector_count: self.store.vectors_snapshot()?.len(),
            dirty_record_count: self.dirty_nodes.len()
                + self.dirty_edges.len()
                + self.dirty_vectors.len()
                + self.removed_node_ids.len()
                + self.removed_edge_ids.len()
                + self.removed_vector_ids.len(),
            pending_persistence: self.has_pending_persistence(),
            index_count: self.indexes.len(),
            query_count: self.query_metrics.borrow().count,
            query_duration_ms: self.query_metrics.borrow().duration_ms,
            query_scanned_records: self.query_metrics.borrow().scanned_records,
            query_index_usage: self.query_metrics.borrow().index_usage.clone(),
        })
    }

    /// Register a callback invoked for every mutation. Replaces any
    /// previously registered callback.
    pub fn on_change(&mut self, cb: impl FnMut(GraphChangeEvent) + 'static) {
        self.on_change = Some(Box::new(cb));
    }

    /// Register a callback invoked when a 'shared'-owned edge target becomes
    /// disconnected. See `onOrphan` in `graph.ts`.
    pub fn on_orphan(&mut self, cb: impl FnMut(&str) + 'static) {
        self.on_orphan = Some(Box::new(cb));
    }

    /// Queue `on_change` notifications until the matching `end_batch`.
    /// Nestable. Mirrors `PolyGraph.startBatch`.
    pub fn start_batch(&mut self) {
        self.batch_depth += 1;
    }

    /// Flush notifications queued since the matching `start_batch`. Errors
    /// if no batch is open. Mirrors `PolyGraph.endBatch`.
    pub fn end_batch(&mut self) -> Result<()> {
        if self.batch_depth == 0 {
            return Err(PolypackError::InvalidArgument("end_batch without start_batch".into()));
        }
        self.batch_depth -= 1;
        if self.batch_depth > 0 {
            return Ok(());
        }
        let events = std::mem::take(&mut self.pending_batch_events);
        for event in events {
            if let Some(cb) = self.on_change.as_mut() {
                cb(event);
            }
        }
        Ok(())
    }

    // ── lifecycle ──

    /// Execute a checkpointed transaction. Mutations performed by the
    /// callback are immediately readable, but queued change events are only
    /// released after persistence succeeds. Any callback or persistence error
    /// restores the complete in-memory graph state.
    pub fn transaction<F, T>(&mut self, callback: F) -> Result<T>
    where
        F: FnOnce(&mut Graph) -> Result<T>,
    {
        self.transaction_with_identity(None, callback)
    }

    /// Execute a transaction with a caller-supplied operation identifier.
    pub fn transaction_with_identity<F, T>(&mut self, operation_id: Option<String>, callback: F) -> Result<T>
    where
        F: FnOnce(&mut Graph) -> Result<T>,
    {
        self.require_capabilities(AdapterCapabilities { atomic_batches: true, transactions: true, ..Default::default() })?;
        if self.transaction_active {
            return Err(PolypackError::InvalidArgument("nested transactions are not supported".into()));
        }
        let checkpoint = GraphCheckpoint::capture(self);
        self.transaction_sequence = self.transaction_sequence.saturating_add(1);
        self.current_transaction_id = Some(format!("tx-{}-{}", now_millis(), self.transaction_sequence));
        self.current_operation_id = Some(operation_id.unwrap_or_else(|| format!("op-{}-{}", now_millis(), self.transaction_sequence)));
        self.transaction_active = true;
        self.transaction_mutation_count = 0;
        self.start_batch();
        let result = callback(self);
        match result {
            Ok(value) => {
                if let Err(error) = self.flush().and_then(|_| self.end_batch()) {
                    self.transaction_active = false;
                    self.transaction_mutation_count = 0;
                    self.current_transaction_id = None;
                    self.current_operation_id = None;
                    checkpoint.restore(self);
                    return Err(error);
                }
                self.transaction_active = false;
                self.transaction_mutation_count = 0;
                self.current_transaction_id = None;
                self.current_operation_id = None;
                Ok(value)
            }
            Err(error) => {
                self.transaction_active = false;
                self.transaction_mutation_count = 0;
                self.current_transaction_id = None;
                self.current_operation_id = None;
                checkpoint.restore(self);
                Err(error)
            }
        }
    }

    /// Load persisted nodes, vectors, and edges into the hot working set.
    /// Idempotent until `clear()`. Mirrors `PolyGraph.warm`.
    ///
    /// Diverges from the TS version in how "hot" is chosen: `Store` isn't a
    /// lazy persistence adapter — `ensure_loaded` already materializes every
    /// node in memory — so there's no JS-`Map`-insertion-order to lean on
    /// for "the newest `hot_cache_max` nodes". This loads every persisted
    /// node, sorts by `inserted_at` descending, and hot-caches the newest
    /// `hot_cache_max` of them: the same intent (keep the newest, cap the
    /// working set), expressed with the ordering information actually
    /// available here. Because the cap is applied before loading, this never
    /// needs `touch_hot_cache`'s eviction to run.
    pub fn warm(&mut self) -> Result<()> {
        if self.warmed {
            return Ok(());
        }
        self.warmed = true;

        let mut all_nodes = self.store.query_nodes(&NodeQuery::default())?;
        if all_nodes.is_empty() {
            return Ok(());
        }
        all_nodes.sort_by_key(|n| std::cmp::Reverse(n.inserted_at));
        all_nodes.truncate(self.config.hot_cache_max);

        for node in all_nodes {
            if self.nodes.contains_key(&node.id) {
                continue;
            }
            let id = node.id.clone();
            let node_type = node.node_type.clone();
            let vector = node.vector.clone();
            self.nodes.insert(id.clone(), node);
            self.index_node(&id, &node_type);
            self.touch_hot_cache(&id);
            if let Some(vector) = vector {
                // Hydration of already-persisted data, not a fresh add — no
                // `dirty_vectors` marking (mirrors `vectors.hydrate`).
                self.hnsw.add(&id, &vector)?;
            }
        }

        self.rebuild_edge_index()?;

        let types: Vec<String> = self.by_type.keys().cloned().collect();
        for node_type in types {
            self.emit(GraphChangeEvent::NodeAdded { node_id: None, node_type });
        }

        Ok(())
    }

    /// Alias for `warm`. Mirrors `PolyGraph.load`.
    pub fn load(&mut self) -> Result<()> {
        self.warm()
    }

    /// Rebuild the in-memory edge index from every persisted edge. Mirrors
    /// `PolyGraph.rebuildEdgeIndex`.
    fn rebuild_edge_index(&mut self) -> Result<()> {
        self.edges.clear();
        self.node_to_edge.clear();
        let all_edges = self.store.edges_snapshot()?;
        for (id, edge) in all_edges {
            let legacy_key = format!("{}::{}", edge.edge_type, edge.target);
            let key = if id == edge_id(&edge.source, &edge.edge_type, &edge.target) { legacy_key } else { id.clone() };
            let (ownership, data) = decode_ownership(edge.data);
            self.edges.entry(edge.source.clone()).or_default().insert(
                key,
                EdgeEntry { id, revision: edge.revision, target: edge.target.clone(), edge_type: edge.edge_type, data, ownership },
            );
            self.node_to_edge.entry(edge.target).or_default().insert(edge.source);
        }
        Ok(())
    }

    /// Flush dirty node/edge/vector state to the `Store`. Synchronous —
    /// unlike `PolyGraph.flush`, there is no debounce timer; callers decide
    /// their own batching/threading policy (e.g. a periodic task calling this).
    ///
    /// Mirrors `PolyGraph.flushPending`: drain the dirty/removed sets, build
    /// a single `ChangeBatch`, and apply it. `Store::apply` only mutates its
    /// own state after the WAL append succeeds, so on error nothing has been
    /// persisted — the drained ids are restored so a later `flush()` retries
    /// them, matching the TS `catch` block.
    pub fn flush(&mut self) -> Result<()> {
        let dirty_node_ids: Vec<String> = self.dirty_nodes.drain().collect();
        let dirty_edge_ids: Vec<String> = self.dirty_edges.drain().collect();
        let dirty_vector_ids: Vec<String> = self.dirty_vectors.drain().collect();
        let removed_node_ids: Vec<String> = self.removed_node_ids.drain().collect();
        let removed_edge_ids: Vec<String> = self.removed_edge_ids.drain().collect();
        let removed_vector_ids: Vec<String> = self.removed_vector_ids.drain().collect();

        // A dirty node can be missing from `self.nodes` because it was
        // evicted from the hot cache before being flushed — fall back to its
        // `evicted_dirty_nodes` snapshot. Track which ids were drawn from
        // there so a failed `apply` can restore them (mirrors
        // `flushPending`'s `evictedSnapshots`/catch-block restore).
        let mut evicted_used: Vec<(String, Node)> = Vec::new();
        let put_nodes: Vec<Node> = dirty_node_ids
            .iter()
            .filter_map(|id| {
                if let Some(node) = self.nodes.get(id) {
                    Some(node.clone())
                } else if let Some(node) = self.evicted_dirty_nodes.remove(id) {
                    evicted_used.push((id.clone(), node.clone()));
                    Some(node)
                } else {
                    None
                }
            })
            .collect();

        let put_edges: Vec<Edge> = dirty_edge_ids
            .iter()
            .filter_map(|id| {
                let (source, entry) = self.edges.iter().find_map(|(source, edges)| {
                    edges.values().find(|entry| entry.id == *id).map(|entry| (source, entry))
                })?;
                Some(Edge {
                    id: id.clone(),
                    source: source.clone(),
                    target: entry.target.clone(),
                    edge_type: entry.edge_type.clone(),
                    data: encode_ownership(entry.data.clone(), entry.ownership),
                    created_at: now_millis(),
                    revision: entry.revision,
                })
            })
            .collect();

        let put_vectors: Vec<polypack_core::VectorEntry> = dirty_vector_ids
            .iter()
            .filter_map(|id| {
                self.hnsw
                    .get(id)
                    .map(|v| polypack_core::VectorEntry { id: id.clone(), vector: v.to_vec() })
            })
            .collect();

        let delete_vector_ids: Vec<String> = removed_node_ids
            .iter()
            .chain(removed_vector_ids.iter())
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let changes = ChangeBatch {
            put_nodes,
            delete_node_ids: removed_node_ids.clone(),
            put_edges,
            delete_edge_ids: removed_edge_ids.clone(),
            put_vectors,
            delete_vector_ids,
        };

        if let Err(err) = self.store.apply_with_identity(
            &changes,
            self.current_operation_id.as_deref(),
            self.current_transaction_id.as_deref(),
            None,
            None,
            None,
        ) {
            for id in dirty_node_ids {
                if !removed_node_ids.contains(&id) {
                    self.dirty_nodes.insert(id);
                }
            }
            for id in dirty_edge_ids {
                if !removed_edge_ids.contains(&id) {
                    self.dirty_edges.insert(id);
                }
            }
            for id in dirty_vector_ids {
                if !removed_node_ids.contains(&id) {
                    self.dirty_vectors.insert(id);
                }
            }
            for id in removed_node_ids {
                self.removed_node_ids.insert(id);
            }
            for id in removed_edge_ids {
                self.removed_edge_ids.insert(id);
            }
            for id in removed_vector_ids {
                if !self.hnsw.has(&id) {
                    self.removed_vector_ids.insert(id);
                }
            }
            for (id, node) in evicted_used {
                self.evicted_dirty_nodes.insert(id, node);
            }
            return Err(err);
        }

        self.evict_oldest_if_over_cap();
        Ok(())
    }

    /// True if there is anything for `flush` to do, mirrors
    /// `PolyGraph.hasPendingPersistence`.
    pub fn has_pending_persistence(&self) -> bool {
        !self.dirty_nodes.is_empty()
            || !self.dirty_edges.is_empty()
            || !self.dirty_vectors.is_empty()
            || !self.removed_node_ids.is_empty()
            || !self.removed_edge_ids.is_empty()
            || !self.removed_vector_ids.is_empty()
    }

    /// Flush pending mutations and force a WAL-to-snapshot checkpoint.
    pub fn checkpoint(&mut self) -> Result<()> {
        self.flush()?;
        self.store.compact()
    }

    /// Create a consistent backup of the graph's durable store.
    pub fn backup(&mut self, destination: &mut dyn Storage) -> Result<()> {
        self.flush()?;
        self.store.backup(destination)
    }

    /// Flush pending mutations and close the underlying `Store`.
    pub fn close(&mut self) -> Result<()> {
        self.flush()?;
        self.store.close()
    }

    /// Write the complete currently loaded graph to the `Store`, without
    /// touching dirty-tracking state. Mirrors `PolyGraph.save`.
    pub fn save(&mut self) -> Result<()> {
        let put_nodes: Vec<Node> = self.nodes.values().cloned().collect();

        let mut put_edges = Vec::new();
        for (source, inner) in &self.edges {
            for entry in inner.values() {
                put_edges.push(Edge {
                    id: entry.id.clone(),
                    source: source.clone(),
                    target: entry.target.clone(),
                    edge_type: entry.edge_type.clone(),
                    data: encode_ownership(entry.data.clone(), entry.ownership),
                    created_at: now_millis(),
                    revision: entry.revision,
                });
            }
        }

        let put_vectors: Vec<polypack_core::VectorEntry> = self
            .hnsw
            .nodes()
            .iter()
            .map(|(id, vector)| polypack_core::VectorEntry { id: id.clone(), vector: vector.clone() })
            .collect();

        self.store.apply(&ChangeBatch {
            put_nodes,
            delete_node_ids: Vec::new(),
            put_edges,
            delete_edge_ids: Vec::new(),
            put_vectors,
            delete_vector_ids: Vec::new(),
        })
    }

    /// Clear in-memory state only — does not flush pending mutations or
    /// touch the underlying `Store`'s persisted contents. Mirrors
    /// `PolyGraph.clear`.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.hnsw.clear();
        self.hot_cache_order = LruList::new();
        self.node_to_edge.clear();
        self.by_type.clear();
        // Secondary buckets represent the complete persisted store view, not
        // only the hot cache. Keep them available for persisted queries after
        // clearing the in-memory working set.
        self.evicted_dirty_nodes.clear();
        self.dirty_edges.clear();
        self.dirty_vectors.clear();
        self.dirty_nodes.clear();
        self.removed_node_ids.clear();
        self.removed_edge_ids.clear();
        self.removed_vector_ids.clear();
        self.warmed = false;
    }

    /// Flush pending mutations, clear in-memory state, then close the
    /// `Store`. Mirrors `PolyGraph.dispose`.
    pub fn dispose(&mut self) -> Result<()> {
        self.flush()?;
        self.clear();
        self.store.close()
    }

    /// Trim the hot working set down to `max_nodes` by removing the oldest
    /// nodes (by `inserted_at`) via `remove_node` — cascading through
    /// 'owned' edges the same as any other removal. Mirrors `PolyGraph.prune`.
    pub fn prune(&mut self, max_nodes: usize) -> Result<()> {
        if self.nodes.len() <= max_nodes {
            return Ok(());
        }
        let mut by_age: Vec<(String, i64)> =
            self.nodes.iter().map(|(id, node)| (id.clone(), node.inserted_at)).collect();
        by_age.sort_by_key(|(_, inserted_at)| *inserted_at);
        let excess = self.nodes.len() - max_nodes;
        for (id, _) in by_age.into_iter().take(excess) {
            self.remove_node(&id)?;
        }
        Ok(())
    }

    // ── node CRUD ──

    /// Insert or replace a node. Replacement re-indexes by type and
    /// vector. Mirrors `PolyGraph.addNode` — validation matches
    /// `prepareNode`'s checks (`validate_node` covers id/type/timestamp/
    /// vector-finiteness).
    pub fn add_node(&mut self, node: Node) -> Result<()> {
        self.reserve_transaction_mutations(1)?;
        self.validate_node_resource_limits(&node)?;
        validate_node(&node)?;
        self.validate_node_schema(&node)?;
        self.validate_node_indexes(&node, Some(&node.id))?;
        self.insert_node(node);
        Ok(())
    }

    /// Add several nodes in one call. All nodes are validated before any are
    /// inserted (an invalid entry inserts nothing), and change-event
    /// notifications for the whole batch are coalesced via
    /// `start_batch`/`end_batch`. Matches `PolyGraph.addNodes`.
    pub fn add_nodes(&mut self, nodes: Vec<Node>) -> Result<()> {
        if let Some(limit) = self.config.resource_limits.max_batch_size {
            if nodes.len() > limit {
                return Err(PolypackError::ResourceLimit { name: "maxBatchSize".into(), limit });
            }
        }
        self.reserve_transaction_mutations(nodes.len())?;
        for n in &nodes {
            self.validate_node_resource_limits(n)?;
            validate_node(n)?;
            self.validate_node_schema(n)?;
        }
        self.start_batch();
        for n in nodes {
            self.insert_node(n);
        }
        self.end_batch()
    }

    /// Embed `text` with the configured provider (see `GraphConfig::embedding`),
    /// validated by `create_embedding`. Mirrors `PolyGraph.embed`.
    pub fn embed(&self, text: &str) -> Result<Vec<f64>> {
        create_embedding(self.config.embedding.as_ref(), text)
    }

    /// Embed `text` with the configured provider and add the resulting node.
    /// Any `vector` already set on `node` is overwritten — Rust has no
    /// `Omit<PolyNode, 'vector'>` equivalent, so this takes a full `Node` and
    /// ignores its vector field rather than a vector-less variant. Mirrors
    /// `PolyGraph.addNodeWithEmbedding`.
    pub fn add_node_with_embedding(&mut self, mut node: Node, text: &str) -> Result<()> {
        node.vector = Some(self.embed(text)?);
        self.add_node(node)
    }

    /// Detached view of a currently loaded node. Does not restore evicted
    /// nodes — see `get_node_safe`.
    pub fn get_node(&mut self, id: &str) -> Option<&Node> {
        if self.nodes.contains_key(id) {
            self.touch_hot_cache(id);
        }
        self.nodes.get(id)
    }

    /// Like `get_node`, but restores the node from the `Store` first if it
    /// was evicted from the hot cache — checking `evicted_dirty_nodes` first
    /// for edits that hadn't been flushed yet, falling back to `Store::get_node`
    /// otherwise. Mirrors `PolyGraph.getNodeSafe`, simplified: this crate has
    /// no data-transform/sidecar layer, so restoration doesn't need one.
    pub fn get_node_safe(&mut self, id: &str) -> Result<Option<&Node>> {
        if self.removed_node_ids.contains(id) {
            return Ok(None);
        }
        if self.nodes.contains_key(id) {
            self.touch_hot_cache(id);
            return Ok(self.nodes.get(id));
        }
        let restored = match self.evicted_dirty_nodes.remove(id) {
            Some(node) => node,
            None => match self.store.get_node(id)? {
                Some(node) => node,
                None => return Ok(None),
            },
        };
        let node_type = restored.node_type.clone();
        let vector = restored.vector.clone();
        self.nodes.insert(id.to_string(), restored);
        self.index_node(id, &node_type);
        self.touch_hot_cache(id);
        if let Some(vector) = vector {
            // Hydration of already-persisted data — no `dirty_vectors`
            // marking, matching `VectorIndex.hydrate` vs `.add`.
            self.hnsw.add(id, &vector)?;
        }
        Ok(self.nodes.get(id))
    }

    /// Shallow-merge `data` into a loaded node and optionally replace its
    /// vector or durable activation. A no-op (returns `None`) if the node
    /// isn't loaded — see `update_node_safe`. Mirrors `PolyGraph.updateNode`;
    /// the data-transform serialize/deserialize hooks aren't part of this
    /// crate, so `data` is merged as-is.
    pub fn update_node(
        &mut self,
        id: &str,
        data: serde_json::Map<String, serde_json::Value>,
        vector: Option<Vec<f64>>,
        activation: Option<NodeActivation>,
    ) -> Result<Option<&Node>> {
        self.reserve_transaction_mutations(1)?;
        if !self.nodes.contains_key(id) {
            return Ok(None);
        }
        if let Some(vector) = &vector {
            if !vector.iter().all(|x| x.is_finite()) {
                return Err(PolypackError::InvalidArgument("vector must contain finite values".into()));
            }
        }
        if let Some(activation) = &activation {
            validate_activation(activation)?;
        }

        let node_type = {
            let mut candidate = self.nodes.get(id).cloned().expect("node checked above");
            candidate.data.extend(data.clone());
            if let Some(vector) = &vector {
                candidate.vector = Some(vector.clone());
            }
            if activation.is_some() {
                candidate.activation = activation.clone();
            }
            candidate.updated_at = now_millis();
            candidate.revision = candidate.revision.saturating_add(1);
            self.validate_node_resource_limits(&candidate)?;
            self.validate_node_schema(&candidate)?;
            self.validate_node_indexes(&candidate, Some(id))?;
            self.unindex_node(id);
            let node = self.nodes.get_mut(id).unwrap();
            node.data.extend(data);
            if let Some(vector) = &vector {
                node.vector = Some(vector.clone());
            }
            if activation.is_some() {
                node.activation = activation.clone();
            }
            node.updated_at = now_millis();
            node.revision = node.revision.saturating_add(1);
            node.node_type.clone()
        };
        self.index_node(id, &node_type);

        if let Some(vector) = vector {
            self.removed_vector_ids.remove(id);
            self.hnsw.add(id, &vector)?;
            self.dirty_vectors.insert(id.to_string());
        }

        self.touch_hot_cache(id);
        self.mark_dirty(id);
        self.emit(GraphChangeEvent::NodeUpdated { node_id: id.to_string(), node_type });
        Ok(self.nodes.get(id))
    }

    /// Update a node only when its current revision matches `expected_revision`.
    /// The revision check happens before validation or mutation.
    pub fn update_node_if_revision(
        &mut self,
        id: &str,
        expected_revision: u64,
        data: serde_json::Map<String, serde_json::Value>,
        vector: Option<Vec<f64>>,
        activation: Option<NodeActivation>,
    ) -> Result<Option<&Node>> {
        let Some(node) = self.nodes.get(id) else { return Ok(None) };
        if node.revision != expected_revision {
            return Err(PolypackError::Conflict {
                id: id.to_string(),
                expected: expected_revision,
                actual: node.revision,
            });
        }
        self.update_node(id, data, vector, activation)
    }

    /// Apply dotted-path `set`, `unset`, and numeric `increment` operations to
    /// node data. All operations are evaluated against a copy before the node
    /// is mutated, and an optional revision check is performed first.
    pub fn patch_node(
        &mut self,
        id: &str,
        set: serde_json::Map<String, serde_json::Value>,
        unset: Vec<String>,
        increment: serde_json::Map<String, serde_json::Value>,
        compare_and_set: serde_json::Map<String, serde_json::Value>,
        expected_revision: Option<u64>,
    ) -> Result<Option<&Node>> {
        self.reserve_transaction_mutations(1)?;
        let Some(existing) = self.nodes.get(id) else { return Ok(None) };
        if let Some(expected) = expected_revision {
            if existing.revision != expected {
                return Err(PolypackError::Conflict {
                    id: id.to_string(),
                    expected,
                    actual: existing.revision,
                });
            }
        }

        let mut candidate = serde_json::Value::Object(existing.data.clone());
        for (path, operation) in compare_and_set {
            let operation = operation.as_object().ok_or_else(|| PolypackError::InvalidArgument("compare-and-set entries must be objects".into()))?;
            let expected = operation.get("expected").ok_or_else(|| PolypackError::InvalidArgument("compare-and-set entries require expected".into()))?;
            let value = operation.get("value").ok_or_else(|| PolypackError::InvalidArgument("compare-and-set entries require value".into()))?;
            let parts = patch_parts(&path)?;
            if get_data_path(&candidate, &parts) != Some(expected) {
                return Err(PolypackError::Conflict {
                    id: id.to_string(),
                    expected: expected_revision.unwrap_or(existing.revision),
                    actual: existing.revision,
                });
            }
            set_data_path(&mut candidate, &parts, value.clone())?;
        }
        for (path, value) in set {
            let parts = patch_parts(&path)?;
            set_data_path(&mut candidate, &parts, value)?;
        }
        for path in unset {
            let parts = patch_parts(&path)?;
            unset_data_path(&mut candidate, &parts);
        }
        for (path, delta) in increment {
            let parts = patch_parts(&path)?;
            let delta = delta
                .as_f64()
                .ok_or_else(|| PolypackError::InvalidArgument("increment values must be numeric".into()))?;
            let current = match get_data_path(&candidate, &parts) {
                None => 0.0,
                Some(value) => value
                    .as_f64()
                    .ok_or_else(|| PolypackError::InvalidArgument("increment targets must be numeric".into()))?,
            };
            if !current.is_finite() || !delta.is_finite() {
                return Err(PolypackError::InvalidArgument("increment values must be finite".into()));
            }
            set_data_path(&mut candidate, &parts, serde_json::json!(current + delta))?;
        }

        let candidate_node = Node {
            id: existing.id.clone(),
            node_type: existing.node_type.clone(),
            data: candidate.as_object().cloned().expect("patch root is an object"),
            vector: existing.vector.clone(),
            inserted_at: existing.inserted_at,
            updated_at: now_millis(),
            revision: existing.revision.saturating_add(1),
            activation: existing.activation.clone(),
        };
        self.validate_node_resource_limits(&candidate_node)?;
        self.validate_node_schema(&candidate_node)?;
        self.validate_node_indexes(&candidate_node, Some(id))?;

        let node_type = {
            self.unindex_node(id);
            let node = self.nodes.get_mut(id).expect("node checked above");
            node.data = candidate.as_object().cloned().expect("patch root is an object");
            node.updated_at = now_millis();
            node.revision = node.revision.saturating_add(1);
            node.node_type.clone()
        };
        self.index_node(id, &node_type);
        self.touch_hot_cache(id);
        self.mark_dirty(id);
        self.emit(GraphChangeEvent::NodeUpdated { node_id: id.to_string(), node_type });
        Ok(self.nodes.get(id))
    }

    /// Restore `id` from the `Store` if necessary, then update it. Mirrors
    /// `PolyGraph.updateNodeSafe`.
    pub fn update_node_safe(
        &mut self,
        id: &str,
        data: serde_json::Map<String, serde_json::Value>,
        vector: Option<Vec<f64>>,
        activation: Option<NodeActivation>,
    ) -> Result<Option<&Node>> {
        if self.get_node_safe(id)?.is_none() {
            return Ok(None);
        }
        self.update_node(id, data, vector, activation)
    }

    /// Embed `text` and update a loaded node's data and vector together.
    /// Mirrors `PolyGraph.updateNodeWithEmbedding`.
    pub fn update_node_with_embedding(
        &mut self,
        id: &str,
        data: serde_json::Map<String, serde_json::Value>,
        text: &str,
    ) -> Result<Option<&Node>> {
        let vector = self.embed(text)?;
        self.update_node(id, data, Some(vector), None)
    }

    /// Embed `text`, restoring an evicted node before updating when
    /// necessary. Mirrors `PolyGraph.updateNodeSafeWithEmbedding`.
    pub fn update_node_safe_with_embedding(
        &mut self,
        id: &str,
        data: serde_json::Map<String, serde_json::Value>,
        text: &str,
    ) -> Result<Option<&Node>> {
        let vector = self.embed(text)?;
        self.update_node_safe(id, data, Some(vector), None)
    }

    /// Remove a node's vector while keeping the node and its data. A no-op
    /// (returns `None`) if the node isn't loaded. Mirrors
    /// `PolyGraph.removeNodeVector`.
    pub fn remove_node_vector(&mut self, id: &str) -> Option<&Node> {
        if !self.nodes.contains_key(id) {
            return None;
        }
        self.hnsw.remove(id);
        self.dirty_vectors.remove(id);
        self.removed_vector_ids.insert(id.to_string());

        let node_type = {
            let node = self.nodes.get_mut(id).unwrap();
            node.vector = None;
            node.updated_at = now_millis();
            node.node_type.clone()
        };

        self.touch_hot_cache(id);
        self.mark_dirty(id);
        self.emit(GraphChangeEvent::NodeUpdated { node_id: id.to_string(), node_type });
        self.nodes.get(id)
    }

    /// Restore `id` from the `Store` if necessary, then remove its vector.
    /// `None` if no such node exists anywhere. Mirrors
    /// `PolyGraph.removeNodeVectorSafe`.
    pub fn remove_node_vector_safe(&mut self, id: &str) -> Result<Option<&Node>> {
        if self.get_node_safe(id)?.is_none() {
            return Ok(None);
        }
        Ok(self.remove_node_vector(id))
    }

    /// Remove `id` and cascade through 'owned' edges. Targets of 'owned'
    /// edges are also removed unless they have another 'owned' source
    /// keeping them alive. Cyclic owned edges (A -> B -> A) are detected and
    /// each node is only removed once. Mirrors `PolyGraph.removeNode` — note
    /// that, like the TS version, this does *not* fire `on_orphan` for
    /// 'shared' targets; only `remove_edges` does that.
    pub fn remove_node(&mut self, id: &str) -> Result<()> {
        self.reserve_transaction_mutations(1)?;
        let mut visited = HashSet::new();
        self.remove_node_cascade(id, &mut visited)
    }

    /// Remove a node only when its current revision matches `expected_revision`.
    pub fn remove_node_if_revision(&mut self, id: &str, expected_revision: u64) -> Result<()> {
        let Some(node) = self.nodes.get(id) else { return Ok(()) };
        if node.revision != expected_revision {
            return Err(PolypackError::Conflict {
                id: id.to_string(),
                expected: expected_revision,
                actual: node.revision,
            });
        }
        self.remove_node(id)
    }

    fn remove_node_cascade(&mut self, id: &str, visited: &mut HashSet<String>) -> Result<()> {
        if !visited.insert(id.to_string()) {
            return Ok(());
        }

        let Some(node) = self.nodes.get(id) else { return Ok(()) };
        let node_type = node.node_type.clone();

        // Snapshot outgoing edges before cleanup — cascading recursion below
        // mutates `self.edges`, so iterating it live would be unsound.
        let outgoing: Vec<EdgeEntry> =
            self.edges.get(id).map(|m| m.values().cloned().collect()).unwrap_or_default();
        for edge in &outgoing {
            if edge.ownership == EdgeOwnership::Owned && !self.has_other_owned_source(&edge.target, id) {
                self.remove_node_cascade(&edge.target, visited)?;
            }
        }

        self.cleanup_node_edges(id);
        self.unindex_node(id);
        self.nodes.remove(id);
        self.hnsw.remove(id);
        self.dirty_vectors.remove(id);
        self.removed_vector_ids.remove(id);

        self.dirty_nodes.remove(id);
        self.removed_node_ids.insert(id.to_string());
        self.hot_cache_order.remove(id);
        self.emit(GraphChangeEvent::NodeRemoved { node_id: id.to_string(), node_type });

        Ok(())
    }

    /// Detach `id` from the edge index: drop its incoming edges (from every
    /// source) and its outgoing edges, recording each as removed. Mirrors
    /// `PolyGraph.cleanupNodeEdges`.
    fn cleanup_node_edges(&mut self, id: &str) {
        let incoming_sources: Vec<String> =
            self.node_to_edge.get(id).map(|s| s.iter().cloned().collect()).unwrap_or_default();
        for source in incoming_sources {
            let removed: Vec<(String, EdgeEntry)> = self
                .edges
                .get(&source)
                .map(|edges| {
                    edges
                        .iter()
                        .filter(|(_, e)| e.target == id)
                        .map(|(k, e)| (k.clone(), e.clone()))
                        .collect()
                })
                .unwrap_or_default();
            if let Some(source_edges) = self.edges.get_mut(&source) {
                for (key, _) in &removed {
                    source_edges.remove(key);
                }
                if source_edges.is_empty() {
                    self.edges.remove(&source);
                }
            }
            for (_, edge) in &removed {
                self.record_removed_edge(&source, edge);
            }
        }

        if let Some(source_edges) = self.edges.remove(id) {
            for edge in source_edges.values() {
                if let Some(set) = self.node_to_edge.get_mut(&edge.target) {
                    set.remove(id);
                }
                self.record_removed_edge(id, edge);
            }
        }
        self.node_to_edge.remove(id);
    }

    /// Mirrors `PolyGraph.recordRemovedEdge`.
    fn record_removed_edge(&mut self, source: &str, edge: &EdgeEntry) {
        let id = edge.id.clone();
        self.dirty_edges.remove(&id);
        self.removed_edge_ids.insert(id);
        self.emit(GraphChangeEvent::EdgeRemoved {
            edge_type: edge.edge_type.clone(),
            source: source.to_string(),
            target: edge.target.clone(),
        });
    }

    /// Restore `id` from the `Store` if necessary, then remove it, cascading
    /// through 'owned' edges the same way as `remove_node` (targets are
    /// restored-then-removed recursively; another owning source keeps a
    /// target alive; cycles are cut by `visited`). Returns `false` if no
    /// such node exists anywhere. Mirrors
    /// `PolyGraph.removeNodeSafe`/`removeNodeSafeRecursive`.
    ///
    /// Existing persisted graphs must be warmed first so edge ownership is
    /// available in the in-memory edge index — restoring a node via
    /// `get_node_safe` does not reconstruct its edges.
    pub fn remove_node_safe(&mut self, id: &str) -> Result<bool> {
        let mut visited = HashSet::new();
        self.remove_node_safe_recursive(id, &mut visited)
    }

    fn remove_node_safe_recursive(&mut self, id: &str, visited: &mut HashSet<String>) -> Result<bool> {
        if !visited.insert(id.to_string()) {
            return Ok(false);
        }

        if self.get_node_safe(id)?.is_none() {
            return Ok(false);
        }

        let outgoing: Vec<EdgeEntry> =
            self.edges.get(id).map(|m| m.values().cloned().collect()).unwrap_or_default();
        for edge in &outgoing {
            if edge.ownership == EdgeOwnership::Owned && !self.has_other_owned_source(&edge.target, id) {
                self.remove_node_safe_recursive(&edge.target, visited)?;
            }
        }

        if self.removed_node_ids.contains(id) {
            return Ok(true);
        }

        // Recursive restoration can evict this node again when the working
        // set is very small, so restore it once more before using the
        // synchronous remover.
        if !self.nodes.contains_key(id) {
            self.get_node_safe(id)?;
        }
        if !self.nodes.contains_key(id) {
            return Ok(false);
        }
        self.remove_node(id)?;
        Ok(true)
    }

    // ── edge CRUD ──

    /// Add one directed edge. A no-op if an edge with the same
    /// source/type/target already exists — edges are unique per triple, and
    /// (matching `PolyGraph.addEdge`) the existing entry's `data`/`ownership`
    /// are left untouched rather than overwritten.
    pub fn add_edge(
        &mut self,
        source: &str,
        edge_type: &str,
        target: &str,
        data: Option<serde_json::Map<String, serde_json::Value>>,
        ownership: EdgeOwnership,
    ) -> Result<()> {
        self.reserve_transaction_mutations(1)?;
        if source.is_empty() || edge_type.is_empty() || target.is_empty() {
            return Err(PolypackError::InvalidArgument(
                "edge source, type, and target must not be empty".into(),
            ));
        }
        self.validate_edge_schema(source, edge_type, target, None)?;
        self.validate_edge_data_schema(edge_type, data.as_ref())?;
        let id = edge_id(source, edge_type, target);
        self.removed_edge_ids.remove(&id);
        let inner = format!("{edge_type}::{target}");
        let source_edges = self.edges.entry(source.to_string()).or_default();
        if source_edges.contains_key(&inner) {
            return Ok(());
        }
        source_edges.insert(
            inner,
                EdgeEntry { id: id.clone(), revision: 0, target: target.to_string(), edge_type: edge_type.to_string(), data, ownership },
        );
        self.node_to_edge.entry(target.to_string()).or_default().insert(source.to_string());
        self.dirty_edges.insert(id.clone());
        self.emit(GraphChangeEvent::EdgeAdded {
            edge_id: id,
            edge_type: edge_type.to_string(),
            source: source.to_string(),
            target: target.to_string(),
        });
        Ok(())
    }

    /// Add a directed edge with an explicit independent identity. Unlike
    /// [`add_edge`], this permits parallel edges with the same
    /// source/type/target triple.
    pub fn add_edge_with_id(
        &mut self,
        id: &str,
        source: &str,
        edge_type: &str,
        target: &str,
        data: Option<serde_json::Map<String, serde_json::Value>>,
        ownership: EdgeOwnership,
    ) -> Result<()> {
        self.reserve_transaction_mutations(1)?;
        if id.is_empty() || source.is_empty() || edge_type.is_empty() || target.is_empty() {
            return Err(PolypackError::InvalidArgument("edge id, source, type, and target must not be empty".into()));
        }
        self.validate_edge_schema(source, edge_type, target, None)?;
        self.validate_edge_data_schema(edge_type, data.as_ref())?;
        if self.edges.values().any(|edges| edges.values().any(|edge| edge.id == id)) {
            return Ok(());
        }
        self.removed_edge_ids.remove(id);
        let legacy_id = edge_id(source, edge_type, target);
        let key = if id == legacy_id { format!("{edge_type}::{target}") } else { id.to_string() };
        self.edges.entry(source.to_string()).or_default().insert(
            key,
            EdgeEntry {
                id: id.to_string(),
                revision: 0,
                target: target.to_string(),
                edge_type: edge_type.to_string(),
                data,
                ownership,
            },
        );
        self.node_to_edge.entry(target.to_string()).or_default().insert(source.to_string());
        self.dirty_edges.insert(id.to_string());
        self.emit(GraphChangeEvent::EdgeAdded {
            edge_id: id.to_string(),
            edge_type: edge_type.to_string(),
            source: source.to_string(),
            target: target.to_string(),
        });
        Ok(())
    }

    /// Update edge data and/or ownership only when the current revision
    /// matches `expected_revision`.
    pub fn update_edge_if_revision(
        &mut self,
        id: &str,
        data: Option<serde_json::Map<String, serde_json::Value>>,
        ownership: Option<EdgeOwnership>,
        expected_revision: u64,
    ) -> Result<bool> {
        self.reserve_transaction_mutations(1)?;
        let Some((source, key)) = self.edges.iter().find_map(|(source, edges)| {
            edges.iter().find_map(|(key, edge)| (edge.id == id).then_some((source.clone(), key.clone())))
        }) else {
            return Ok(false);
        };
        let edge = self.edges.get_mut(&source).and_then(|edges| edges.get_mut(&key)).expect("edge location checked");
        if edge.revision != expected_revision {
            return Err(PolypackError::Conflict {
                id: id.to_string(),
                expected: expected_revision,
                actual: edge.revision,
            });
        }
        if data.is_some() {
            edge.data = data;
        }
        if let Some(ownership) = ownership {
            edge.ownership = ownership;
        }
        edge.revision = edge.revision.saturating_add(1);
        let edge_type = edge.edge_type.clone();
        let target = edge.target.clone();
        self.dirty_edges.insert(id.to_string());
        self.emit(GraphChangeEvent::EdgeUpdated { edge_id: id.to_string(), edge_type, source, target });
        Ok(true)
    }

    /// Remove exactly one edge by independent ID when its revision matches.
    pub fn remove_edge_if_revision(&mut self, id: &str, expected_revision: u64) -> Result<bool> {
        self.reserve_transaction_mutations(1)?;
        let Some((source, key, edge)) = self.edges.iter().find_map(|(source, edges)| {
            edges.iter().find_map(|(key, edge)| (edge.id == id).then_some((source.clone(), key.clone(), edge.clone())))
        }) else {
            return Ok(false);
        };
        if edge.revision != expected_revision {
            return Err(PolypackError::Conflict {
                id: id.to_string(),
                expected: expected_revision,
                actual: edge.revision,
            });
        }
        if edge.ownership == EdgeOwnership::Owned && !self.has_other_owned_source(&edge.target, &source) {
            self.remove_node(&edge.target)?;
            return Ok(true);
        }
        if let Some(source_edges) = self.edges.get_mut(&source) {
            source_edges.remove(&key);
            if source_edges.is_empty() {
                self.edges.remove(&source);
            }
        }
        let still_connected = self
            .edges
            .get(&source)
            .is_some_and(|edges| edges.values().any(|other| other.target == edge.target));
        if let Some(sources) = self.node_to_edge.get_mut(&edge.target) {
            if !still_connected {
                sources.remove(&source);
            }
            if sources.is_empty() {
                self.node_to_edge.remove(&edge.target);
            }
        }
        self.dirty_edges.remove(id);
        self.removed_edge_ids.insert(id.to_string());
        self.emit(GraphChangeEvent::EdgeRemoved {
            edge_type: edge.edge_type.clone(),
            source: source.clone(),
            target: edge.target.clone(),
        });
        if edge.ownership == EdgeOwnership::Shared && !self.has_other_incoming(&edge.target, &source) {
            if let Some(cb) = self.on_orphan.as_mut() {
                cb(&edge.target);
            }
        }
        Ok(true)
    }

    /// Direct access to the vector index. Mutating it directly (rather than
    /// through `add_node`/`update_node`, which handle this themselves)
    /// bypasses dirty-tracking — call `mark_vector_dirty` afterwards so the
    /// change actually gets flushed. Mirrors the public `vectors` field on
    /// `PolyGraph`.
    pub fn vectors(&self) -> &HnswIndex {
        &self.hnsw
    }

    /// Mutable access to the vector index — see `vectors`'s docs on
    /// dirty-tracking.
    pub fn vectors_mut(&mut self) -> &mut HnswIndex {
        &mut self.hnsw
    }

    /// Mark `id`'s vector for persistence after it was mutated directly
    /// through `vectors_mut` (e.g. `graph.vectors_mut().add(...)`) rather
    /// than via `add_node`/`update_node`, which mark it themselves. A no-op
    /// if `id` isn't currently in the vector index. Mirrors
    /// `PolyGraph.markVectorDirty`.
    pub fn mark_vector_dirty(&mut self, id: &str) {
        if self.hnsw.has(id) {
            self.dirty_vectors.insert(id.to_string());
        }
    }

    /// `source`'s outgoing edges, optionally filtered by type. Mirrors
    /// `PolyGraph.getEdges` — returns borrowed views rather than detached
    /// clones (no `cloneData` needed; the borrow checker already prevents
    /// concurrent mutation).
    pub fn get_edges(&self, source: &str, edge_type: Option<&str>) -> Vec<&EdgeEntry> {
        let Some(edges) = self.edges.get(source) else { return Vec::new() };
        edges
            .values()
            .filter(|e| edge_type.is_none_or(|t| e.edge_type == t))
            .collect()
    }

    /// IDs of nodes reachable from `source` via outgoing edges of `edge_type`.
    pub fn get_edge_targets(&self, source: &str, edge_type: &str) -> Vec<&str> {
        let Some(edges) = self.edges.get(source) else { return Vec::new() };
        edges
            .values()
            .filter(|e| e.edge_type == edge_type)
            .map(|e| e.target.as_str())
            .collect()
    }

    /// IDs of nodes with an outgoing edge of `edge_type` into `target`.
    pub fn get_edge_sources(&self, target: &str, edge_type: &str) -> Vec<&str> {
        let Some(sources) = self.node_to_edge.get(target) else { return Vec::new() };
        sources
            .iter()
            .filter(|source| {
                self.edges.get(source.as_str()).is_some_and(|edges| {
                    edges.values().any(|e| e.edge_type == edge_type && e.target == target)
                })
            })
            .map(|s| s.as_str())
            .collect()
    }

    /// Remove edges from `source`, optionally narrowed by `edge_type` and/or
    /// `target`. 'owned' edges cascade-delete their target (unless another
    /// source also owns it); 'shared' edges fire `on_orphan` if the target
    /// becomes disconnected. Mirrors `PolyGraph.removeEdges`.
    ///
    /// The owned-edge cascade calls `remove_node`, which handles cycles
    /// (`A -> B -> A`) safely, removing each node only once.
    pub fn remove_edges(&mut self, source: &str, edge_type: Option<&str>, target: Option<&str>) -> Result<()> {
        self.reserve_transaction_mutations(1)?;
        let Some(edges) = self.edges.get(source) else { return Ok(()) };
        let removed: Vec<EdgeEntry> = edges
            .values()
            .filter(|e| edge_type.is_none_or(|t| e.edge_type == t) && target.is_none_or(|tg| e.target == tg))
            .cloned()
            .collect();
        if removed.is_empty() {
            return Ok(());
        }

        for edge in &removed {
            if edge.ownership == EdgeOwnership::Owned && !self.has_other_owned_source(&edge.target, source) {
                self.remove_node(&edge.target)?;
            }
        }

        let remove_all = edge_type.is_none() && target.is_none();
        for edge in &removed {
            if !remove_all {
                if let Some(source_edges) = self.edges.get_mut(source) {
                    if let Some(key) = source_edges
                        .iter()
                        .find_map(|(key, current)| (current.id == edge.id).then_some(key.clone()))
                    {
                        source_edges.remove(&key);
                    }
                }
            }
            if let Some(set) = self.node_to_edge.get_mut(&edge.target) {
                set.remove(source);
            }
            let id = edge.id.clone();
            self.dirty_edges.remove(&id);
            self.removed_edge_ids.insert(id);
            self.emit(GraphChangeEvent::EdgeRemoved {
                edge_type: edge.edge_type.clone(),
                source: source.to_string(),
                target: edge.target.clone(),
            });

            if edge.ownership == EdgeOwnership::Shared && !self.has_other_incoming(&edge.target, source) {
                if let Some(cb) = self.on_orphan.as_mut() {
                    cb(&edge.target);
                }
            }
        }

        if remove_all || self.edges.get(source).is_some_and(|m| m.is_empty()) {
            self.edges.remove(source);
        }

        Ok(())
    }

    /// True if `target` has at least one incoming 'owned' edge from a source
    /// other than `exclude_source`. Mirrors `PolyGraph.hasOtherOwnedSource`.
    fn has_other_owned_source(&self, target: &str, exclude_source: &str) -> bool {
        let Some(sources) = self.node_to_edge.get(target) else { return false };
        sources.iter().any(|src| {
            src != exclude_source
                && self.edges.get(src).is_some_and(|edges| {
                    edges.values().any(|e| e.target == target && e.ownership == EdgeOwnership::Owned)
                })
        })
    }

    /// True if `target` has at least one incoming edge of any type from any
    /// source other than `exclude_source`. Mirrors `PolyGraph.hasOtherIncoming`.
    fn has_other_incoming(&self, target: &str, exclude_source: &str) -> bool {
        let Some(sources) = self.node_to_edge.get(target) else { return false };
        sources.iter().any(|src| {
            src != exclude_source
                && self.edges.get(src).is_some_and(|edges| edges.values().any(|e| e.target == target))
        })
    }

    // ── query ──

    /// Execute a raw `QueryPlan` over the current hot working set via
    /// `polypack_core::query_exec::execute`. The hot `HnswIndex` is always
    /// passed through so a plan's `similarity.engine: "hnsw"` can use it.
    ///
    /// This is a lower-level entry point than `query()` (which returns a
    /// `GraphQuery` fluent builder, mirroring `PolyGraph.query()`) — no
    /// direct TS equivalent takes a raw plan; it exists to expose
    /// `query_exec` directly for callers that already have a `QueryPlan`.
    pub fn query_plan(&self, plan: &QueryPlan) -> Result<Vec<String>> {
        core_execute(&self.snapshot(), plan, Some(&self.hnsw))
    }

    /// Aggregate a numeric field over the nodes a `QueryPlan` selects from
    /// the hot working set. See `query_plan`'s docs on scope.
    pub fn aggregate_plan(&self, plan: &QueryPlan, field: &str, op: &str) -> Result<(f64, usize)> {
        core_aggregate(&self.snapshot(), plan, field, op)
    }

    /// Start a fluent query over the current hot working set. Mirrors
    /// `PolyGraph.query`.
    pub fn query(&self) -> GraphQuery<'_> {
        GraphQuery::new(&self.nodes, &self.edges, &self.node_to_edge, &self.indexes, &self.secondary_indexes, &self.query_metrics)
    }

    /// Start a fluent query over every persisted node, without loading
    /// results into the hot working set. Mirrors `PolyGraph.queryPersisted`.
    pub fn query_persisted(&mut self) -> PersistedGraphQuery<'_> {
        PersistedGraphQuery::new(&mut self.store, &self.indexes, &self.secondary_indexes, &self.query_metrics)
    }

    /// Create a persisted query with explicit traversal and result limits.
    pub fn query_persisted_with_limits(&mut self, limits: QueryResourceLimits) -> PersistedGraphQuery<'_> {
        PersistedGraphQuery::new(&mut self.store, &self.indexes, &self.secondary_indexes, &self.query_metrics).with_resource_limits(limits)
    }

    /// Start an in-memory similarity query from text, embedded with the
    /// configured provider. Mirrors `PolyGraph.queryText`.
    pub fn query_text(&self, text: &str, threshold: f64, top_k: Option<usize>) -> Result<GraphQuery<'_>> {
        let vector = self.embed(text)?;
        Ok(self.query().similar_to(vector, threshold, top_k))
    }

    /// Start a persisted similarity query from text, embedded with the
    /// configured provider. Mirrors `PolyGraph.queryPersistedText`.
    pub fn query_persisted_text(
        &mut self,
        text: &str,
        threshold: f64,
        top_k: Option<usize>,
    ) -> Result<PersistedGraphQuery<'_>> {
        let vector = self.embed(text)?;
        Ok(self.query_persisted().similar_to(vector, threshold, top_k))
    }

    /// Quick full-text search across persisted nodes of a single type.
    /// Shorthand for
    /// `query_persisted_text(text, threshold, top_k).where_node_type(vec![node_type]).to_array()`.
    /// Mirrors `PolyGraph.searchNodes`.
    pub fn search_nodes(&mut self, text: &str, node_type: &str, threshold: f64, top_k: Option<usize>) -> Result<Vec<Node>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        self.query_persisted_text(text, threshold, top_k)?.where_node_type(vec![node_type.to_string()]).to_array()
    }

    /// Capture a detached query snapshot of the current hot node/edge working
    /// set. Later graph mutations do not affect the returned snapshot.
    /// `created_at` on reconstructed edges is a placeholder because the query
    /// executor only reads edge identity, endpoints, type, and data.
    pub fn snapshot(&self) -> GraphSnapshot {
        let nodes: Vec<Node> = self.nodes.values().cloned().collect();
        let mut edges = Vec::new();
        for (source, inner) in &self.edges {
            for entry in inner.values() {
                edges.push(Edge {
                    id: entry.id.clone(),
                    source: source.clone(),
                    target: entry.target.clone(),
                    edge_type: entry.edge_type.clone(),
                    data: entry.data.clone(),
                    created_at: 0,
                    revision: entry.revision,
                });
            }
        }
        GraphSnapshot::new(nodes, edges)
    }

    /// Approximate nearest-neighbor search via the hot `HnswIndex`.
    pub fn similar_to(&self, vector: &[f64], threshold: f64, top_k: Option<usize>) -> Result<Vec<(String, f64)>> {
        let scored = self.hnsw.query(vector, top_k.unwrap_or(10), threshold)?;
        Ok(scored.into_iter().map(|s| (s.id, s.score)).collect())
    }

    // ── traversal ──

    /// Walk the ancestor chain backwards through incoming edges of
    /// `edge_type`, following the first source at each step. Returns nodes
    /// from the root ancestor to `id` (inclusive). Detects cycles. Mirrors
    /// `PolyGraph.walkAncestors`.
    pub fn walk_ancestors(&self, id: &str, edge_type: &str) -> Vec<&Node> {
        let mut path = self.walk(id, edge_type, Self::get_edge_sources);
        path.reverse();
        path
    }

    /// Walk the descendant chain forwards through outgoing edges of
    /// `edge_type`, following the first target at each step. Returns nodes
    /// from `id` to the deepest child (inclusive). Detects cycles. Mirrors
    /// `PolyGraph.walkDescendants`.
    pub fn walk_descendants(&self, id: &str, edge_type: &str) -> Vec<&Node> {
        self.walk(id, edge_type, Self::get_edge_targets)
    }

    /// Shared traversal loop for `walk_ancestors`/`walk_descendants`:
    /// follows the first neighbor reported by `next` (either
    /// `get_edge_sources` or `get_edge_targets`) at each step, stopping on a
    /// missing node or a revisited one.
    fn walk<'a>(
        &'a self,
        id: &str,
        edge_type: &str,
        next: impl Fn(&'a Self, &str, &str) -> Vec<&'a str>,
    ) -> Vec<&'a Node> {
        let mut path = Vec::new();
        let mut seen = HashSet::new();
        let mut current = Some(id.to_string());
        while let Some(cur) = current {
            if !seen.insert(cur.clone()) {
                break;
            }
            let Some(node) = self.nodes.get(&cur) else { break };
            path.push(node);
            current = next(self, &cur, edge_type).first().map(|s| s.to_string());
        }
        path
    }

    // ── convenience ──

    pub fn where_type(&self, node_type: &str) -> Vec<&Node> {
        self.by_type
            .get(node_type)
            .into_iter()
            .flatten()
            .filter_map(|id| self.nodes.get(id))
            .collect()
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Number of nodes in the loaded working set. Alias for `size`. Mirrors
    /// `PolyGraph.loadedSize`.
    pub fn loaded_size(&self) -> usize {
        self.nodes.len()
    }

    pub fn has_loaded_node(&self, id: &str) -> bool {
        self.nodes.contains_key(id)
    }

    /// Number of nodes currently stored by the `Store`, independent of what's
    /// in the hot working set. Mirrors `PolyGraph.persistedSize`.
    pub fn persisted_size(&mut self) -> Result<usize> {
        self.store.node_count()
    }

    // ── activation ──

    /// Current decayed activation score of a loaded node, or 0 when it has
    /// none. Decay is a pure function of elapsed time from
    /// `lastMeaningfulActivation`, so this is deterministic across replicas
    /// with the same stored state. `half_life_ms` defaults to the standard 24h
    /// score curve. Mirrors `PolyGraph.getActivation`.
    pub fn get_activation(&self, id: &str, half_life_ms: i64) -> f64 {
        match self.nodes.get(id) {
            Some(node) => activation_score_of(node, now_millis(), half_life_ms),
            None => 0.0,
        }
    }

    /// Decay-corrected view of a loaded node's durable activation, or `None`.
    /// Mirrors `PolyGraph.getActivationState`.
    pub fn get_activation_state(&self, id: &str) -> Option<NodeActivation> {
        let node = self.nodes.get(id)?;
        node.activation.as_ref().map(|a| {
            decay_activation_state(
                a,
                now_millis(),
                DEFAULT_ACTIVATION.score_half_life_ms,
                DEFAULT_ACTIVATION.importance_half_life_ms,
            )
        })
    }

    /// Apply a durable reinforcement delta to a loaded node's activation. The
    /// prior state is decay-corrected to now, `amount` is added to `score`, a
    /// fraction is folded into `importance`, the reinforcement counter
    /// increments, and the decay anchor re-sets to now. Persists and emits an
    /// `ActivationUpdated` change event. Returns the updated node or `None`
    /// when the node isn't loaded (see `reinforce_node_safe`). Mirrors
    /// `PolyGraph.reinforceNode`.
    pub fn reinforce_node(&mut self, id: &str, amount: f64, reason: Option<&str>) -> Result<Option<&Node>> {
        if !amount.is_finite() {
            return Err(PolypackError::InvalidArgument("reinforcement amount must be finite".into()));
        }
        let Some(node) = self.nodes.get_mut(id) else {
            return Ok(None);
        };
        let now = now_millis();
        node.activation = Some(reinforce_activation(node.activation.as_ref(), amount, now, &DEFAULT_ACTIVATION));
        node.updated_at = now;
        let node_type = node.node_type.clone();

        self.touch_hot_cache(id);
        self.mark_dirty(id);
        self.emit(GraphChangeEvent::ActivationUpdated {
            node_id: id.to_string(),
            node_type,
            delta: amount,
            reason: reason.map(|r| r.to_string()),
        });
        Ok(self.nodes.get(id))
    }

    /// Restore an evicted node when necessary, then reinforce it. Mirrors
    /// `PolyGraph.reinforceNodeSafe`.
    pub fn reinforce_node_safe(&mut self, id: &str, amount: f64, reason: Option<&str>) -> Result<Option<&Node>> {
        if !amount.is_finite() {
            return Err(PolypackError::InvalidArgument("reinforcement amount must be finite".into()));
        }
        if self.get_node_safe(id)?.is_none() {
            return Ok(None);
        }
        self.reinforce_node(id, amount, reason)
    }

    /// Loaded nodes with the highest current activation, descending. The
    /// working-memory primitive. Mirrors `PolyGraph.topActivated`.
    pub fn top_activated(&self, limit: usize, min_score: f64) -> Vec<&Node> {
        if limit == 0 {
            return Vec::new();
        }
        let now = now_millis();
        let mut scored: Vec<(&Node, f64)> = self
            .nodes
            .values()
            .map(|n| (n, activation_score_of(n, now, DEFAULT_ACTIVATION.score_half_life_ms)))
            .filter(|(_, score)| *score > min_score)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).map(|(n, _)| n).collect()
    }

    /// Materialize decay for every loaded node with activation: rewrite each
    /// node's stored values to their current decayed state and re-anchor to
    /// `now`. Reads already decay lazily, so this only matters for persisting
    /// fresh values (e.g. before eviction-driven lifecycle events). Mirrors
    /// `PolyGraph.decay`.
    pub fn decay(&mut self, now: i64) {
        let mut dirty: Vec<String> = Vec::new();
        for node in self.nodes.values_mut() {
            let Some(activation) = &node.activation else { continue };
            let corrected = decay_activation_state(
                activation,
                now,
                DEFAULT_ACTIVATION.score_half_life_ms,
                DEFAULT_ACTIVATION.importance_half_life_ms,
            );
            node.activation = Some(NodeActivation {
                score: corrected.score,
                importance: corrected.importance,
                reinforcement_count: activation.reinforcement_count,
                last_meaningful_activation: now,
            });
            node.updated_at = now;
            dirty.push(node.id.clone());
        }
        for id in dirty {
            self.mark_dirty(&id);
        }
    }

    // ── internal ──

    /// Reserve logical mutation capacity for the current transaction. The
    /// reservation happens before validation/mutation, matching the public
    /// TypeScript transaction API: a failed operation still consumed its
    /// attempted mutation slot, while rollback restores the prior count.
    fn reserve_transaction_mutations(&mut self, amount: usize) -> Result<()> {
        if !self.transaction_active || amount == 0 {
            return Ok(());
        }
        if let Some(limit) = self.config.resource_limits.max_batch_size {
            if self.transaction_mutation_count.saturating_add(amount) > limit {
                return Err(PolypackError::ResourceLimit { name: "maxBatchSize".into(), limit });
            }
        }
        self.transaction_mutation_count = self.transaction_mutation_count.saturating_add(amount);
        Ok(())
    }

    /// Mirrors `PolyGraph.insertNode`: assumes the node has already passed
    /// `validate_node`.
    fn insert_node(&mut self, mut node: Node) {
        let id = node.id.clone();
        let node_type = node.node_type.clone();
        let vector = node.vector.clone();

        if self.nodes.contains_key(&id) {
            if let Some(previous) = self.nodes.get(&id) {
                node.revision = previous.revision.saturating_add(1);
            }
            self.unindex_node(&id);
        }
        if vector.is_none() {
            self.hnsw.remove(&id);
            self.dirty_vectors.remove(&id);
            self.removed_vector_ids.insert(id.clone());
        }
        self.removed_node_ids.remove(&id);
        self.nodes.insert(id.clone(), node);
        self.touch_hot_cache(&id);
        self.dirty_nodes.insert(id.clone());
        self.index_node(&id, &node_type);
        if let Some(vector) = vector {
            self.removed_vector_ids.remove(&id);
            // Finiteness already checked by `validate_node`.
            self.hnsw.add(&id, &vector).expect("vector validated by validate_node");
            // Mirrors `vectors.add` triggering `VectorIndex`'s `onChange` in
            // `graph.ts` — unlike `hydrate` (used to restore already-persisted
            // vectors, see `get_node_safe`), a fresh `add` must be flushed.
            self.dirty_vectors.insert(id.clone());
        }
        self.emit(GraphChangeEvent::NodeAdded { node_id: Some(id), node_type });
    }

    /// Mirrors `PolyGraph.indexNode`'s `_byType` bookkeeping.
    fn index_node(&mut self, id: &str, node_type: &str) {
        self.by_type.entry(node_type.to_string()).or_default().insert(id.to_string());
        if let Some(node) = self.nodes.get(id).cloned() { self.add_secondary_index_entry(&node); }
    }

    /// Mirrors `PolyGraph.unindexNode`.
    fn unindex_node(&mut self, id: &str) {
        let Some(node) = self.nodes.get(id).cloned() else { return };
        self.remove_secondary_index_entry(&node);
        if let Some(set) = self.by_type.get_mut(&node.node_type) {
            set.remove(id);
            if set.is_empty() {
                let node_type = node.node_type.clone();
                self.by_type.remove(&node_type);
            }
        }
    }

    fn add_secondary_index_entry(&mut self, node: &Node) {
        for (name, definition) in &self.indexes {
            let Some(key) = indexed_value_key(node, definition) else { continue };
            self.secondary_indexes.entry(name.clone()).or_default().entry(key).or_default().insert(node.id.clone());
        }
    }

    fn remove_secondary_index_entry(&mut self, node: &Node) {
        for (name, definition) in &self.indexes {
            let Some(key) = indexed_value_key(node, definition) else { continue };
            if let Some(bucket) = self.secondary_indexes.get_mut(name).and_then(|buckets| buckets.get_mut(&key)) {
                bucket.remove(&node.id);
                if bucket.is_empty() { self.secondary_indexes.get_mut(name).unwrap().remove(&key); }
            }
        }
    }

    /// Mirrors `PolyGraph.markDirty`: guards against re-marking a node that's
    /// mid-removal.
    fn mark_dirty(&mut self, id: &str) {
        if !self.removed_node_ids.contains(id) {
            self.dirty_nodes.insert(id.to_string());
        }
    }

    /// Mirrors `PolyGraph.touchHotCache`: moves `id` to the most-recently-used
    /// end, checking the cap only every 10th call (amortized — `flush` also
    /// runs the check unconditionally after every successful apply).
    fn touch_hot_cache(&mut self, id: &str) {
        self.hot_cache_order.touch(id);
        self.eviction_skip_counter += 1;
        if self.eviction_skip_counter.is_multiple_of(10) {
            self.evict_oldest_if_over_cap();
        }
    }

    /// Evict least-recently-used nodes down to `config.hot_cache_max`. A
    /// dirty (unflushed) evictee is snapshotted into `evicted_dirty_nodes`
    /// first so its edits aren't lost. Mirrors
    /// `PolyGraph.evictOldestIfOverCap`.
    fn evict_oldest_if_over_cap(&mut self) {
        while self.hot_cache_order.len() > self.config.hot_cache_max {
            let Some(evict_id) = self.hot_cache_order.pop_front() else { break };
            if let Some(node) = self.nodes.get(&evict_id) {
                if self.dirty_nodes.contains(&evict_id) {
                    self.evicted_dirty_nodes.insert(evict_id.clone(), node.clone());
                }
            }
            self.unindex_node(&evict_id);
            self.nodes.remove(&evict_id);
            self.hnsw.remove(&evict_id);
        }
    }

    /// Mirrors `PolyGraph.emitChange`: queues while a batch is open (see
    /// `start_batch`), otherwise dispatches immediately.
    fn emit(&mut self, event: GraphChangeEvent) {
        if self.batch_depth > 0 {
            self.pending_batch_events.push(event);
        } else if let Some(cb) = self.on_change.as_mut() {
            cb(event);
        }
    }
}

pub(crate) fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypack_core::model::edge_id;
    use polypack_core::InMemoryStorage;

    fn test_graph() -> Graph {
        Graph::open(Box::new(InMemoryStorage::new()), StoreConfig::default(), GraphConfig::default())
            .unwrap()
    }

    fn node(id: &str) -> Node {
        node_of_type(id, "doc")
    }

    fn node_of_type(id: &str, node_type: &str) -> Node {
        Node {
            id: id.into(),
            node_type: node_type.into(),
            data: serde_json::Map::new(),
            vector: None,
            inserted_at: 1,
            updated_at: 1,
            revision: 0,
            activation: None,
        }
    }

    #[test]
    fn migrations_are_atomic_and_report_changed_records() {
        let mut graph = test_graph();
        graph.add_node(node("a")).unwrap();
        graph.register_migration(crate::migration::MigrationDefinition::new(1, 2, |mut node| {
            node.data.insert("schemaVersion".into(), 2.into());
            Ok(node)
        })).unwrap();

        let dry_run = graph.migrate(1, 2, MigrationOptions { dry_run: true, ..Default::default() }).unwrap();
        assert_eq!(dry_run.migrated_nodes, 1);
        let report = graph.migrate(1, 2, MigrationOptions::default()).unwrap();
        assert_eq!(report.processed_nodes, 1);
        assert_eq!(report.migrated_nodes, 1);
        assert_eq!(graph.get_node("a").unwrap().data["schemaVersion"], 2);
    }

    #[test]
    fn migrations_report_cumulative_progress_and_reject_empty_batches() {
        let mut graph = test_graph();
        graph.add_node(node("a")).unwrap();
        graph.add_node(node("b")).unwrap();
        graph.register_migration(crate::migration::MigrationDefinition::new(1, 2, |mut node| {
            node.data.insert("migrated".into(), true.into());
            Ok(node)
        })).unwrap();
        let progress = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = progress.clone();
        let report = graph.migrate(1, 2, MigrationOptions {
            batch_size: 1,
            on_progress: Some(Box::new(move |update| observed.lock().unwrap().push((update.processed_nodes, update.migrated_nodes)))),
            ..Default::default()
        }).unwrap();
        assert_eq!(report.migrated_nodes, 2);
        assert_eq!(*progress.lock().unwrap(), vec![(1, 1), (2, 2)]);
        assert!(graph.migrate(1, 2, MigrationOptions { batch_size: 0, ..Default::default() }).is_err());
    }

    #[test]
    fn flush_persists_a_dirty_node() {
        let mut g = test_graph();
        g.nodes.insert("a".into(), node("a"));
        g.dirty_nodes.insert("a".into());

        g.flush().unwrap();

        assert!(!g.has_pending_persistence());
        assert_eq!(g.store.get_node("a").unwrap().unwrap().id, "a");
    }

    #[test]
    fn flush_persists_a_dirty_vector() {
        let mut g = test_graph();
        g.hnsw.add("a", &[1.0, 0.0]).unwrap();
        g.dirty_vectors.insert("a".into());

        g.flush().unwrap();

        assert!(!g.has_pending_persistence());
        assert_eq!(g.store.get_vector("a").unwrap().unwrap(), vec![1.0, 0.0]);
    }

    #[test]
    fn flush_persists_a_dirty_edge() {
        let mut g = test_graph();
        let id = edge_id("a", "REL", "b");
        let inner = format!("{}::{}", "REL", "b");
        g.edges.entry("a".into()).or_default().insert(
            inner,
            EdgeEntry { id: "a::REL::b".into(), revision: 0, target: "b".into(), edge_type: "REL".into(), data: None, ownership: EdgeOwnership::Reference },
        );
        g.dirty_edges.insert(id.clone());

        g.flush().unwrap();

        assert!(!g.has_pending_persistence());
        let edges = g.store.edges_snapshot().unwrap();
        assert!(edges.iter().any(|(eid, e)| *eid == id && e.source == "a" && e.target == "b"));
    }

    #[test]
    fn flush_deletes_a_removed_node_and_its_vector() {
        let mut g = test_graph();
        g.nodes.insert("a".into(), node("a"));
        g.dirty_nodes.insert("a".into());
        g.flush().unwrap();

        g.removed_node_ids.insert("a".into());
        g.flush().unwrap();

        assert!(!g.has_pending_persistence());
        assert!(g.store.get_node("a").unwrap().is_none());
        assert!(g.store.get_vector("a").unwrap().is_none());
    }

    #[test]
    fn flush_is_a_noop_with_nothing_pending() {
        let mut g = test_graph();
        assert!(!g.has_pending_persistence());
        g.flush().unwrap();
    }

    #[test]
    fn add_node_indexes_by_type_and_marks_dirty() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();

        assert_eq!(g.size(), 1);
        assert!(g.has_loaded_node("a"));
        assert_eq!(g.where_type("doc").len(), 1);
        assert!(g.dirty_nodes.contains("a"));
    }

    #[test]
    fn add_node_with_vector_adds_to_hnsw() {
        let mut g = test_graph();
        let mut n = node("a");
        n.vector = Some(vec![1.0, 0.0]);
        g.add_node(n).unwrap();

        assert!(g.hnsw.has("a"));
        assert!(g.dirty_nodes.contains("a"));
    }

    #[test]
    fn write_resource_limits_reject_before_mutation() {
        let mut g = Graph::open(
            Box::new(InMemoryStorage::new()),
            StoreConfig::default(),
            GraphConfig {
                resource_limits: GraphResourceLimits { max_vector_dimensions: Some(2), max_node_payload_bytes: Some(20), max_batch_size: Some(1) },
                ..GraphConfig::default()
            },
        ).unwrap();
        let mut wide = node("wide");
        wide.vector = Some(vec![1.0, 2.0, 3.0]);
        assert!(matches!(g.add_node(wide), Err(PolypackError::ResourceLimit { name, .. }) if name == "maxVectorDimensions"));
        let mut large = node("large");
        large.data.insert("value".into(), "payload exceeds limit".into());
        assert!(matches!(g.add_node(large), Err(PolypackError::ResourceLimit { name, .. }) if name == "maxNodePayloadBytes"));
        assert_eq!(g.size(), 0);
        assert!(matches!(g.add_nodes(vec![node("a"), node("b")]), Err(PolypackError::ResourceLimit { name, .. }) if name == "maxBatchSize"));
        assert_eq!(g.size(), 0);
        let mut existing = node("existing");
        existing.data.insert("value".into(), "ok".into());
        g.add_node(existing).unwrap();
        let mut oversized_update = serde_json::Map::new();
        oversized_update.insert("value".into(), "updated payload exceeds limit".into());
        assert!(matches!(g.update_node("existing", oversized_update, None, None), Err(PolypackError::ResourceLimit { name, .. }) if name == "maxNodePayloadBytes"));
        assert_eq!(g.get_node("existing").unwrap().data["value"], "ok");
    }

    #[test]
    fn resource_limits_can_be_reconfigured_after_open() {
        let mut g = test_graph();
        let limits = GraphResourceLimits { max_vector_dimensions: Some(2), max_node_payload_bytes: Some(20), max_batch_size: Some(1) };
        g.set_resource_limits(limits.clone()).unwrap();
        assert_eq!(g.resource_limit_config(), limits);
        assert!(matches!(g.set_resource_limits(GraphResourceLimits { max_batch_size: Some(0), ..Default::default() }), Err(PolypackError::InvalidArgument(_))));

        let mut wide = node("wide");
        wide.vector = Some(vec![1.0, 2.0, 3.0]);
        assert!(matches!(g.add_node(wide), Err(PolypackError::ResourceLimit { name, .. }) if name == "maxVectorDimensions"));
    }

    #[test]
    fn transaction_resource_limit_rolls_back_and_resets_after_rejection() {
        let mut g = Graph::open(
            Box::new(InMemoryStorage::new()),
            StoreConfig::default(),
            GraphConfig {
                resource_limits: GraphResourceLimits { max_batch_size: Some(1), ..Default::default() },
                ..GraphConfig::default()
            },
        )
        .unwrap();

        let result: Result<()> = g.transaction(|tx| {
            tx.add_node(node("first"))?;
            tx.add_node(node("second"))?;
            Ok(())
        });
        assert!(matches!(result, Err(PolypackError::ResourceLimit { name, limit }) if name == "maxBatchSize" && limit == 1));
        assert!(g.get_node("first").is_none());
        assert!(g.get_node("second").is_none());

        g.transaction(|tx| {
            tx.add_node(node("after-rollback"))?;
            Ok(())
        })
        .unwrap();
        assert!(g.get_node("after-rollback").is_some());
    }

    #[test]
    fn add_node_rejects_empty_id() {
        let mut g = test_graph();
        let mut n = node("a");
        n.id = "".into();
        assert!(g.add_node(n).is_err());
        assert_eq!(g.size(), 0);
    }

    #[test]
    fn add_node_replacing_by_id_reindexes_type_and_vector() {
        let mut g = test_graph();
        let mut first = node_of_type("a", "doc");
        first.vector = Some(vec![1.0, 0.0]);
        g.add_node(first).unwrap();
        assert!(g.hnsw.has("a"));
        assert_eq!(g.where_type("doc").len(), 1);

        // Replace with a different type and no vector: old type index is
        // dropped, the stale vector is removed, and the node is marked for
        // vector deletion on next flush.
        let second = node_of_type("a", "article");
        g.add_node(second).unwrap();

        assert_eq!(g.where_type("doc").len(), 0);
        assert_eq!(g.where_type("article").len(), 1);
        assert!(!g.hnsw.has("a"));
        assert!(g.removed_vector_ids.contains("a"));
    }

    #[test]
    fn add_nodes_batches_and_rejects_all_on_one_invalid() {
        let mut g = test_graph();
        let mut bad = node("b");
        bad.node_type = "".into();
        let err = g.add_nodes(vec![node("a"), bad]).unwrap_err();
        let _ = err;

        assert_eq!(g.size(), 0, "an invalid entry must insert nothing");
    }

    #[test]
    fn add_nodes_inserts_all_when_valid() {
        let mut g = test_graph();
        g.add_nodes(vec![node("a"), node("b")]).unwrap();

        assert_eq!(g.size(), 2);
        assert_eq!(g.where_type("doc").len(), 2);
    }

    #[test]
    fn add_edge_indexes_both_directions_and_marks_dirty() {
        let mut g = test_graph();
        g.add_edge("a", "REL", "b", None, EdgeOwnership::Reference).unwrap();

        let id = edge_id("a", "REL", "b");
        assert!(g.dirty_edges.contains(&id));
        assert!(g.edges.get("a").unwrap().contains_key("REL::b"));
        assert!(g.node_to_edge.get("b").unwrap().contains("a"));
    }

    #[test]
    fn add_edge_rejects_empty_fields() {
        let mut g = test_graph();
        assert!(g.add_edge("", "REL", "b", None, EdgeOwnership::Reference).is_err());
        assert!(g.add_edge("a", "", "b", None, EdgeOwnership::Reference).is_err());
        assert!(g.add_edge("a", "REL", "", None, EdgeOwnership::Reference).is_err());
        assert!(g.edges.is_empty());
    }

    #[test]
    fn add_edge_is_a_noop_for_a_duplicate_triple() {
        let mut g = test_graph();
        let mut data = serde_json::Map::new();
        data.insert("weight".into(), serde_json::json!(1));
        g.add_edge("a", "REL", "b", Some(data), EdgeOwnership::Owned).unwrap();

        // Same (source, type, target) triple again, with different data/ownership —
        // must not overwrite the existing entry.
        g.add_edge("a", "REL", "b", None, EdgeOwnership::Reference).unwrap();

        let entry = &g.edges["a"]["REL::b"];
        assert_eq!(entry.ownership, EdgeOwnership::Owned);
        assert!(entry.data.is_some());
        assert_eq!(g.edges["a"].len(), 1);
    }

    #[test]
    fn explicit_edge_ids_allow_parallel_edges_and_round_trip() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_edge("a", "REL", "b", None, EdgeOwnership::Reference).unwrap();
        g.add_edge_with_id("claim-2", "a", "REL", "b", None, EdgeOwnership::Reference).unwrap();
        assert_eq!(g.get_edges("a", Some("REL")).len(), 2);

        g.flush().unwrap();
        g.clear();
        g.warm().unwrap();
        let ids: HashSet<String> = g.get_edges("a", Some("REL")).iter().map(|edge| edge.id.clone()).collect();
        assert_eq!(ids, HashSet::from(["a::REL::b".into(), "claim-2".into()]));
    }

    #[test]
    fn conditional_edge_update_rejects_stale_revision() {
        let mut g = test_graph();
        g.add_edge_with_id("claim-1", "a", "REL", "b", None, EdgeOwnership::Reference).unwrap();
        let mut data = serde_json::Map::new();
        data.insert("source".into(), serde_json::json!("archive"));
        let err = g.update_edge_if_revision("claim-1", Some(data), None, 1).unwrap_err();
        assert!(matches!(err, PolypackError::Conflict { expected: 1, actual: 0, .. }));
        assert_eq!(g.get_edges("a", Some("REL"))[0].revision, 0);
    }

    #[test]
    fn conditional_edge_remove_only_removes_the_requested_parallel_edge() {
        let mut g = test_graph();
        g.add_edge("a", "REL", "b", None, EdgeOwnership::Reference).unwrap();
        g.add_edge_with_id("claim-2", "a", "REL", "b", None, EdgeOwnership::Reference).unwrap();

        assert!(g.remove_edge_if_revision("claim-2", 1).is_err());
        assert_eq!(g.get_edges("a", Some("REL")).len(), 2);
        assert!(g.remove_edge_if_revision("claim-2", 0).unwrap());
        let remaining = g.get_edges("a", Some("REL"));
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, "a::REL::b");
    }

    #[test]
    fn add_edge_allows_multiple_edge_types_between_same_pair() {
        let mut g = test_graph();
        g.add_edge("a", "LIKES", "b", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("a", "FOLLOWS", "b", None, EdgeOwnership::Reference).unwrap();

        assert_eq!(g.edges["a"].len(), 2);
        assert_eq!(g.node_to_edge["b"].len(), 1, "still a single source, just two edge types");
    }

    #[test]
    fn add_edge_removes_the_id_from_removed_edge_ids() {
        let mut g = test_graph();
        let id = edge_id("a", "REL", "b");
        g.removed_edge_ids.insert(id.clone());

        g.add_edge("a", "REL", "b", None, EdgeOwnership::Reference).unwrap();

        assert!(!g.removed_edge_ids.contains(&id));
    }

    #[test]
    fn get_edges_returns_all_or_filtered_by_type() {
        let mut g = test_graph();
        g.add_edge("a", "LIKES", "b", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("a", "FOLLOWS", "c", None, EdgeOwnership::Reference).unwrap();

        assert_eq!(g.get_edges("a", None).len(), 2);
        let liked = g.get_edges("a", Some("LIKES"));
        assert_eq!(liked.len(), 1);
        assert_eq!(liked[0].target, "b");
        assert!(g.get_edges("missing", None).is_empty());
    }

    #[test]
    fn get_edge_targets_filters_by_type() {
        let mut g = test_graph();
        g.add_edge("a", "LIKES", "b", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("a", "LIKES", "c", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("a", "FOLLOWS", "d", None, EdgeOwnership::Reference).unwrap();

        let mut liked = g.get_edge_targets("a", "LIKES");
        liked.sort();
        assert_eq!(liked, vec!["b", "c"]);
        assert!(g.get_edge_targets("a", "MISSING").is_empty());
    }

    #[test]
    fn get_edge_sources_finds_incoming_by_type() {
        let mut g = test_graph();
        g.add_edge("a", "LIKES", "target", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("b", "LIKES", "target", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("c", "FOLLOWS", "target", None, EdgeOwnership::Reference).unwrap();

        let mut sources = g.get_edge_sources("target", "LIKES");
        sources.sort();
        assert_eq!(sources, vec!["a", "b"]);
        assert!(g.get_edge_sources("target", "MISSING").is_empty());
    }

    #[test]
    fn remove_edges_all_from_source_drops_the_source_entry() {
        let mut g = test_graph();
        g.add_edge("a", "LIKES", "b", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("a", "FOLLOWS", "c", None, EdgeOwnership::Reference).unwrap();

        g.remove_edges("a", None, None).unwrap();

        assert!(!g.edges.contains_key("a"));
        assert!(!g.node_to_edge["b"].contains("a"));
        assert!(!g.node_to_edge["c"].contains("a"));
        assert!(g.removed_edge_ids.contains(&edge_id("a", "LIKES", "b")));
        assert!(g.removed_edge_ids.contains(&edge_id("a", "FOLLOWS", "c")));
    }

    #[test]
    fn remove_edges_filtered_by_type_keeps_the_rest() {
        let mut g = test_graph();
        g.add_edge("a", "LIKES", "b", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("a", "FOLLOWS", "c", None, EdgeOwnership::Reference).unwrap();

        g.remove_edges("a", Some("LIKES"), None).unwrap();

        assert_eq!(g.edges["a"].len(), 1);
        assert!(g.edges["a"].contains_key("FOLLOWS::c"));
        assert!(!g.node_to_edge["b"].contains("a"));
    }

    #[test]
    fn remove_edges_is_a_noop_when_nothing_matches() {
        let mut g = test_graph();
        g.add_edge("a", "LIKES", "b", None, EdgeOwnership::Reference).unwrap();

        g.remove_edges("a", Some("MISSING"), None).unwrap();

        assert_eq!(g.edges["a"].len(), 1);
        assert!(g.dirty_edges.contains(&edge_id("a", "LIKES", "b")));
        assert!(g.removed_edge_ids.is_empty());
    }

    #[test]
    fn remove_edges_fires_on_orphan_for_a_disconnected_shared_target() {
        let mut g = test_graph();
        g.add_edge("a", "OWNS", "b", None, EdgeOwnership::Shared).unwrap();

        let orphaned = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let orphaned_cb = orphaned.clone();
        g.on_orphan(move |id| orphaned_cb.borrow_mut().push(id.to_string()));

        g.remove_edges("a", None, None).unwrap();

        assert_eq!(*orphaned.borrow(), vec!["b".to_string()]);
    }

    #[test]
    fn remove_edges_does_not_fire_on_orphan_while_another_source_remains() {
        let mut g = test_graph();
        g.add_edge("a", "OWNS", "target", None, EdgeOwnership::Shared).unwrap();
        g.add_edge("b", "OWNS", "target", None, EdgeOwnership::Shared).unwrap();

        let orphaned = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let orphaned_cb = orphaned.clone();
        g.on_orphan(move |id| orphaned_cb.borrow_mut().push(id.to_string()));

        g.remove_edges("a", None, None).unwrap();

        assert!(orphaned.borrow().is_empty(), "target still reachable via b");
    }

    #[test]
    fn remove_node_removes_the_node_and_marks_removed() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();

        g.remove_node("a").unwrap();

        assert!(!g.has_loaded_node("a"));
        assert_eq!(g.where_type("doc").len(), 0);
        assert!(g.removed_node_ids.contains("a"));
        assert!(!g.dirty_nodes.contains("a"));
    }

    #[test]
    fn remove_node_on_a_missing_node_is_a_noop() {
        let mut g = test_graph();
        g.remove_node("missing").unwrap();
        assert!(!g.removed_node_ids.contains("missing"));
    }

    #[test]
    fn remove_node_removes_its_vector() {
        let mut g = test_graph();
        let mut n = node("a");
        n.vector = Some(vec![1.0, 0.0]);
        g.add_node(n).unwrap();

        g.remove_node("a").unwrap();

        assert!(!g.hnsw.has("a"));
    }

    #[test]
    fn remove_node_cleans_up_incoming_and_outgoing_edges() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_node(node("c")).unwrap();
        // a -> b, c -> a
        g.add_edge("a", "REL", "b", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("c", "REL", "a", None, EdgeOwnership::Reference).unwrap();

        g.remove_node("a").unwrap();

        assert!(!g.edges.contains_key("a"), "outgoing edges from a are gone");
        assert!(g.edges.get("c").is_none_or(|m| m.is_empty()), "c's edge into a is gone");
        assert!(!g.node_to_edge.get("b").is_some_and(|s| s.contains("a")));
        assert!(g.removed_edge_ids.contains(&edge_id("a", "REL", "b")));
        assert!(g.removed_edge_ids.contains(&edge_id("c", "REL", "a")));
    }

    #[test]
    fn remove_node_cascades_through_owned_edges() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_edge("a", "OWNS", "b", None, EdgeOwnership::Owned).unwrap();

        g.remove_node("a").unwrap();

        assert!(!g.has_loaded_node("a"));
        assert!(!g.has_loaded_node("b"), "owned target is cascade-removed");
    }

    #[test]
    fn checkpoint_flushes_pending_mutations_and_clears_dirty_state() {
        let mut g = test_graph();
        g.add_node(node("checkpointed")).unwrap();
        assert!(g.has_pending_persistence());
        g.checkpoint().unwrap();
        assert!(!g.has_pending_persistence());
        assert_eq!(g.persisted_size().unwrap(), 1);
    }

    #[test]
    fn remove_node_does_not_cascade_when_another_owned_source_remains() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_node(node("shared_target")).unwrap();
        g.add_edge("a", "OWNS", "shared_target", None, EdgeOwnership::Owned).unwrap();
        g.add_edge("b", "OWNS", "shared_target", None, EdgeOwnership::Owned).unwrap();

        g.remove_node("a").unwrap();

        assert!(g.has_loaded_node("shared_target"), "still owned by b");
    }

    #[test]
    fn remove_node_handles_a_cyclic_owned_chain() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        // a -> b (owned), b -> a (owned): a cycle.
        g.add_edge("a", "OWNS", "b", None, EdgeOwnership::Owned).unwrap();
        g.add_edge("b", "OWNS", "a", None, EdgeOwnership::Owned).unwrap();

        g.remove_node("a").unwrap();

        assert!(!g.has_loaded_node("a"));
        assert!(!g.has_loaded_node("b"));
    }

    #[test]
    fn remove_edges_cascades_owned_target_via_remove_node() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_edge("a", "OWNS", "b", None, EdgeOwnership::Owned).unwrap();

        g.remove_edges("a", None, None).unwrap();

        assert!(!g.has_loaded_node("b"), "owned target cascade-removed by remove_edges");
    }

    /// Simulate hot-cache eviction (`evictOldestIfOverCap` in `graph.ts`) —
    /// `touch_hot_cache`'s eviction isn't wired up yet, so tests that need an
    /// "evicted but persisted" node reproduce it by hand.
    fn evict(g: &mut Graph, id: &str) {
        g.unindex_node(id);
        g.nodes.remove(id);
        g.hnsw.remove(id);
    }

    #[test]
    fn get_node_safe_restores_an_evicted_node_from_the_store() {
        let mut g = test_graph();
        let mut n = node("a");
        n.vector = Some(vec![1.0, 0.0]);
        g.add_node(n).unwrap();
        g.flush().unwrap();
        evict(&mut g, "a");

        let restored = g.get_node_safe("a").unwrap();

        assert!(restored.is_some());
        assert!(g.has_loaded_node("a"));
        assert!(g.hnsw.has("a"));
        assert_eq!(g.where_type("doc").len(), 1);
    }

    #[test]
    fn get_node_safe_returns_none_for_a_removed_node() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.flush().unwrap();
        g.remove_node("a").unwrap();
        g.flush().unwrap();

        assert!(g.get_node_safe("a").unwrap().is_none());
    }

    #[test]
    fn get_node_safe_returns_none_for_a_missing_node() {
        let mut g = test_graph();
        assert!(g.get_node_safe("missing").unwrap().is_none());
    }

    #[test]
    fn remove_node_safe_restores_then_removes_an_evicted_node() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.flush().unwrap();
        evict(&mut g, "a");

        let removed = g.remove_node_safe("a").unwrap();

        assert!(removed);
        assert!(!g.has_loaded_node("a"));
        assert!(g.removed_node_ids.contains("a"));
    }

    #[test]
    fn remove_node_safe_returns_false_for_a_missing_node() {
        let mut g = test_graph();
        assert!(!g.remove_node_safe("missing").unwrap());
    }

    #[test]
    fn remove_node_safe_cascades_through_an_evicted_owned_target() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_edge("a", "OWNS", "b", None, EdgeOwnership::Owned).unwrap();
        g.flush().unwrap();
        evict(&mut g, "b");

        let removed = g.remove_node_safe("a").unwrap();

        assert!(removed);
        assert!(!g.has_loaded_node("a"));
        assert!(!g.has_loaded_node("b"));
        assert!(g.removed_node_ids.contains("b"));
    }

    #[test]
    fn remove_node_safe_does_not_cascade_when_another_owner_remains() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_node(node("shared_target")).unwrap();
        g.add_edge("a", "OWNS", "shared_target", None, EdgeOwnership::Owned).unwrap();
        g.add_edge("b", "OWNS", "shared_target", None, EdgeOwnership::Owned).unwrap();

        let removed = g.remove_node_safe("a").unwrap();

        assert!(removed);
        assert!(g.has_loaded_node("shared_target"));
    }

    #[test]
    fn update_node_merges_data_and_marks_dirty() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.flush().unwrap();

        let mut patch = serde_json::Map::new();
        patch.insert("title".into(), serde_json::json!("hello"));
        let updated = g.update_node("a", patch, None, None).unwrap().unwrap();

        assert_eq!(updated.data.get("title"), Some(&serde_json::json!("hello")));
        assert_eq!(updated.revision, 1);
        assert!(g.dirty_nodes.contains("a"));
    }

    #[test]
    fn update_node_preserves_untouched_fields() {
        let mut g = test_graph();
        let mut n = node("a");
        n.data.insert("kept".into(), serde_json::json!(1));
        g.add_node(n).unwrap();

        let mut patch = serde_json::Map::new();
        patch.insert("added".into(), serde_json::json!(2));
        let updated = g.update_node("a", patch, None, None).unwrap().unwrap();

        assert_eq!(updated.data.get("kept"), Some(&serde_json::json!(1)));
        assert_eq!(updated.data.get("added"), Some(&serde_json::json!(2)));
    }

    #[test]
    fn conditional_update_rejects_stale_revision_without_mutation() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        let mut patch = serde_json::Map::new();
        patch.insert("title".into(), serde_json::json!("new"));

        let err = g.update_node_if_revision("a", 1, patch, None, None).unwrap_err();
        assert!(matches!(err, PolypackError::Conflict { expected: 1, actual: 0, .. }));
        assert!(g.get_node("a").unwrap().data.get("title").is_none());
    }

    #[test]
    fn conditional_remove_rejects_stale_revision_without_removing() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();

        let err = g.remove_node_if_revision("a", 1).unwrap_err();
        assert!(matches!(err, PolypackError::Conflict { expected: 1, actual: 0, .. }));
        assert!(g.get_node("a").is_some());
    }

    #[test]
    fn patch_node_applies_nested_set_unset_and_increment_atomically() {
        let mut g = test_graph();
        let mut n = node("a");
        n.data.insert("profile".into(), serde_json::json!({"name": "old", "views": 2}));
        n.data.insert("temporary".into(), serde_json::json!(true));
        g.add_node(n).unwrap();

        let mut set = serde_json::Map::new();
        set.insert("profile.name".into(), serde_json::json!("new"));
        let mut increment = serde_json::Map::new();
        increment.insert("profile.views".into(), serde_json::json!(3));
        let updated = g.patch_node("a", set, vec!["temporary".into()], increment, serde_json::Map::new(), Some(0)).unwrap().unwrap();

        assert_eq!(updated.revision, 1);
        assert_eq!(updated.data["profile"]["name"], serde_json::json!("new"));
        assert_eq!(updated.data["profile"]["views"], serde_json::json!(5.0));
        assert!(!updated.data.contains_key("temporary"));

        let mut compare_and_set = serde_json::Map::new();
        compare_and_set.insert("data.profile.name".into(), serde_json::json!({"expected": "new", "value": "final"}));
        let updated = g.patch_node("a", serde_json::Map::new(), Vec::new(), serde_json::Map::new(), compare_and_set, Some(1)).unwrap().unwrap();
        assert_eq!(updated.data["profile"]["name"], serde_json::json!("final"));

        let mut stale_compare = serde_json::Map::new();
        stale_compare.insert("data.profile.name".into(), serde_json::json!({"expected": "new", "value": "stale"}));
        assert!(matches!(g.patch_node("a", serde_json::Map::new(), Vec::new(), serde_json::Map::new(), stale_compare, Some(2)), Err(PolypackError::Conflict { .. })));
    }

    #[test]
    fn transaction_rolls_back_callback_errors_and_commits_events_after_success() {
        let mut g = test_graph();
        let error: Result<()> = g.transaction(|tx| {
            assert!(tx.transaction_id().is_some());
            tx.add_node(node("rolled-back"))?;
            assert!(tx.get_node("rolled-back").is_some());
            Err(PolypackError::InvalidArgument("abort".into()))
        });
        assert!(error.is_err());
        assert!(g.get_node("rolled-back").is_none());

        let mut committed_transaction_id = None;
        let mut committed_operation_id = None;
        g.transaction(|tx| {
            committed_transaction_id = tx.transaction_id().map(str::to_string);
            committed_operation_id = tx.operation_id().map(str::to_string);
            assert!(committed_transaction_id.is_some());
            assert!(committed_operation_id.is_some());
            tx.add_node(node("committed"))?;
            assert!(tx.get_node("committed").is_some());
            Ok(())
        })
        .unwrap();
        assert!(g.get_node("committed").is_some());
        assert!(g.transaction_id().is_none());
        assert!(g.operation_id().is_none());
        let log = g.mutation_log().unwrap();
        assert_eq!(log.last().map(|record| record.transaction_id.as_str()), committed_transaction_id.as_deref());
        assert_eq!(log.last().map(|record| record.operation_id.as_str()), committed_operation_id.as_deref());
    }

    #[test]
    fn query_snapshot_is_detached_from_later_mutations() {
        let mut g = test_graph();
        g.add_node(node("stable")).unwrap();
        let snapshot = g.snapshot();

        let mut patch = serde_json::Map::new();
        patch.insert("value".into(), serde_json::json!(2));
        g.update_node("stable", patch, None, None).unwrap();
        g.add_node(node("later")).unwrap();

        assert_eq!(snapshot.nodes.len(), 1);
        assert_eq!(snapshot.nodes[0].id, "stable");
        assert!(snapshot.nodes[0].data.get("value").is_none());
    }

    #[test]
    fn transaction_rejects_nesting() {
        let mut g = test_graph();
        let result = g.transaction(|tx| tx.transaction(|_| Ok(())));
        assert!(matches!(result, Err(PolypackError::InvalidArgument(_))));
    }

    #[test]
    fn transaction_accepts_caller_operation_identity() {
        let mut g = test_graph();
        g.transaction_with_identity(Some("import-1".into()), |tx| {
            tx.add_node(node("imported"))?;
            assert_eq!(tx.operation_id(), Some("import-1"));
            Ok(())
        }).unwrap();
        let record = g.mutation_log().unwrap().pop().unwrap();
        assert_eq!(record.operation_id, "import-1");
    }

    #[test]
    fn reports_in_memory_adapter_capabilities() {
        let g = test_graph();
        let capabilities = g.capabilities();
        assert!(capabilities.atomic_batches);
        assert!(capabilities.transactions);
        assert_eq!(capabilities.vector_search, polypack_core::storage::VectorSearchCapability::Exact);
        assert!(!capabilities.concurrent_writers);
    }

    #[test]
    fn enforces_required_adapter_capabilities() {
        let graph = test_graph();
        graph.require_capabilities(AdapterCapabilities {
            atomic_batches: true,
            transactions: true,
            vector_search: polypack_core::storage::VectorSearchCapability::Exact,
            ..Default::default()
        }).unwrap();
        let error = graph.require_capabilities(AdapterCapabilities { fsync: true, ..Default::default() }).unwrap_err();
        assert!(error.to_string().contains("fsync"));
    }

    #[test]
    fn unique_indexes_reject_duplicate_insert_update_and_patch() {
        let mut graph = test_graph();
        graph.define_index(IndexDefinition {
            name: "external-id".into(),
            node_type: Some("record".into()),
            fields: vec!["provider".into(), "externalId".into()],
            unique: true,
            sparse: true,
        }).unwrap();
        let mut first = node_of_type("a", "record");
        first.data.insert("provider".into(), "acme".into());
        first.data.insert("externalId".into(), "1".into());
        graph.add_node(first).unwrap();
        let mut duplicate = node_of_type("b", "record");
        duplicate.data.insert("provider".into(), "acme".into());
        duplicate.data.insert("externalId".into(), "1".into());
        assert!(graph.add_node(duplicate).is_err());
        assert!(graph.patch_node("a", serde_json::Map::from_iter([(String::from("externalId"), "2".into())]), Vec::new(), serde_json::Map::new(), serde_json::Map::new(), None).is_ok());
        assert!(graph.update_node("a", serde_json::Map::from_iter([(String::from("externalId"), "1".into())]), None, None).is_ok());
        assert_eq!(graph.indexes()[0].name, "external-id");
        let query = graph.query().where_field("provider", "acme".into()).where_field("externalId", "1".into());
        assert_eq!(query.explain().index.as_deref(), Some("external-id"));
        assert_eq!(query.to_array().iter().map(|node| node.id.as_str()).collect::<Vec<_>>(), vec!["a"]);
        drop(query);
        graph.clear();
        assert!(graph.query().where_field("provider", "acme".into()).where_field("externalId", "1".into()).to_array().is_empty());
    }

    #[test]
    fn numeric_range_queries_use_secondary_index_candidates() {
        let mut graph = test_graph();
        graph.define_index(IndexDefinition { name: "birth-year".into(), fields: vec!["birthYear".into()], sparse: true, ..Default::default() }).unwrap();
        for (id, year) in [("young", 1980), ("middle", 2020), ("old", 2050)] {
            let mut node = node_of_type(id, "person");
            node.data.insert("birthYear".into(), year.into());
            graph.add_node(node).unwrap();
        }
        let query = graph.query().where_node_type(vec!["person".into()]).where_attribute_range("birthYear", Some(2000.0), None);
        assert_eq!(query.explain().index.as_deref(), Some("birth-year"));
        assert_eq!(query.explain().stages[0], "property-index(birth-year)");
        let mut hot_ids = query.to_array().into_iter().map(|node| node.id).collect::<Vec<_>>();
        hot_ids.sort();
        assert_eq!(hot_ids, vec!["middle", "old"]);
        drop(query);
        graph.flush().unwrap();
        graph.clear();
        let mut persisted_ids = graph.query_persisted().where_attribute_range("birthYear", Some(2000.0), None).ids().unwrap();
        persisted_ids.sort();
        assert_eq!(persisted_ids, vec!["middle", "old"]);
    }

    #[test]
    fn hot_queries_intersect_secondary_index_candidates() {
        let mut graph = test_graph();
        graph.define_index(IndexDefinition { name: "surname".into(), fields: vec!["surname".into()], ..Default::default() }).unwrap();
        graph.define_index(IndexDefinition { name: "birth-year".into(), fields: vec!["birthYear".into()], ..Default::default() }).unwrap();
        for (id, surname, year) in [("a", "Smith", 1980), ("b", "Smith", 1990), ("c", "Jones", 1980)] {
            let mut node = node_of_type(id, "person");
            node.data.insert("surname".into(), surname.into());
            node.data.insert("birthYear".into(), year.into());
            graph.add_node(node).unwrap();
        }
        let query = graph.query().where_field("surname", "Smith".into()).where_field("birthYear", 1980.into());
        let explanation = query.explain();
        assert_eq!(explanation.indexes, vec!["birth-year", "surname"]);
        assert!(explanation.stages.contains(&"index-intersection(2)".to_string()));
        assert_eq!(query.ids(), vec!["a"]);
    }

    #[test]
    fn persisted_queries_intersect_secondary_index_candidates() {
        let mut graph = test_graph();
        graph.define_index(IndexDefinition { name: "surname".into(), fields: vec!["surname".into()], ..Default::default() }).unwrap();
        graph.define_index(IndexDefinition { name: "birth-year".into(), fields: vec!["birthYear".into()], ..Default::default() }).unwrap();
        for (id, surname, year) in [("a", "Smith", 1980), ("b", "Smith", 1990), ("c", "Jones", 1980)] {
            let mut node = node_of_type(id, "person");
            node.data.insert("surname".into(), surname.into());
            node.data.insert("birthYear".into(), year.into());
            graph.add_node(node).unwrap();
        }
        graph.flush().unwrap();
        let mut query = graph.query_persisted().where_field("surname", "Smith".into()).where_field("birthYear", 1980.into());
        let explanation = query.explain().unwrap();
        assert_eq!(explanation.indexes, vec!["birth-year", "surname"]);
        assert!(explanation.stages.contains(&"index-intersection(2)".to_string()));
        assert_eq!(query.ids().unwrap(), vec!["a"]);
    }

    #[test]
    fn indexes_reject_duplicate_compound_fields() {
        let mut graph = test_graph();
        let error = graph.define_index(IndexDefinition { name: "invalid".into(), fields: vec!["email".into(), "email".into()], ..Default::default() }).unwrap_err();
        assert!(error.to_string().contains("fields"));
    }

    #[test]
    fn index_metadata_survives_store_reopen() {
        let storage = shared_storage();
        let mut first = graph_on(&storage, GraphConfig::default());
        first.define_index(IndexDefinition { name: "lookup".into(), fields: vec!["key".into()], ..Default::default() }).unwrap();
        let mut record = node_of_type("persisted", "record");
        record.data.insert("key".into(), "value".into());
        first.add_node(record).unwrap();
        first.flush().unwrap();
        first.clear();
        let mut persisted = first.query_persisted().where_field("key", "value".into());
        assert_eq!(persisted.explain().unwrap().index.as_deref(), Some("lookup"));
        assert_eq!(persisted.explain().unwrap().stages[0], "property-index(lookup)");
        assert_eq!(persisted.ids().unwrap(), vec!["persisted"]);
        drop(persisted);
        let mut second = graph_on(&storage, GraphConfig::default());
        assert_eq!(second.indexes().iter().map(|index| index.name.as_str()).collect::<Vec<_>>(), vec!["lookup"]);
        assert_eq!(second.query_persisted().where_field("key", "value".into()).ids().unwrap(), vec!["persisted"]);
    }

    #[test]
    fn reports_graph_statistics() {
        let mut g = test_graph();
        g.define_index(IndexDefinition { name: "lookup".into(), fields: vec!["key".into()], ..Default::default() }).unwrap();
        let mut record = node("a");
        record.data.insert("key".into(), "value".into());
        g.add_node(record).unwrap();
        g.add_edge("a", "REL", "b", None, EdgeOwnership::Reference).unwrap();
        assert_eq!(g.query().where_field("key", "value".into()).ids(), vec!["a"]);
        let stats = g.stats().unwrap();
        assert_eq!(stats.loaded_node_count, 1);
        assert_eq!(stats.persisted_node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert!(stats.pending_persistence);
        assert!(stats.dirty_record_count > 0);
        assert_eq!(stats.query_count, 1);
        assert_eq!(stats.query_scanned_records, 1);
        assert_eq!(stats.query_index_usage.get("lookup"), Some(&1));
    }

    #[test]
    fn schema_hooks_validate_nodes_and_edge_endpoint_types() {
        let mut g = test_graph();
        g.register_node_type("person", NodeTypeDefinition { required_fields: vec!["name".into()], data_types: HashMap::new() }).unwrap();
        g.register_edge_type(
            "KNOWS",
            EdgeTypeDefinition { source_types: vec!["person".into()], target_types: vec!["person".into()], cardinality: None, required_fields: vec![], data_types: HashMap::new() },
        ).unwrap();
        let mut invalid = node_of_type("a", "person");
        assert!(g.add_node(invalid.clone()).is_err());
        invalid.data.insert("name".into(), serde_json::json!("Ada"));
        g.add_node(invalid).unwrap();
        g.add_node(node_of_type("b", "doc")).unwrap();
        assert!(g.add_edge("a", "KNOWS", "b", None, EdgeOwnership::Reference).is_err());
    }

    #[test]
    fn schema_hooks_enforce_edge_cardinality_and_referential_integrity() {
        let mut g = test_graph();
        g.add_node(node_of_type("a", "person")).unwrap();
        g.add_node(node_of_type("b", "person")).unwrap();
        g.add_node(node_of_type("c", "person")).unwrap();
        g.register_edge_type(
            "PARENT",
            EdgeTypeDefinition { source_types: vec![], target_types: vec![], cardinality: Some("one-to-many".into()), required_fields: vec![], data_types: HashMap::new() },
        ).unwrap();
        g.add_edge("a", "PARENT", "b", None, EdgeOwnership::Reference).unwrap();
        assert!(g.add_edge("c", "PARENT", "b", None, EdgeOwnership::Reference).is_err());
        assert!(g.add_edge("a", "PARENT", "missing", None, EdgeOwnership::Reference).is_err());
    }

    #[test]
    fn update_node_replaces_the_vector_and_marks_it_dirty() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.flush().unwrap();

        let updated = g.update_node("a", serde_json::Map::new(), Some(vec![1.0, 2.0]), None).unwrap().unwrap();

        assert_eq!(updated.vector, Some(vec![1.0, 2.0]));
        assert!(g.hnsw.has("a"));
        assert!(g.dirty_vectors.contains("a"));
    }

    #[test]
    fn update_node_rejects_a_non_finite_vector() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();

        let err = g.update_node("a", serde_json::Map::new(), Some(vec![f64::NAN]), None).unwrap_err();
        let _ = err;
        assert!(g.get_node("a").unwrap().vector.is_none(), "node must be untouched on error");
    }

    #[test]
    fn update_node_returns_none_for_a_node_not_in_the_hot_cache() {
        let mut g = test_graph();
        let mut patch = serde_json::Map::new();
        patch.insert("title".into(), serde_json::json!("hello"));
        assert!(g.update_node("missing", patch, None, None).unwrap().is_none());
    }

    fn shared_storage() -> std::sync::Arc<std::sync::Mutex<InMemoryStorage>> {
        std::sync::Arc::new(std::sync::Mutex::new(InMemoryStorage::new()))
    }

    fn graph_on(storage: &std::sync::Arc<std::sync::Mutex<InMemoryStorage>>, config: GraphConfig) -> Graph {
        Graph::open(Box::new(storage.clone()), StoreConfig::default(), config).unwrap()
    }

    fn small_cache_graph(hot_cache_max: usize) -> Graph {
        Graph::open(
            Box::new(InMemoryStorage::new()),
            StoreConfig::default(),
            GraphConfig { hot_cache_max, ..GraphConfig::default() },
        )
        .unwrap()
    }

    #[test]
    fn warm_loads_persisted_nodes_edges_and_vectors_into_a_fresh_graph() {
        let storage = shared_storage();
        {
            let mut g = graph_on(&storage, GraphConfig::default());
            let mut a = node("a");
            a.vector = Some(vec![1.0, 0.0]);
            g.add_node(a).unwrap();
            g.add_node(node("b")).unwrap();
            g.add_edge("a", "REL", "b", None, EdgeOwnership::Owned).unwrap();
            g.flush().unwrap();
        }

        let mut g2 = graph_on(&storage, GraphConfig::default());
        g2.warm().unwrap();

        assert!(g2.has_loaded_node("a"));
        assert!(g2.has_loaded_node("b"));
        assert!(g2.hnsw.has("a"));
        assert_eq!(g2.get_edge_targets("a", "REL"), vec!["b"]);
        assert_eq!(
            g2.edges["a"]["REL::b"].ownership,
            EdgeOwnership::Owned,
            "ownership must survive the flush/warm round trip"
        );
    }

    #[test]
    fn warm_is_idempotent() {
        let storage = shared_storage();
        {
            let mut g = graph_on(&storage, GraphConfig::default());
            g.add_node(node("a")).unwrap();
            g.flush().unwrap();
        }

        let mut g2 = graph_on(&storage, GraphConfig::default());
        g2.warm().unwrap();
        g2.warm().unwrap();

        assert_eq!(g2.size(), 1);
    }

    #[test]
    fn warm_on_an_empty_store_is_a_noop() {
        let mut g = test_graph();
        g.warm().unwrap();
        assert_eq!(g.size(), 0);
    }

    #[test]
    fn warm_respects_hot_cache_max_by_keeping_the_newest_nodes() {
        let storage = shared_storage();
        {
            let mut g = graph_on(&storage, GraphConfig::default());
            let mut old = node("old");
            old.inserted_at = 1;
            let mut newer = node("newer");
            newer.inserted_at = 2;
            g.add_node(old).unwrap();
            g.add_node(newer).unwrap();
            g.flush().unwrap();
        }

        let mut g2 = graph_on(&storage, GraphConfig { hot_cache_max: 1, ..GraphConfig::default() });
        g2.warm().unwrap();

        assert_eq!(g2.size(), 1);
        assert!(g2.has_loaded_node("newer"));
        assert!(!g2.has_loaded_node("old"));
    }

    #[test]
    fn walk_ancestors_returns_root_to_start_inclusive() {
        let mut g = test_graph();
        g.add_node(node("root")).unwrap();
        g.add_node(node("mid")).unwrap();
        g.add_node(node("leaf")).unwrap();
        g.add_edge("root", "PARENT_OF", "mid", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("mid", "PARENT_OF", "leaf", None, EdgeOwnership::Reference).unwrap();

        let path = g.walk_ancestors("leaf", "PARENT_OF");
        let ids: Vec<&str> = path.iter().map(|n| n.id.as_str()).collect();

        assert_eq!(ids, vec!["root", "mid", "leaf"]);
    }

    #[test]
    fn walk_descendants_returns_start_to_leaf_inclusive() {
        let mut g = test_graph();
        g.add_node(node("root")).unwrap();
        g.add_node(node("mid")).unwrap();
        g.add_node(node("leaf")).unwrap();
        g.add_edge("root", "PARENT_OF", "mid", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("mid", "PARENT_OF", "leaf", None, EdgeOwnership::Reference).unwrap();

        let path = g.walk_descendants("root", "PARENT_OF");
        let ids: Vec<&str> = path.iter().map(|n| n.id.as_str()).collect();

        assert_eq!(ids, vec!["root", "mid", "leaf"]);
    }

    #[test]
    fn walk_ancestors_stops_at_a_node_with_no_matching_incoming_edge() {
        let mut g = test_graph();
        g.add_node(node("only")).unwrap();

        assert_eq!(g.walk_ancestors("only", "PARENT_OF").len(), 1);
        assert_eq!(g.walk_ancestors("only", "PARENT_OF")[0].id, "only");
    }

    #[test]
    fn walk_descendants_returns_empty_for_a_missing_start_node() {
        let g = test_graph();
        assert!(g.walk_descendants("missing", "PARENT_OF").is_empty());
    }

    #[test]
    fn walk_descendants_detects_a_cycle_and_terminates() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_edge("a", "NEXT", "b", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("b", "NEXT", "a", None, EdgeOwnership::Reference).unwrap();

        let path = g.walk_descendants("a", "NEXT");
        let ids: Vec<&str> = path.iter().map(|n| n.id.as_str()).collect();

        assert_eq!(ids, vec!["a", "b"], "must stop once it would revisit a");
    }

    #[test]
    fn walk_ancestors_only_follows_the_given_edge_type() {
        let mut g = test_graph();
        g.add_node(node("root")).unwrap();
        g.add_node(node("child")).unwrap();
        g.add_edge("root", "OTHER", "child", None, EdgeOwnership::Reference).unwrap();

        let path = g.walk_ancestors("child", "PARENT_OF");

        assert_eq!(path.len(), 1, "no PARENT_OF edge into child, so the walk stops at child");
        assert_eq!(path[0].id, "child");
    }

    #[test]
    fn query_plan_filters_by_node_type() {
        let mut g = test_graph();
        g.add_node(node_of_type("a", "doc")).unwrap();
        g.add_node(node_of_type("b", "article")).unwrap();

        let plan = QueryPlan { node_types: Some(vec!["doc".into()]), ..Default::default() };
        let ids = g.query_plan(&plan).unwrap();

        assert_eq!(ids, vec!["a".to_string()]);
    }

    #[test]
    fn query_plan_filters_by_attribute_eq() {
        use polypack_core::query::AttributeFilter;

        let mut g = test_graph();
        let mut a = node("a");
        a.data.insert("category".into(), serde_json::json!("science"));
        g.add_node(a).unwrap();
        let mut b = node("b");
        b.data.insert("category".into(), serde_json::json!("art"));
        g.add_node(b).unwrap();

        let plan = QueryPlan {
            attributes: Some(vec![AttributeFilter::Eq {
                field: "category".into(),
                value: serde_json::json!("science"),
            }]),
            ..Default::default()
        };
        let ids = g.query_plan(&plan).unwrap();

        assert_eq!(ids, vec!["a".to_string()]);
    }

    #[test]
    fn query_plan_filters_by_edge_filter_target() {
        use polypack_core::query::EdgeFilter;

        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_node(node("c")).unwrap();
        g.add_edge("a", "REL", "c", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("b", "OTHER", "c", None, EdgeOwnership::Reference).unwrap();

        let plan = QueryPlan {
            edge_filter: Some(EdgeFilter { edge_type: "REL".into(), target: Some("c".into()), source: None }),
            ..Default::default()
        };
        let ids = g.query_plan(&plan).unwrap();

        assert_eq!(ids, vec!["a".to_string()], "only a has a REL edge into c");
    }

    #[test]
    fn query_plan_respects_limit() {
        let mut g = test_graph();
        g.add_nodes(vec![node("a"), node("b"), node("c")]).unwrap();

        let plan = QueryPlan { limit: Some(2), ..Default::default() };
        assert_eq!(g.query_plan(&plan).unwrap().len(), 2);
    }

    #[test]
    fn aggregate_plan_sums_a_numeric_field_over_matched_nodes() {
        let mut g = test_graph();
        let mut a = node_of_type("a", "doc");
        a.data.insert("score".into(), serde_json::json!(3.0));
        g.add_node(a).unwrap();
        let mut b = node_of_type("b", "doc");
        b.data.insert("score".into(), serde_json::json!(4.0));
        g.add_node(b).unwrap();
        g.add_node(node_of_type("c", "article")).unwrap();

        let plan = QueryPlan { node_types: Some(vec!["doc".into()]), ..Default::default() };
        let (sum, count) = g.aggregate_plan(&plan, "score", "sum").unwrap();

        assert_eq!(sum, 7.0);
        assert_eq!(count, 2);
    }

    #[test]
    fn aggregate_plan_rejects_an_unknown_op() {
        let mut g = test_graph();
        let mut a = node("a");
        a.data.insert("score".into(), serde_json::json!(1.0));
        g.add_node(a).unwrap();

        assert!(g.aggregate_plan(&QueryPlan::default(), "score", "median").is_err());
    }

    #[test]
    fn flush_enforces_the_hot_cache_cap_via_eviction() {
        let mut g = small_cache_graph(2);
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_node(node("c")).unwrap();
        g.flush().unwrap();

        assert_eq!(g.size(), 2);
        assert!(!g.has_loaded_node("a"), "a is least-recently-touched and gets evicted");
        assert!(g.has_loaded_node("b"));
        assert!(g.has_loaded_node("c"));
        assert!(
            !g.evicted_dirty_nodes.contains_key("a"),
            "a was already flushed (clean) when evicted, so no snapshot is needed"
        );
    }

    #[test]
    fn touch_hot_cache_moves_a_node_to_the_recently_used_end() {
        let mut g = small_cache_graph(2);
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.get_node("a");
        g.add_node(node("c")).unwrap();
        g.flush().unwrap();

        assert!(g.has_loaded_node("a"), "a was re-touched via get_node, so b is evicted instead");
        assert!(!g.has_loaded_node("b"));
        assert!(g.has_loaded_node("c"));
    }

    #[test]
    fn eviction_snapshots_a_dirty_evictee_so_unflushed_edits_survive() {
        let mut g = small_cache_graph(1);
        g.add_node(node("a")).unwrap();
        let mut patch = serde_json::Map::new();
        patch.insert("title".into(), serde_json::json!("edited"));
        g.update_node("a", patch, None, None).unwrap();

        // Push the periodic eviction-skip counter (touch_hot_cache only
        // checks the cap every 10th touch) past its next multiple of 10
        // without ever calling flush, so "a"'s edit is still unflushed when
        // it gets evicted.
        for id in ["b", "c", "d", "e", "f", "g", "h", "i"] {
            g.add_node(node(id)).unwrap();
        }

        assert!(!g.has_loaded_node("a"), "a should have been evicted once the periodic check ran");
        let snapshot = g.evicted_dirty_nodes.get("a").expect("a's unflushed edit must be snapshotted");
        assert_eq!(snapshot.data.get("title"), Some(&serde_json::json!("edited")));

        let restored = g.get_node_safe("a").unwrap().unwrap();
        assert_eq!(
            restored.data.get("title"),
            Some(&serde_json::json!("edited")),
            "restore must recover the unflushed edit, not a stale/absent persisted copy"
        );
        assert!(!g.evicted_dirty_nodes.contains_key("a"), "get_node_safe consumes the snapshot");
    }

    #[test]
    fn save_persists_the_full_loaded_graph_without_clearing_dirty_state() {
        let mut g = test_graph();
        let mut a = node("a");
        a.vector = Some(vec![1.0, 0.0]);
        g.add_node(a).unwrap();
        g.add_node(node("b")).unwrap();
        g.add_edge("a", "REL", "b", None, EdgeOwnership::Owned).unwrap();

        g.save().unwrap();

        assert_eq!(g.store.get_node("a").unwrap().unwrap().id, "a");
        assert_eq!(g.store.get_vector("a").unwrap().unwrap(), vec![1.0, 0.0]);
        let edges = g.store.edges_snapshot().unwrap();
        assert!(edges.iter().any(|(id, _)| *id == edge_id("a", "REL", "b")));
        assert!(g.has_pending_persistence(), "save must not clear dirty-tracking state");
    }

    #[test]
    fn clear_resets_in_memory_state_without_touching_the_store() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.flush().unwrap();

        g.clear();

        assert_eq!(g.size(), 0);
        assert!(!g.has_pending_persistence());
        assert!(!g.has_loaded_node("a"));
        assert!(g.store.get_node("a").unwrap().is_some(), "clear must not touch the store");
    }

    #[test]
    fn clear_allows_warm_to_reload_from_the_store() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.flush().unwrap();
        g.warm().unwrap();
        assert!(g.has_loaded_node("a"));

        g.clear();
        assert!(!g.has_loaded_node("a"));

        g.warm().unwrap();
        assert!(g.has_loaded_node("a"), "clear() must reset `warmed` so warm() reloads");
    }

    #[test]
    fn dispose_flushes_before_closing() {
        let storage = shared_storage();
        {
            let mut g = graph_on(&storage, GraphConfig::default());
            g.add_node(node("a")).unwrap();
            g.dispose().unwrap();
        }

        let mut g2 = graph_on(&storage, GraphConfig::default());
        g2.warm().unwrap();

        assert!(g2.has_loaded_node("a"), "dispose must flush before closing");
    }

    #[test]
    fn dispose_clears_in_memory_state() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();

        g.dispose().unwrap();

        assert_eq!(g.size(), 0);
        assert!(!g.has_pending_persistence());
    }

    #[test]
    fn prune_removes_the_oldest_nodes_down_to_the_cap() {
        let mut g = test_graph();
        let mut old = node("old");
        old.inserted_at = 1;
        let mut mid = node("mid");
        mid.inserted_at = 2;
        let mut newest = node("newest");
        newest.inserted_at = 3;
        g.add_node(old).unwrap();
        g.add_node(mid).unwrap();
        g.add_node(newest).unwrap();

        g.prune(1).unwrap();

        assert_eq!(g.size(), 1);
        assert!(g.has_loaded_node("newest"));
        assert!(!g.has_loaded_node("old"));
        assert!(!g.has_loaded_node("mid"));
    }

    #[test]
    fn prune_is_a_noop_when_within_the_cap() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();

        g.prune(5).unwrap();

        assert!(g.has_loaded_node("a"));
    }

    #[test]
    fn prune_cascades_through_owned_edges_beyond_the_targeted_count() {
        let mut g = test_graph();
        let mut old = node("old");
        old.inserted_at = 1;
        let mut owned_child = node("owned_child");
        owned_child.inserted_at = 2;
        let mut newest = node("newest");
        newest.inserted_at = 3;
        g.add_node(old).unwrap();
        g.add_node(owned_child).unwrap();
        g.add_node(newest).unwrap();
        g.add_edge("old", "OWNS", "owned_child", None, EdgeOwnership::Owned).unwrap();

        g.prune(2).unwrap(); // directly targets only "old", the single oldest

        assert!(!g.has_loaded_node("old"));
        assert!(
            !g.has_loaded_node("owned_child"),
            "cascade-removed along with old, even though it wasn't directly targeted"
        );
        assert!(g.has_loaded_node("newest"));
    }

    #[test]
    fn remove_node_vector_strips_the_vector_but_keeps_the_node() {
        let mut g = test_graph();
        let mut a = node("a");
        a.vector = Some(vec![1.0, 0.0]);
        g.add_node(a).unwrap();
        g.flush().unwrap();

        let updated = g.remove_node_vector("a").unwrap();

        assert!(updated.vector.is_none());
        assert!(!g.hnsw.has("a"));
        assert!(g.has_loaded_node("a"));
        assert!(g.removed_vector_ids.contains("a"));
        assert!(g.dirty_nodes.contains("a"));
    }

    #[test]
    fn remove_node_vector_returns_none_for_a_node_not_in_the_hot_cache() {
        let mut g = test_graph();
        assert!(g.remove_node_vector("missing").is_none());
    }

    #[test]
    fn remove_node_vector_then_flush_deletes_the_persisted_vector() {
        let mut g = test_graph();
        let mut a = node("a");
        a.vector = Some(vec![1.0, 0.0]);
        g.add_node(a).unwrap();
        g.flush().unwrap();
        assert!(g.store.get_vector("a").unwrap().is_some());

        g.remove_node_vector("a").unwrap();
        g.flush().unwrap();

        assert!(g.store.get_vector("a").unwrap().is_none());
        assert!(g.store.get_node("a").unwrap().is_some(), "node itself remains persisted");
    }

    #[test]
    fn mark_vector_dirty_marks_a_vector_added_directly_through_vectors_mut() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.flush().unwrap();

        g.vectors_mut().add("a", &[1.0, 0.0]).unwrap();
        assert!(!g.dirty_vectors.contains("a"), "direct mutation bypasses dirty-tracking");

        g.mark_vector_dirty("a");
        assert!(g.dirty_vectors.contains("a"));

        g.flush().unwrap();
        assert_eq!(g.store.get_vector("a").unwrap().unwrap(), vec![1.0, 0.0]);
    }

    #[test]
    fn mark_vector_dirty_is_a_noop_for_an_id_not_in_the_vector_index() {
        let mut g = test_graph();
        g.mark_vector_dirty("missing");
        assert!(g.dirty_vectors.is_empty());
    }

    #[test]
    fn remove_node_vector_safe_restores_an_evicted_node_then_removes_its_vector() {
        let mut g = test_graph();
        let mut a = node("a");
        a.vector = Some(vec![1.0, 0.0]);
        g.add_node(a).unwrap();
        g.flush().unwrap();
        evict(&mut g, "a");

        let updated = g.remove_node_vector_safe("a").unwrap().unwrap();

        assert!(updated.vector.is_none());
        assert!(g.has_loaded_node("a"), "restored into the hot cache along the way");
        assert!(!g.hnsw.has("a"));
    }

    #[test]
    fn remove_node_vector_safe_returns_none_for_a_missing_node() {
        let mut g = test_graph();
        assert!(g.remove_node_vector_safe("missing").unwrap().is_none());
    }

    #[test]
    fn remove_node_vector_safe_returns_none_for_a_removed_node() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.flush().unwrap();
        g.remove_node("a").unwrap();
        g.flush().unwrap();

        assert!(g.remove_node_vector_safe("a").unwrap().is_none());
    }

    #[test]
    fn loaded_size_matches_size() {
        let mut g = test_graph();
        g.add_nodes(vec![node("a"), node("b")]).unwrap();

        assert_eq!(g.loaded_size(), g.size());
        assert_eq!(g.loaded_size(), 2);
    }

    #[test]
    fn persisted_size_counts_the_store_independent_of_the_hot_cache() {
        let mut g = small_cache_graph(1);
        g.add_node(node("a")).unwrap();
        g.add_node(node("b")).unwrap();
        g.flush().unwrap(); // evicts down to 1 hot node, but both are persisted

        assert_eq!(g.size(), 1);
        assert_eq!(g.persisted_size().unwrap(), 2);
    }

    #[test]
    fn load_is_an_alias_for_warm() {
        let storage = shared_storage();
        {
            let mut g = graph_on(&storage, GraphConfig::default());
            g.add_node(node("a")).unwrap();
            g.flush().unwrap();
        }

        let mut g2 = graph_on(&storage, GraphConfig::default());
        g2.load().unwrap();

        assert!(g2.has_loaded_node("a"));
    }

    fn recording_graph() -> (Graph, std::rc::Rc<std::cell::RefCell<Vec<GraphChangeEvent>>>) {
        let mut g = test_graph();
        let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let events_cb = events.clone();
        g.on_change(move |ev| events_cb.borrow_mut().push(ev));
        (g, events)
    }

    #[test]
    fn emit_dispatches_immediately_outside_a_batch() {        let (mut g, events) = recording_graph();

        g.add_node(node("a")).unwrap();

        assert_eq!(events.borrow().len(), 1);
    }

    #[test]
    fn batching_defers_events_until_end_batch() {
        let (mut g, events) = recording_graph();

        g.start_batch();
        g.add_node(node("a")).unwrap();
        assert!(events.borrow().is_empty(), "event must be queued while the batch is open");

        g.end_batch().unwrap();
        assert_eq!(events.borrow().len(), 1);
    }

    #[test]
    fn nested_batches_only_flush_once_the_outermost_ends() {
        let (mut g, events) = recording_graph();

        g.start_batch();
        g.start_batch();
        g.add_node(node("a")).unwrap();
        g.end_batch().unwrap();
        assert!(events.borrow().is_empty(), "inner end_batch must not flush yet");

        g.end_batch().unwrap();
        assert_eq!(events.borrow().len(), 1);
    }

    #[test]
    fn end_batch_without_start_batch_errors() {
        let mut g = test_graph();
        assert!(g.end_batch().is_err());
    }

    #[test]
    fn add_nodes_coalesces_events_into_one_batch() {
        let (mut g, events) = recording_graph();

        g.add_nodes(vec![node("a"), node("b"), node("c")]).unwrap();

        assert_eq!(events.borrow().len(), 3, "all node_added events fire once add_nodes's batch ends");
    }

    // ── GraphQuery / PersistedGraphQuery integration (unit coverage for the
    // matching/traversal/aggregation logic itself lives in query.rs) ──

    use crate::query::OrderDirection;
    use polypack_core::query::Direction;

    fn persisted_library() -> Graph {
        let mut g = test_graph();
        g.add_node(node_of_type("alice", "user")).unwrap();
        g.add_node(node_of_type("bob", "user")).unwrap();
        let mut scifi1 = node_of_type("scifi1", "book");
        scifi1.data.insert("genre".into(), serde_json::json!("sci-fi"));
        scifi1.data.insert("rating".into(), serde_json::json!(5.0));
        scifi1.vector = Some(vec![1.0, 0.0]);
        g.add_node(scifi1).unwrap();
        let mut fantasy1 = node_of_type("fantasy1", "book");
        fantasy1.data.insert("genre".into(), serde_json::json!("fantasy"));
        fantasy1.data.insert("rating".into(), serde_json::json!(3.0));
        fantasy1.vector = Some(vec![0.0, 1.0]);
        g.add_node(fantasy1).unwrap();
        g.add_edge("alice", "RATED", "scifi1", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("alice", "RATED", "fantasy1", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("bob", "RATED", "scifi1", None, EdgeOwnership::Reference).unwrap();
        g.flush().unwrap();
        g
    }

    fn ids_of(mut nodes: Vec<Node>) -> Vec<String> {
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        nodes.into_iter().map(|n| n.id).collect()
    }

    #[test]
    fn persisted_query_filters_by_node_type_and_attribute() {
        let mut g = persisted_library();
        let mut q = g.query_persisted().where_node_type(vec!["book".into()]).where_field("genre", serde_json::json!("sci-fi"));
        assert_eq!(ids_of(q.to_array().unwrap()), vec!["scifi1"]);
    }

    #[test]
    fn persisted_query_where_edge_and_edge_source() {
        let mut g = persisted_library();
        assert_eq!(
            ids_of(g.query_persisted().where_edge("RATED", Some("fantasy1")).to_array().unwrap()),
            vec!["alice"]
        );
        assert_eq!(ids_of(g.query_persisted().where_edge_source("bob").to_array().unwrap()), vec!["scifi1"]);
    }

    #[test]
    fn persisted_query_traverse_expands_from_seeds() {
        let mut g = persisted_library();

        // A base filter matching nothing leaves traversal with no seeds to
        // expand from — filters apply before traversal, not after.
        let expanded = g
            .query_persisted()
            .where_node_type(vec!["user".into()])
            .where_field("no_such_field", serde_json::json!(true))
            .traverse("RATED", 1, Direction::Out)
            .to_array()
            .unwrap();
        assert!(expanded.is_empty());

        let expanded =
            g.query_persisted().where_node_type(vec!["user".into()]).traverse("RATED", 1, Direction::Out).to_array().unwrap();
        assert_eq!(ids_of(expanded), vec!["alice", "bob", "fantasy1", "scifi1"]);
    }

    #[test]
    fn persisted_query_enforces_resource_limits() {
        let mut g = persisted_library();
        let limited = g
            .query_persisted_with_limits(crate::QueryResourceLimits { max_results: Some(1), ..Default::default() })
            .where_node_type(vec!["book".into()])
            .to_array();
        assert!(matches!(limited, Err(PolypackError::ResourceLimit { .. })));

        let limited = g
            .query_persisted_with_limits(crate::QueryResourceLimits { max_traversal_depth: Some(0), ..Default::default() })
            .where_node_type(vec!["user".into()])
            .traverse("RATED", 1, Direction::Out)
            .to_array();
        assert!(matches!(limited, Err(PolypackError::ResourceLimit { .. })));
    }

    #[test]
    fn persisted_query_join_filters_by_connected_node() {
        let mut g = persisted_library();
        let results = g
            .query_persisted()
            .where_node_type(vec!["user".into()])
            .join(
                "RATED",
                Direction::Out,
                Some(Box::new(|n: &Node| n.data.get("genre") == Some(&serde_json::json!("fantasy")))),
            )
            .to_array()
            .unwrap();
        assert_eq!(ids_of(results), vec!["alice"]);
    }

    #[test]
    fn persisted_query_order_by_offset_and_limit() {
        let mut g = persisted_library();
        let results = g
            .query_persisted()
            .where_node_type(vec!["book".into()])
            .order_by("rating", OrderDirection::Asc)
            .offset(1)
            .limit(1)
            .to_array()
            .unwrap();
        assert_eq!(ids_of(results), vec!["scifi1"]);
    }

    #[test]
    fn persisted_query_similar_to_ranks_and_filters() {
        let mut g = persisted_library();
        let results = g
            .query_persisted()
            .where_node_type(vec!["book".into()])
            .similar_to(vec![1.0, 0.0], 0.5, Some(1))
            .to_array()
            .unwrap();
        assert_eq!(ids_of(results), vec!["scifi1"]);
    }

    #[test]
    fn persisted_query_count_first_and_ids() {
        let mut g = persisted_library();
        assert_eq!(g.query_persisted().where_node_type(vec!["book".into()]).count().unwrap(), 2);
        let mut ids = g.query_persisted().where_node_type(vec!["book".into()]).ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["fantasy1".to_string(), "scifi1".to_string()]);
        assert!(g.query_persisted().where_node_type(vec!["book".into()]).first().unwrap().is_some());
        assert!(g.query_persisted().where_node_type(vec!["missing".into()]).first().unwrap().is_none());
    }

    #[test]
    fn persisted_query_collect_gathers_connected_nodes() {
        let mut g = persisted_library();
        let mut ids =
            ids_of(g.query_persisted().where_node_type(vec!["user".into()]).collect("RATED", Direction::Out, None).unwrap());
        ids.sort();
        assert_eq!(ids, vec!["fantasy1".to_string(), "scifi1".to_string()]);
    }

    #[test]
    fn embed_uses_the_configured_provider() {
        let g = test_graph();
        let a = g.embed("hello world").unwrap();
        let b = g.embed("hello world").unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 384, "default provider is the 384-dim FeatureHashEmbedding");
    }

    #[test]
    fn add_node_with_embedding_sets_the_vector_and_ignores_any_existing_one() {
        let mut g = test_graph();
        let mut n = node("a");
        n.vector = Some(vec![9.0, 9.0]); // must be overwritten
        let expected = g.embed("hello world").unwrap();

        g.add_node_with_embedding(n, "hello world").unwrap();

        assert_eq!(g.get_node("a").unwrap().vector, Some(expected));
    }

    #[test]
    fn add_node_with_embedding_uses_a_custom_provider() {
        use crate::embedding::{FeatureHashEmbedding, FeatureHashEmbeddingOptions};

        let mut g = Graph::open(
            Box::new(InMemoryStorage::new()),
            StoreConfig::default(),
            GraphConfig {
                embedding: Box::new(FeatureHashEmbedding::new(FeatureHashEmbeddingOptions { dimensions: 8 }).unwrap()),
                ..GraphConfig::default()
            },
        )
        .unwrap();

        g.add_node_with_embedding(node("a"), "hello world").unwrap();

        assert_eq!(g.get_node("a").unwrap().vector.as_ref().unwrap().len(), 8);
    }

    #[test]
    fn update_node_safe_updates_a_loaded_node() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();

        let mut patch = serde_json::Map::new();
        patch.insert("title".into(), serde_json::json!("hello"));
        let updated = g.update_node_safe("a", patch, None, None).unwrap().unwrap();

        assert_eq!(updated.data.get("title"), Some(&serde_json::json!("hello")));
    }

    #[test]
    fn update_node_safe_restores_an_evicted_node_then_updates_it() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.flush().unwrap();
        evict(&mut g, "a");

        let mut patch = serde_json::Map::new();
        patch.insert("title".into(), serde_json::json!("hello"));
        let updated = g.update_node_safe("a", patch, None, None).unwrap().unwrap().clone();

        assert!(g.has_loaded_node("a"), "restored into the hot cache along the way");
        assert_eq!(updated.data.get("title"), Some(&serde_json::json!("hello")));
    }

    #[test]
    fn update_node_safe_returns_none_for_a_missing_node() {
        let mut g = test_graph();
        let mut patch = serde_json::Map::new();
        patch.insert("title".into(), serde_json::json!("hello"));
        assert!(g.update_node_safe("missing", patch, None, None).unwrap().is_none());
    }

    #[test]
    fn update_node_with_embedding_sets_data_and_vector() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        let expected = g.embed("hello world").unwrap();

        let mut patch = serde_json::Map::new();
        patch.insert("title".into(), serde_json::json!("hello"));
        let updated = g.update_node_with_embedding("a", patch, "hello world").unwrap().unwrap();

        assert_eq!(updated.data.get("title"), Some(&serde_json::json!("hello")));
        assert_eq!(updated.vector, Some(expected));
    }

    #[test]
    fn update_node_safe_with_embedding_restores_an_evicted_node_then_updates_it() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.flush().unwrap();
        evict(&mut g, "a");
        let expected = g.embed("hello world").unwrap();

        let mut patch = serde_json::Map::new();
        patch.insert("title".into(), serde_json::json!("hello"));
        let updated = g.update_node_safe_with_embedding("a", patch, "hello world").unwrap().unwrap().clone();

        assert!(g.has_loaded_node("a"));
        assert_eq!(updated.data.get("title"), Some(&serde_json::json!("hello")));
        assert_eq!(updated.vector, Some(expected));
    }

    #[test]
    fn update_node_safe_with_embedding_returns_none_for_a_missing_node() {
        let mut g = test_graph();
        let mut patch = serde_json::Map::new();
        patch.insert("title".into(), serde_json::json!("hello"));
        assert!(g.update_node_safe_with_embedding("missing", patch, "text").unwrap().is_none());
    }

    #[test]
    fn query_text_ranks_by_embedded_similarity() {
        let mut g = test_graph();
        g.add_node_with_embedding(node("a"), "hello world").unwrap();
        g.add_node_with_embedding(node("b"), "goodbye moon").unwrap();

        let results = g.query_text("hello world", 0.99, None).unwrap().to_array();

        assert_eq!(ids_of(results), vec!["a"], "only the identically-embedded text clears a 0.99 threshold");
    }

    #[test]
    fn query_text_returns_a_chainable_graphquery() {
        let mut g = test_graph();
        g.add_node_with_embedding(node_of_type("a", "doc"), "hello world").unwrap();
        g.add_node_with_embedding(node_of_type("b", "other"), "hello world").unwrap();

        let results = g.query_text("hello world", 0.0, None).unwrap().where_node_type(vec!["doc".into()]).to_array();

        assert_eq!(ids_of(results), vec!["a"]);
    }

    #[test]
    fn query_persisted_text_ranks_by_embedded_similarity() {
        let mut g = test_graph();
        g.add_node_with_embedding(node("a"), "hello world").unwrap();
        g.add_node_with_embedding(node("b"), "goodbye moon").unwrap();
        g.flush().unwrap();

        let results = g.query_persisted_text("hello world", 0.99, None).unwrap().to_array().unwrap();

        assert_eq!(ids_of(results), vec!["a"]);
    }

    #[test]
    fn search_nodes_filters_by_type_and_similarity() {
        let mut g = test_graph();
        g.add_node_with_embedding(node_of_type("a", "doc"), "hello world").unwrap();
        g.add_node_with_embedding(node_of_type("b", "other"), "hello world").unwrap();
        g.flush().unwrap();

        let results = g.search_nodes("hello world", "doc", 0.99, None).unwrap();

        assert_eq!(ids_of(results), vec!["a"], "b is excluded despite matching text, wrong node type");
    }

    #[test]
    fn search_nodes_respects_the_similarity_threshold() {
        let mut g = test_graph();
        g.add_node_with_embedding(node_of_type("a", "doc"), "hello world").unwrap();
        g.flush().unwrap();

        let results = g.search_nodes("completely unrelated text", "doc", 0.9, None).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn search_nodes_returns_empty_for_blank_text_without_embedding() {
        let mut g = test_graph();
        assert!(g.search_nodes("   ", "doc", 0.0, None).unwrap().is_empty());
    }

    // ── activation ──

    #[test]
    fn reinforce_node_sets_activation_and_emits_an_activation_updated_event() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let events_cb = events.clone();
        g.on_change(move |e| events_cb.borrow_mut().push(e));

        g.reinforce_node("a", 0.5, Some("user_read")).unwrap();

        // Decay is re-evaluated against a fresh wall-clock read here, so allow
        // for the (small) real elapsed time since reinforce_node wrote it.
        let state = g.get_activation_state("a").unwrap();
        assert!((state.score - 0.5).abs() < 1e-5);
        assert!((state.importance - 0.025).abs() < 1e-5);
        assert_eq!(state.reinforcement_count, 1);
        assert!((g.get_activation("a", DEFAULT_ACTIVATION.score_half_life_ms) - 0.5).abs() < 1e-5);
        assert_eq!(
            events.borrow().as_slice(),
            &[GraphChangeEvent::ActivationUpdated {
                node_id: "a".into(),
                node_type: "doc".into(),
                delta: 0.5,
                reason: Some("user_read".into()),
            }]
        );
    }

    #[test]
    fn reinforce_node_clamps_and_accumulates() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.reinforce_node("a", 5.0, None).unwrap();
        assert!((g.get_activation_state("a").unwrap().score - 1.0).abs() < 1e-5);
        g.reinforce_node("a", 0.2, None).unwrap();
        g.reinforce_node("a", 0.3, None).unwrap();
        let state = g.get_activation_state("a").unwrap();
        assert_eq!(state.reinforcement_count, 3);
    }

    #[test]
    fn reinforce_node_decay_corrects_before_adding() {
        let mut g = test_graph();
        let now = now_millis();
        let mut n = node("a");
        n.activation = Some(NodeActivation {
            score: 1.0,
            importance: 0.5,
            reinforcement_count: 1,
            last_meaningful_activation: now - DEFAULT_ACTIVATION.score_half_life_ms,
        });
        g.add_node(n).unwrap();
        g.reinforce_node("a", 0.5, None).unwrap();
        let state = g.get_activation_state("a").unwrap();
        // Decayed to 0.5, then +0.5 → 1.0, re-anchored to now. Allow for the
        // real elapsed time since this fresh wall-clock read re-decays it.
        assert!((state.score - 1.0).abs() < 1e-5);
        assert_eq!(state.last_meaningful_activation, now_millis());
    }

    #[test]
    fn reinforce_node_missing_or_unloaded_returns_none() {
        let mut g = test_graph();
        assert!(g.reinforce_node("missing", 0.5, None).unwrap().is_none());
        assert_eq!(g.get_activation("missing", DEFAULT_ACTIVATION.score_half_life_ms), 0.0);
    }

    #[test]
    fn activation_persists_through_flush_and_warm() {
        let storage = shared_storage();
        {
            let mut g = graph_on(&storage, GraphConfig::default());
            g.add_node(node("a")).unwrap();
            g.reinforce_node("a", 0.6, None).unwrap();
            g.flush().unwrap();
            g.dispose().unwrap();
        }

        let mut g2 = graph_on(&storage, GraphConfig::default());
        g2.warm().unwrap();
        let state = g2.get_activation_state("a").unwrap();
        assert!((state.score - 0.6).abs() < 1e-5);
        assert_eq!(state.reinforcement_count, 1);
    }

    #[test]
    fn reinforce_node_safe_restores_an_evicted_node() {
        let mut g = Graph::open(
            Box::new(InMemoryStorage::new()),
            StoreConfig::default(),
            GraphConfig { hot_cache_max: 1, ..GraphConfig::default() },
        )
        .unwrap();
        g.add_node(node("a")).unwrap();
        g.reinforce_node("a", 0.4, None).unwrap();
        g.flush().unwrap();
        g.add_node(node("b")).unwrap();
        g.flush().unwrap();
        assert!(!g.has_loaded_node("a"));

        g.reinforce_node_safe("a", 0.3, None).unwrap().unwrap();
        let state = g.get_activation_state("a").unwrap();
        assert!((state.score - 0.7).abs() < 1e-5);
        assert!(g.has_loaded_node("a"));
    }

    #[test]
    fn top_activated_ranks_by_current_score_descending() {
        let mut g = test_graph();
        for (id, amount) in [("a", 0.9), ("b", 0.5), ("c", 0.1)] {
            g.add_node(node(id)).unwrap();
            g.reinforce_node(id, amount, None).unwrap();
        }
        let top: Vec<String> = g.top_activated(2, 0.0).into_iter().map(|n| n.id.clone()).collect();
        assert_eq!(top, vec!["a".to_string(), "b".to_string()]);
        let hot: Vec<String> = g.top_activated(10, 0.6).into_iter().map(|n| n.id.clone()).collect();
        assert_eq!(hot, vec!["a".to_string()]);
    }

    #[test]
    fn persisted_query_activation_filters_and_orders() {
        let mut g = test_graph();
        let now = now_millis();
        for (id, score) in [("a", 0.9), ("b", 0.5), ("c", 0.1)] {
            let mut n = node(id);
            n.activation = Some(NodeActivation {
                score,
                importance: 0.0,
                reinforcement_count: 1,
                last_meaningful_activation: now,
            });
            g.add_node(n).unwrap();
        }
        g.flush().unwrap();

        let mut high = g.query_persisted().where_activated(0.4).ids().unwrap();
        high.sort();
        assert_eq!(high, vec!["a".to_string(), "b".to_string()]);
        let ordered = g
            .query_persisted()
            .order_by_activation(crate::query::OrderDirection::Desc)
            .ids()
            .unwrap();
        assert_eq!(ordered, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }
}
