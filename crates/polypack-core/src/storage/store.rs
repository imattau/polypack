//! The persistence state machine, mirroring the TypeScript `BinaryStoreAdapter`
//! semantics: serialised load/apply/compact/close, WAL replay then
//! snapshot-before-WAL-delete, generation-boundary compaction, truncated-tail
//! tolerance, and version checking. Hosts own byte I/O via the `Storage` trait.

use crate::error::{PolypackError, Result};
use crate::model::{ChangeBatch, Edge, Node};
use crate::storage::format::{decode_snapshot, decode_wal, encode_snapshot, encode_wal};
use crate::storage::wal::WalEntry;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const SNAPSHOT_FILE: &str = "snapshot.msgpack";
pub const WAL_FILE: &str = "wal.msgpack";
pub const DEFAULT_COMPACT_THRESHOLD: usize = 10_000;
/// Compact once the WAL holds at least this share of the store's record count.
const COMPACT_RATIO: usize = 4;

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
}

fn node_field(node: &Node, field: &str) -> serde_json::Value {
    if field == "type" {
        serde_json::Value::String(node.node_type.clone())
    } else {
        node.data.get(field).cloned().unwrap_or(serde_json::Value::Null)
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
            let Some(value) = node.data.get(key).and_then(|v| v.as_f64()) else {
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
        let mut existing = self.files.get(name).cloned().unwrap_or_default();
        existing.extend_from_slice(data);
        self.files.insert(name.to_string(), existing);
        Ok(())
    }
    fn delete(&mut self, name: &str) -> Result<()> {
        self.files.remove(name);
        Ok(())
    }
    fn exists(&self, name: &str) -> Result<bool> {
        Ok(self.files.contains_key(name))
    }
}

pub struct StoreConfig {
    /// Minimum WAL-entry count at which compaction is scheduled. Acts as a
    /// lower bound — the effective threshold also grows with the store
    /// (`max(compact_threshold, records / COMPACT_RATIO)`), so a large store
    /// doesn't rewrite its snapshot on every batch. Default 10,000.
    pub compact_threshold: usize,
    pub durability: Durability,
}

impl Default for StoreConfig {
    fn default() -> Self {
        StoreConfig {
            compact_threshold: DEFAULT_COMPACT_THRESHOLD,
            durability: Durability::Process,
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
    edges_by_source: HashMap<String, HashSet<String>>,
    edges_by_target: HashMap<String, HashSet<String>>,
    wal_entry_count: usize,
    config: StoreConfig,
    storage: Box<dyn Storage>,
    closed: bool,
    loaded: bool,
}

impl Store {
    pub fn new(storage: Box<dyn Storage>, config: StoreConfig) -> Self {
        Store {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            vectors: HashMap::new(),
            by_type: HashMap::new(),
            edges_by_source: HashMap::new(),
            edges_by_target: HashMap::new(),
            wal_entry_count: 0,
            config,
            storage,
            closed: false,
            loaded: false,
        }
    }

    // ── secondary indexes ──

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
    }

    fn unindex_node(&mut self, id: &str) {
        let type_key = self.nodes.get(id).map(|n| n.node_type.clone());
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
        self.ensure_loaded()?;
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
        let candidate_ids: Vec<String> = match self.type_only_ids(query) {
            Some(ids) => ids.into_iter().collect(),
            None => self.nodes.keys().cloned().collect(),
        };
        let mut results: Vec<Node> = candidate_ids
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .filter(|n| matches_node(n, query))
            .cloned()
            .collect();
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
        let no_filters =
            query.node_types.is_none() && query.attributes.is_none() && query.attribute_ranges.is_none();
        let count = if no_filters {
            self.nodes.len()
        } else if let Some(ids) = self.type_only_ids(query) {
            ids.len()
        } else {
            self.nodes.values().filter(|n| matches_node(n, query)).count()
        };
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
        let n1 = Node { id: "a".into(), node_type: "draft".into(), data: Default::default(), vector: None, inserted_at: 1, updated_at: 1 };
        let n2 = Node { id: "a".into(), node_type: "published".into(), data: Default::default(), vector: None, inserted_at: 2, updated_at: 2 };
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
