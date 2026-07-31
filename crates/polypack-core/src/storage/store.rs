//! The persistence state machine, mirroring the TypeScript `BinaryStoreAdapter`
//! semantics: serialised load/apply/compact/close, WAL replay then
//! snapshot-before-WAL-delete, generation-boundary compaction, truncated-tail
//! tolerance, and version checking. Hosts own byte I/O via the `Storage` trait.

use crate::error::{PolypackError, Result};
use crate::model::{ChangeBatch, Edge, Node};
use crate::storage::format::{decode_snapshot, decode_wal, encode_snapshot, encode_wal};
use crate::storage::wal::WalEntry;
use std::collections::HashMap;

pub const SNAPSHOT_FILE: &str = "snapshot.msgpack";
pub const WAL_FILE: &str = "wal.msgpack";
pub const DEFAULT_COMPACT_THRESHOLD: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    Memory,
    Process,
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

pub struct Store {
    nodes: HashMap<String, Node>,
    edges: HashMap<String, Edge>,
    vectors: HashMap<String, Vec<f64>>,
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
            wal_entry_count: 0,
            config,
            storage,
            closed: false,
            loaded: false,
        }
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
                self.nodes.insert(node.id.clone(), node);
            }
            WalEntry::DeleteNode(id) => {
                self.nodes.remove(&id);
            }
            WalEntry::PutEdge(edge) => {
                self.edges.insert(edge.id.clone(), edge);
            }
            WalEntry::DeleteEdge(id) => {
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
    pub fn apply(&mut self, changes: &ChangeBatch) -> Result<()> {
        self.ensure_loaded()?;
        let mut entries: Vec<WalEntry> = Vec::new();
        for id in &changes.delete_node_ids {
            self.nodes.remove(id);
            entries.push(WalEntry::DeleteNode(id.clone()));
        }
        for id in &changes.delete_edge_ids {
            self.edges.remove(id);
            entries.push(WalEntry::DeleteEdge(id.clone()));
        }
        for id in &changes.delete_vector_ids {
            self.vectors.remove(id);
            entries.push(WalEntry::DeleteVector(id.clone()));
        }
        for node in &changes.put_nodes {
            self.nodes.insert(node.id.clone(), node.clone());
            entries.push(WalEntry::PutNode(node.clone()));
        }
        for edge in &changes.put_edges {
            self.edges.insert(edge.id.clone(), edge.clone());
            entries.push(WalEntry::PutEdge(edge.clone()));
        }
        for v in &changes.put_vectors {
            self.vectors.insert(v.id.clone(), v.vector.clone());
            entries.push(WalEntry::PutVector { id: v.id.clone(), vector: v.vector.clone() });
        }
        if !entries.is_empty() {
            let encoded = encode_wal(&entries);
            self.storage.append(WAL_FILE, &encoded)?;
            if self.config.durability == Durability::Fsync {
                self.storage.sync(WAL_FILE)?;
            }
            self.wal_entry_count += entries.len();
            if self.wal_entry_count >= self.config.compact_threshold {
                self.compact()?;
            }
        }
        Ok(())
    }

    pub fn clear_all(&mut self) -> Result<()> {
        self.ensure_loaded()?;
        self.nodes.clear();
        self.edges.clear();
        self.vectors.clear();
        self.wal_entry_count = 0;
        self.write_snapshot()?;
        self.storage.write(WAL_FILE, &[])?;
        Ok(())
    }

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
}
