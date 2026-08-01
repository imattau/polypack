//! [`Graph`]: the Rust counterpart to `PolyGraph` (`src/graph.ts`).

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use polypack_core::model::{edge_id, validate_node};
use polypack_core::storage::NodeQuery;
use polypack_core::{
    aggregate as core_aggregate, execute as core_execute, ChangeBatch, Edge, GraphSnapshot, HnswConfig,
    HnswIndex, Node, PolypackError, QueryPlan, Result, Storage, Store, StoreConfig,
};

use crate::edge::{decode_ownership, encode_ownership, EdgeEntry, EdgeOwnership};
use crate::event::GraphChangeEvent;
use crate::lru::LruList;

/// Tuning knobs, mirrors the `PolyGraph` constructor's `hotCacheMax` plus the
/// `HnswIndex` config threaded through `createVectorIndex` in the TS version.
pub struct GraphConfig {
    pub hot_cache_max: usize,
    pub hnsw: HnswConfig,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self { hot_cache_max: 50_000, hnsw: HnswConfig::default() }
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

    on_change: Option<Box<dyn FnMut(GraphChangeEvent)>>,
    on_orphan: Option<Box<dyn FnMut(&str)>>,

    warmed: bool,
}

impl Graph {
    /// Open a graph over `storage`, creating the backing [`Store`].
    pub fn open(storage: Box<dyn Storage>, store_config: StoreConfig, config: GraphConfig) -> Result<Self> {
        let store = Store::new(storage, store_config);
        let hnsw = HnswIndex::new(config.hnsw.clone(), 0);
        Ok(Self {
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
            warmed: false,
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

    // ── lifecycle ──

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
    /// needs `touch_hot_cache`'s eviction (still a stub) to run.
    pub fn warm(&mut self) -> Result<()> {
        if self.warmed {
            return Ok(());
        }
        self.warmed = true;

        let mut all_nodes = self.store.query_nodes(&NodeQuery::default())?;
        if all_nodes.is_empty() {
            return Ok(());
        }
        all_nodes.sort_by(|a, b| b.inserted_at.cmp(&a.inserted_at));
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
            if id != edge_id(&edge.source, &edge.edge_type, &edge.target) {
                return Err(PolypackError::CorruptData(format!("invalid persisted edge id: {id}")));
            }
            let (ownership, data) = decode_ownership(edge.data);
            let inner = format!("{}::{}", edge.edge_type, edge.target);
            self.edges.entry(edge.source.clone()).or_default().insert(
                inner,
                EdgeEntry { target: edge.target.clone(), edge_type: edge.edge_type, data, ownership },
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
                let (source, inner) = id.split_once("::")?;
                let entry = self.edges.get(source)?.get(inner)?;
                Some(Edge {
                    id: id.clone(),
                    source: source.to_string(),
                    target: entry.target.clone(),
                    edge_type: entry.edge_type.clone(),
                    data: encode_ownership(entry.data.clone(), entry.ownership),
                    created_at: now_millis(),
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

        if let Err(err) = self.store.apply(&changes) {
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
                    id: edge_id(source, &entry.edge_type, &entry.target),
                    source: source.clone(),
                    target: entry.target.clone(),
                    edge_type: entry.edge_type.clone(),
                    data: encode_ownership(entry.data.clone(), entry.ownership),
                    created_at: now_millis(),
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
        validate_node(&node)?;
        self.insert_node(node);
        Ok(())
    }

    /// Add several nodes in one call. All nodes are validated before any are
    /// inserted (an invalid entry inserts nothing), matching
    /// `PolyGraph.addNodes`.
    ///
    /// TS coalesces change-event notifications for the whole batch via
    /// `startBatch`/`endBatch`; this port has no batching mechanism yet (see
    /// the `on_change` docs), so each insert emits immediately.
    pub fn add_nodes(&mut self, nodes: Vec<Node>) -> Result<()> {
        for n in &nodes {
            validate_node(n)?;
        }
        for n in nodes {
            self.insert_node(n);
        }
        Ok(())
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
    /// vector. A no-op (returns `None`) if the node isn't loaded — see
    /// `update_node_safe`. Mirrors `PolyGraph.updateNode`; the data-transform
    /// serialize/deserialize hooks aren't part of this crate, so `data` is
    /// merged as-is.
    pub fn update_node(
        &mut self,
        id: &str,
        data: serde_json::Map<String, serde_json::Value>,
        vector: Option<Vec<f64>>,
    ) -> Result<Option<&Node>> {
        if !self.nodes.contains_key(id) {
            return Ok(None);
        }
        if let Some(vector) = &vector {
            if !vector.iter().all(|x| x.is_finite()) {
                return Err(PolypackError::InvalidArgument("vector must contain finite values".into()));
            }
        }

        let node_type = {
            let node = self.nodes.get_mut(id).unwrap();
            node.data.extend(data);
            if let Some(vector) = &vector {
                node.vector = Some(vector.clone());
            }
            node.updated_at = now_millis();
            node.node_type.clone()
        };

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
        let mut visited = HashSet::new();
        self.remove_node_cascade(id, &mut visited)
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
        let id = edge_id(source, &edge.edge_type, &edge.target);
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
        if source.is_empty() || edge_type.is_empty() || target.is_empty() {
            return Err(PolypackError::InvalidArgument(
                "edge source, type, and target must not be empty".into(),
            ));
        }
        let id = edge_id(source, edge_type, target);
        self.removed_edge_ids.remove(&id);
        let inner = format!("{edge_type}::{target}");
        let source_edges = self.edges.entry(source.to_string()).or_default();
        if source_edges.contains_key(&inner) {
            return Ok(());
        }
        source_edges.insert(
            inner,
            EdgeEntry { target: target.to_string(), edge_type: edge_type.to_string(), data, ownership },
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
    /// The owned-edge cascade calls `remove_node`, which is still a stub —
    /// this path will panic until that port lands.
    pub fn remove_edges(&mut self, source: &str, edge_type: Option<&str>, target: Option<&str>) -> Result<()> {
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
                let inner = format!("{}::{}", edge.edge_type, edge.target);
                if let Some(source_edges) = self.edges.get_mut(source) {
                    source_edges.remove(&inner);
                }
            }
            if let Some(set) = self.node_to_edge.get_mut(&edge.target) {
                set.remove(source);
            }
            let id = edge_id(source, &edge.edge_type, &edge.target);
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

        if remove_all {
            self.edges.remove(source);
        } else if self.edges.get(source).is_some_and(|m| m.is_empty()) {
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

    /// Execute a `QueryPlan` over the current hot working set, matching
    /// `PersistedGraphQuery`'s in-memory counterpart in `graph.ts` (there,
    /// `GraphQuery` walks `this.nodes`/`this.edges` directly; here the same
    /// hot state is handed to `polypack_core::query_exec::execute`, which
    /// implements the identical pipeline). The hot `HnswIndex` is always
    /// passed through so a plan's `similarity.engine: "hnsw"` can use it.
    pub fn query(&self, plan: &QueryPlan) -> Result<Vec<String>> {
        core_execute(&self.snapshot(), plan, Some(&self.hnsw))
    }

    /// Aggregate a numeric field over the nodes a `QueryPlan` selects from
    /// the hot working set. See `query`'s docs on scope.
    pub fn aggregate(&self, plan: &QueryPlan, field: &str, op: &str) -> Result<(f64, usize)> {
        core_aggregate(&self.snapshot(), plan, field, op)
    }

    /// Build a `query_exec::GraphSnapshot` from the current hot node/edge
    /// working set. `created_at` on the reconstructed `Edge`s is a
    /// placeholder — `query_exec` never reads it, only `edge_type`/
    /// `source`/`target`/`data`.
    fn snapshot(&self) -> GraphSnapshot {
        let nodes: Vec<Node> = self.nodes.values().cloned().collect();
        let mut edges = Vec::new();
        for (source, inner) in &self.edges {
            for entry in inner.values() {
                edges.push(Edge {
                    id: edge_id(source, &entry.edge_type, &entry.target),
                    source: source.clone(),
                    target: entry.target.clone(),
                    edge_type: entry.edge_type.clone(),
                    data: entry.data.clone(),
                    created_at: 0,
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

    // ── internal ──

    /// Mirrors `PolyGraph.insertNode`: assumes the node has already passed
    /// `validate_node`.
    fn insert_node(&mut self, node: Node) {
        let id = node.id.clone();
        let node_type = node.node_type.clone();
        let vector = node.vector.clone();

        if self.nodes.contains_key(&id) {
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
    }

    /// Mirrors `PolyGraph.unindexNode`.
    fn unindex_node(&mut self, id: &str) {
        let Some(node) = self.nodes.get(id) else { return };
        if let Some(set) = self.by_type.get_mut(&node.node_type) {
            set.remove(id);
            if set.is_empty() {
                let node_type = node.node_type.clone();
                self.by_type.remove(&node_type);
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
        if self.eviction_skip_counter % 10 == 0 {
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

    fn emit(&mut self, event: GraphChangeEvent) {
        if let Some(cb) = self.on_change.as_mut() {
            cb(event);
        }
    }
}

fn now_millis() -> i64 {
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
        }
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
            EdgeEntry { target: "b".into(), edge_type: "REL".into(), data: None, ownership: EdgeOwnership::Reference },
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
        let updated = g.update_node("a", patch, None).unwrap().unwrap();

        assert_eq!(updated.data.get("title"), Some(&serde_json::json!("hello")));
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
        let updated = g.update_node("a", patch, None).unwrap().unwrap();

        assert_eq!(updated.data.get("kept"), Some(&serde_json::json!(1)));
        assert_eq!(updated.data.get("added"), Some(&serde_json::json!(2)));
    }

    #[test]
    fn update_node_replaces_the_vector_and_marks_it_dirty() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();
        g.flush().unwrap();

        let updated = g.update_node("a", serde_json::Map::new(), Some(vec![1.0, 2.0])).unwrap().unwrap();

        assert_eq!(updated.vector, Some(vec![1.0, 2.0]));
        assert!(g.hnsw.has("a"));
        assert!(g.dirty_vectors.contains("a"));
    }

    #[test]
    fn update_node_rejects_a_non_finite_vector() {
        let mut g = test_graph();
        g.add_node(node("a")).unwrap();

        let err = g.update_node("a", serde_json::Map::new(), Some(vec![f64::NAN])).unwrap_err();
        let _ = err;
        assert!(g.get_node("a").unwrap().vector.is_none(), "node must be untouched on error");
    }

    #[test]
    fn update_node_returns_none_for_a_node_not_in_the_hot_cache() {
        let mut g = test_graph();
        let mut patch = serde_json::Map::new();
        patch.insert("title".into(), serde_json::json!("hello"));
        assert!(g.update_node("missing", patch, None).unwrap().is_none());
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
            GraphConfig { hot_cache_max, hnsw: HnswConfig::default() },
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

        let mut g2 = graph_on(&storage, GraphConfig { hot_cache_max: 1, hnsw: HnswConfig::default() });
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
    fn query_filters_by_node_type() {
        let mut g = test_graph();
        g.add_node(node_of_type("a", "doc")).unwrap();
        g.add_node(node_of_type("b", "article")).unwrap();

        let plan = QueryPlan { node_types: Some(vec!["doc".into()]), ..Default::default() };
        let ids = g.query(&plan).unwrap();

        assert_eq!(ids, vec!["a".to_string()]);
    }

    #[test]
    fn query_filters_by_attribute_eq() {
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
        let ids = g.query(&plan).unwrap();

        assert_eq!(ids, vec!["a".to_string()]);
    }

    #[test]
    fn query_filters_by_edge_filter_target() {
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
        let ids = g.query(&plan).unwrap();

        assert_eq!(ids, vec!["a".to_string()], "only a has a REL edge into c");
    }

    #[test]
    fn query_respects_limit() {
        let mut g = test_graph();
        g.add_nodes(vec![node("a"), node("b"), node("c")]).unwrap();

        let plan = QueryPlan { limit: Some(2), ..Default::default() };
        assert_eq!(g.query(&plan).unwrap().len(), 2);
    }

    #[test]
    fn aggregate_sums_a_numeric_field_over_matched_nodes() {
        let mut g = test_graph();
        let mut a = node_of_type("a", "doc");
        a.data.insert("score".into(), serde_json::json!(3.0));
        g.add_node(a).unwrap();
        let mut b = node_of_type("b", "doc");
        b.data.insert("score".into(), serde_json::json!(4.0));
        g.add_node(b).unwrap();
        g.add_node(node_of_type("c", "article")).unwrap();

        let plan = QueryPlan { node_types: Some(vec!["doc".into()]), ..Default::default() };
        let (sum, count) = g.aggregate(&plan, "score", "sum").unwrap();

        assert_eq!(sum, 7.0);
        assert_eq!(count, 2);
    }

    #[test]
    fn aggregate_rejects_an_unknown_op() {
        let mut g = test_graph();
        let mut a = node("a");
        a.data.insert("score".into(), serde_json::json!(1.0));
        g.add_node(a).unwrap();

        assert!(g.aggregate(&QueryPlan::default(), "score", "median").is_err());
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
        g.update_node("a", patch, None).unwrap();

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
}
