#![cfg_attr(
  all(not(debug_assertions), target_os = "windows"),
  windows_subsystem = "windows"
)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use polypack_core::hnsw::{HnswConfig, HnswIndex};
use polypack_core::vector::{DistanceFn, ExactIndex};
use polypack_core::NodeActivation as CoreNodeActivation;
use std::cell::RefCell;

#[napi(object)]
pub struct EngineInfo {
  pub graph: String,
  pub vector: String,
  pub storage: String,
}

#[napi]
pub fn engine_info() -> EngineInfo {
  EngineInfo {
    graph: "typescript".into(),
    vector: "rust-native".into(),
    storage: "host".into(),
  }
}

#[napi(object)]
pub struct ScoredId {
  pub id: String,
  pub score: f64,
}

#[napi(object)]
pub struct IndexEntry {
  pub id: String,
  pub vector: Float64Array,
}

#[napi]
pub struct NativeExactIndex {
  inner: RefCell<ExactIndex>,
}

#[napi]
impl NativeExactIndex {
  #[napi(constructor)]
  pub fn new(distance: Option<String>) -> Self {
    let distance = match distance.as_deref() {
      Some("euclidean") => DistanceFn::Euclidean,
      _ => DistanceFn::Cosine,
    };
    NativeExactIndex {
      inner: RefCell::new(ExactIndex::new(distance)),
    }
  }

  #[napi]
  pub fn add(&self, id: String, vector: Float64Array) -> Result<()> {
    let mut inner = self.inner.borrow_mut();
    inner
      .add(&id, &vector)
      .map_err(|e| Error::from_reason(e.to_string()))
  }

  #[napi]
  pub fn add_many(&self, ids: Vec<String>, vectors: Vec<Float64Array>) -> Result<()> {
    if ids.len() != vectors.len() {
      return Err(Error::from_reason(format!(
        "invalid_argument: ids and vectors length mismatch: {} != {}",
        ids.len(),
        vectors.len()
      )));
    }
    let mut inner = self.inner.borrow_mut();
    for (id, v) in ids.iter().zip(vectors.iter()) {
      inner
        .add(id, &v[..])
        .map_err(|e| Error::from_reason(e.to_string()))?;
    }
    Ok(())
  }

  #[napi]
  pub fn remove(&self, id: String) {
    self.inner.borrow_mut().remove(&id)
  }

  #[napi]
  pub fn remove_many(&self, ids: Vec<String>) {
    let mut inner = self.inner.borrow_mut();
    for id in ids {
      inner.remove(&id);
    }
  }

  #[napi]
  pub fn query(&self, vector: Float64Array, top_k: u32, threshold: Option<f64>) -> Result<Vec<ScoredId>> {
    let inner = self.inner.borrow();
    let results = inner
      .query(&vector, top_k as usize, threshold.unwrap_or(0.0))
      .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(results
      .into_iter()
      .map(|s| ScoredId { id: s.id, score: s.score })
      .collect())
  }

  #[napi]
  pub fn clear(&self) {
    self.inner.borrow_mut().clear()
  }

  #[napi]
  pub fn has(&self, id: String) -> bool {
    self.inner.borrow().has(&id)
  }

  #[napi]
  pub fn get(&self, id: String) -> Option<Float64Array> {
    self.inner.borrow().get(&id).map(|v| Float64Array::from(v.to_vec()))
  }

  #[napi]
  pub fn entries(&self) -> Vec<IndexEntry> {
    self
      .inner
      .borrow()
      .entries()
      .iter()
      .map(|(id, v)| IndexEntry {
        id: id.clone(),
        vector: Float64Array::from(v.to_vec()),
      })
      .collect()
  }

  #[napi(getter)]
  pub fn size(&self) -> u32 {
    self.inner.borrow().size() as u32
  }
}

#[napi(object)]
pub struct HnswConfigInput {
  pub m: Option<u32>,
  pub mmax0: Option<u32>,
  pub ef_construction: Option<u32>,
  pub ef_search: Option<u32>,
}

#[napi]
pub struct NativeHnswIndex {
  inner: RefCell<HnswIndex>,
}

#[napi]
impl NativeHnswIndex {
  #[napi(constructor)]
  pub fn new(config: Option<HnswConfigInput>, level_seed: Option<u32>) -> Self {
    let base = HnswConfig::default();
    let cfg = HnswConfig {
      m: config.as_ref().and_then(|c| c.m).map(|v| v as usize).unwrap_or(base.m),
      mmax0: config.as_ref().and_then(|c| c.mmax0).map(|v| v as usize).unwrap_or(base.mmax0),
      ef_construction: config
        .as_ref()
        .and_then(|c| c.ef_construction)
        .map(|v| v as usize)
        .unwrap_or(base.ef_construction),
      ef_search: config
        .as_ref()
        .and_then(|c| c.ef_search)
        .map(|v| v as usize)
        .unwrap_or(base.ef_search),
    };
    NativeHnswIndex {
      inner: RefCell::new(HnswIndex::new(cfg, level_seed.unwrap_or(7))),
    }
  }

  #[napi]
  pub fn add(&self, id: String, vector: Float64Array) -> Result<()> {
    let mut inner = self.inner.borrow_mut();
    inner
      .add(&id, &vector)
      .map_err(|e| Error::from_reason(e.to_string()))
  }

  #[napi]
  pub fn update(&self, id: String, vector: Float64Array) -> Result<()> {
    let mut inner = self.inner.borrow_mut();
    inner
      .update(&id, &vector)
      .map_err(|e| Error::from_reason(e.to_string()))
  }

  #[napi]
  pub fn add_many(&self, ids: Vec<String>, vectors: Vec<Float64Array>) -> Result<()> {
    if ids.len() != vectors.len() {
      return Err(Error::from_reason(format!(
        "invalid_argument: ids and vectors length mismatch: {} != {}",
        ids.len(),
        vectors.len()
      )));
    }
    let mut inner = self.inner.borrow_mut();
    for (id, v) in ids.iter().zip(vectors.iter()) {
      inner
        .add(id, &v[..])
        .map_err(|e| Error::from_reason(e.to_string()))?;
    }
    Ok(())
  }

  #[napi]
  pub fn remove(&self, id: String) {
    self.inner.borrow_mut().remove(&id)
  }

  #[napi]
  pub fn remove_many(&self, ids: Vec<String>) {
    let mut inner = self.inner.borrow_mut();
    for id in ids {
      inner.remove(&id);
    }
  }

  #[napi]
  pub fn query(&self, vector: Float64Array, top_k: u32, threshold: Option<f64>) -> Result<Vec<ScoredId>> {
    let inner = self.inner.borrow();
    let results = inner
      .query(&vector, top_k as usize, threshold.unwrap_or(0.0))
      .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(results
      .into_iter()
      .map(|s| ScoredId { id: s.id, score: s.score })
      .collect())
  }

  #[napi]
  pub fn clear(&self) {
    self.inner.borrow_mut().clear()
  }

  #[napi]
  pub fn has(&self, id: String) -> bool {
    self.inner.borrow().has(&id)
  }

  #[napi]
  pub fn get(&self, id: String) -> Option<Float64Array> {
    self.inner.borrow().get(&id).map(|v| Float64Array::from(v.to_vec()))
  }

  #[napi]
  pub fn entries(&self) -> Vec<IndexEntry> {
    // HnswIndex stores nodes in a HashMap; collect sorted by id for stability.
    let mut rows: Vec<(String, Vec<f64>)> = self
      .inner
      .borrow()
      .nodes()
      .iter()
      .map(|(id, v)| (id.clone(), v.clone()))
      .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows
      .into_iter()
      .map(|(id, v)| IndexEntry { id, vector: Float64Array::from(v) })
      .collect()
  }

  #[napi(getter)]
  pub fn size(&self) -> u32 {
    self.inner.borrow().size() as u32
  }
}

// ── Storage / NativeStore ──

use polypack_core::model::ChangeBatch as CoreChangeBatch;
use polypack_core::storage::{Durability, NodeQuery, Store as CoreStore, StoreConfig, Storage};
use std::path::PathBuf;

/// Filesystem byte storage used by the native store (host adapter for Node).
struct FsStorage {
    dir: PathBuf,
}

impl Storage for FsStorage {
    fn read(&self, name: &str) -> std::result::Result<Option<Vec<u8>>, polypack_core::PolypackError> {
        match std::fs::read(self.dir.join(name)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(polypack_core::PolypackError::Storage(e.to_string())),
        }
    }
    fn write(
        &mut self,
        name: &str,
        data: &[u8],
    ) -> std::result::Result<(), polypack_core::PolypackError> {
        use std::io::Write;
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
        // Write-then-rename: a crash mid-write leaves the previous snapshot
        // intact instead of a torn file, and the tmp file is fsynced first
        // so the rename can never land ahead of its data on disk.
        let tmp_path = self.dir.join(format!("{name}.tmp"));
        let mut file = std::fs::File::create(&tmp_path)
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
        file.write_all(data)
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
        file.sync_all()
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
        drop(file);
        std::fs::rename(&tmp_path, self.dir.join(name))
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))
    }
    fn append(
        &mut self,
        name: &str,
        data: &[u8],
    ) -> std::result::Result<(), polypack_core::PolypackError> {
        use std::io::Write;
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join(name))
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
        file.write_all(data)
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))
    }
    fn delete(&mut self, name: &str) -> std::result::Result<(), polypack_core::PolypackError> {
        let _ = std::fs::remove_file(self.dir.join(name));
        Ok(())
    }
    fn exists(&self, name: &str) -> std::result::Result<bool, polypack_core::PolypackError> {
        Ok(self.dir.join(name).exists())
    }
    fn sync(&self, name: &str) -> std::result::Result<(), polypack_core::PolypackError> {
        let file = std::fs::File::open(self.dir.join(name))
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
        file.sync_all()
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))
    }
    fn sync_dir(&self) -> std::result::Result<(), polypack_core::PolypackError> {
        // Needed so a renamed-in snapshot/WAL file is durably linked into the
        // directory, not just written; directory fsync has no equivalent on
        // Windows, so this is best-effort there.
        #[cfg(unix)]
        {
            let dir = std::fs::File::open(&self.dir)
                .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
            dir.sync_all()
                .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
        }
        Ok(())
    }
}

fn to_napi_err(e: polypack_core::PolypackError) -> Error {
    Error::from_reason(e.to_string())
}

#[napi]
pub struct NativeStore {
    inner: RefCell<CoreStore>,
}

#[napi]
impl NativeStore {
    #[napi(constructor)]
    pub fn new(dir: String, compact_threshold: Option<u32>) -> Self {
        let config = StoreConfig {
            compact_threshold: compact_threshold.unwrap_or(10_000) as usize,
            durability: Durability::Process,
        };
        NativeStore {
            inner: RefCell::new(CoreStore::new(
                Box::new(FsStorage { dir: PathBuf::from(dir) }),
                config,
            )),
        }
    }

    /// Apply a change batch: `{ putNodes, deleteNodeIds, putEdges,
    /// deleteEdgeIds, putVectors, deleteVectorIds }`.
    #[napi]
    pub fn apply(&self, changes: serde_json::Value) -> Result<()> {
        let batch: CoreChangeBatch = serde_json::from_value(changes)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        self.inner.borrow_mut().apply(&batch).map_err(to_napi_err)
    }

    #[napi]
    pub fn node_ids(&self) -> Result<Vec<String>> {
        self.inner.borrow_mut().node_ids().map_err(to_napi_err)
    }

    /// Total persisted node count, without materialising ids.
    #[napi]
    pub fn node_count(&self) -> Result<u32> {
        self.inner.borrow_mut().node_count().map(|n| n as u32).map_err(to_napi_err)
    }

    /// Query nodes with a storage-level query object:
    /// `{ nodeTypes, attributes, attributeRanges, orderBy, offset, limit }`.
    #[napi]
    pub fn query_nodes(&self, query: serde_json::Value) -> Result<Vec<serde_json::Value>> {
        let q: NodeQuery =
            serde_json::from_value(query).map_err(|e| Error::from_reason(e.to_string()))?;
        let nodes = self.inner.borrow_mut().query_nodes(&q).map_err(to_napi_err)?;
        nodes
            .into_iter()
            .map(|n| serde_json::to_value(n).map_err(|e| Error::from_reason(e.to_string())))
            .collect()
    }

    /// Count nodes matching a storage-level query object.
    #[napi]
    pub fn count_nodes(&self, query: serde_json::Value) -> Result<u32> {
        let q: NodeQuery =
            serde_json::from_value(query).map_err(|e| Error::from_reason(e.to_string()))?;
        self.inner.borrow_mut().count_nodes(&q).map(|n| n as u32).map_err(to_napi_err)
    }

    /// Edges from the given sources, optionally filtered by edge type.
    #[napi]
    pub fn get_edges_by_sources(
        &self,
        sources: Vec<String>,
        edge_type: Option<String>,
    ) -> Result<Vec<serde_json::Value>> {
        let edges = self
            .inner
            .borrow_mut()
            .get_edges_by_sources(&sources, edge_type.as_deref())
            .map_err(to_napi_err)?;
        edges
            .into_iter()
            .map(|e| serde_json::to_value(e).map_err(|e| Error::from_reason(e.to_string())))
            .collect()
    }

    /// Edges targeting the given nodes, optionally filtered by edge type.
    #[napi]
    pub fn get_edges_by_targets(
        &self,
        targets: Vec<String>,
        edge_type: Option<String>,
    ) -> Result<Vec<serde_json::Value>> {
        let edges = self
            .inner
            .borrow_mut()
            .get_edges_by_targets(&targets, edge_type.as_deref())
            .map_err(to_napi_err)?;
        edges
            .into_iter()
            .map(|e| serde_json::to_value(e).map_err(|e| Error::from_reason(e.to_string())))
            .collect()
    }

    #[napi]
    pub fn get_node(&self, id: String) -> Result<Option<serde_json::Value>> {
        match self.inner.borrow_mut().get_node(&id).map_err(to_napi_err)? {
            Some(node) => serde_json::to_value(node)
                .map(Some)
                .map_err(|e| Error::from_reason(e.to_string())),
            None => Ok(None),
        }
    }

    #[napi]
    pub fn all_edges(&self) -> Result<Vec<serde_json::Value>> {
        let edges = self.inner.borrow_mut().edges_snapshot().map_err(to_napi_err)?;
        edges
            .into_iter()
            .map(|(_, e)| serde_json::to_value(e).map_err(|e| Error::from_reason(e.to_string())))
            .collect()
    }

    /// Returns `[[id, vector], ...]` pairs.
    #[napi]
    pub fn all_vectors(&self) -> Result<Vec<serde_json::Value>> {
        let vectors = self.inner.borrow_mut().vectors_snapshot().map_err(to_napi_err)?;
        Ok(vectors.into_iter().map(|(id, v)| serde_json::json!([id, v])).collect())
    }

    #[napi]
    pub fn compact(&self) -> Result<()> {
        self.inner.borrow_mut().compact().map_err(to_napi_err)
    }

    #[napi]
    pub fn close(&self) -> Result<()> {
        self.inner.borrow_mut().close().map_err(to_napi_err)
    }
}

// ── Query execution ──

use polypack_core::model::{Edge as CoreEdge, Node as CoreNode};
use polypack_core::query::QueryPlan as CoreQueryPlan;
use polypack_core::query_exec::{aggregate as core_aggregate, execute as core_execute, GraphSnapshot};

fn json_vec<T: serde::de::DeserializeOwned>(items: Vec<serde_json::Value>, what: &str) -> Result<Vec<T>> {
    items
        .into_iter()
        .map(|v| serde_json::from_value(v).map_err(|e| Error::from_reason(format!("{what}: {e}"))))
        .collect()
}

/// Execute a query plan over a snapshot of nodes/edges, returning ordered ids.
#[napi]
pub fn execute_query_plan(
    nodes: Vec<serde_json::Value>,
    edges: Vec<serde_json::Value>,
    plan: serde_json::Value,
) -> Result<Vec<String>> {
    let nodes: Vec<CoreNode> = json_vec(nodes, "node")?;
    let edges: Vec<CoreEdge> = json_vec(edges, "edge")?;
    let plan: CoreQueryPlan =
        serde_json::from_value(plan).map_err(|e| Error::from_reason(e.to_string()))?;
    let snap = GraphSnapshot::new(nodes, edges);
    core_execute(&snap, &plan, None).map_err(to_napi_err)
}

/// Aggregate a numeric field across the nodes matched by a query plan.
#[napi]
pub fn aggregate_query_plan(
    nodes: Vec<serde_json::Value>,
    edges: Vec<serde_json::Value>,
    plan: serde_json::Value,
    field: String,
    op: String,
) -> Result<serde_json::Value> {
    let nodes: Vec<CoreNode> = json_vec(nodes, "node")?;
    let edges: Vec<CoreEdge> = json_vec(edges, "edge")?;
    let plan: CoreQueryPlan =
        serde_json::from_value(plan).map_err(|e| Error::from_reason(e.to_string()))?;
    let snap = GraphSnapshot::new(nodes, edges);
    let (value, count) = core_aggregate(&snap, &plan, &field, &op).map_err(to_napi_err)?;
    Ok(serde_json::json!({ "value": value, "count": count }))
}

// ── activation helpers (pure math, for parity with the TypeScript layer) ──

#[napi(object)]
pub struct NodeActivation {
  pub score: f64,
  pub importance: f64,
  pub reinforcement_count: f64,
  pub last_meaningful_activation: i64,
}

fn to_napi_activation(a: &CoreNodeActivation) -> NodeActivation {
  NodeActivation {
    score: a.score,
    importance: a.importance,
    reinforcement_count: a.reinforcement_count as f64,
    last_meaningful_activation: a.last_meaningful_activation,
  }
}

fn from_napi_activation(a: &NodeActivation) -> CoreNodeActivation {
  CoreNodeActivation {
    score: a.score,
    importance: a.importance,
    reinforcement_count: a.reinforcement_count as u64,
    last_meaningful_activation: a.last_meaningful_activation,
  }
}

/// Exponential-decay multiplier `0.5 ** (elapsed / halfLife)`.
#[napi]
pub fn decay_factor(elapsed_ms: i64, half_life_ms: i64) -> f64 {
  polypack_core::decay_factor(elapsed_ms, half_life_ms)
}

/// Merge two durable activation totals (max-merge, re-anchored to `now`).
#[napi]
pub fn merge_activation(existing: NodeActivation, incoming: NodeActivation, now: Option<i64>) -> NodeActivation {
  to_napi_activation(&polypack_core::merge_activation(
    &from_napi_activation(&existing),
    &from_napi_activation(&incoming),
    now.unwrap_or_else(|| {
      std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_millis() as i64
    }),
  ))
}

/// Apply a reinforcement delta to a durable activation record.
#[napi]
pub fn reinforce_activation(previous: Option<NodeActivation>, delta: f64, now: i64) -> NodeActivation {
  to_napi_activation(&polypack_core::reinforce_activation(
    previous.as_ref().map(from_napi_activation).as_ref(),
    delta,
    now,
    &polypack_core::DEFAULT_ACTIVATION,
  ))
}

/// Current decayed activation score for a stored `score`/anchor pair (0 when
/// the node has no activation — pass `score` and a missing flag via the
/// `Option<f64>` form instead by calling with `score: 0`).
#[napi]
pub fn activation_score_of(score: f64, last_meaningful_activation: i64, now: i64, half_life_ms: i64) -> f64 {
  polypack_core::clamp01(score * polypack_core::decay_factor(now - last_meaningful_activation, half_life_ms))
}
