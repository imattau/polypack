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
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

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
  /// `"cosine"` (default) or `"euclidean"`.
  pub distance: Option<String>,
}

#[napi]
pub struct NativeHnswIndex {
  inner: RefCell<HnswIndex>,
}

#[napi]
impl NativeHnswIndex {
  #[napi(constructor)]
  pub fn new(config: Option<HnswConfigInput>, level_seed: Option<u32>) -> Result<Self> {
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
      distance: match config.as_ref().and_then(|c| c.distance.as_deref()) {
        Some("euclidean") => DistanceFn::Euclidean,
        _ => base.distance,
      },
    };
    Ok(NativeHnswIndex {
      inner: RefCell::new(HnswIndex::new(cfg, level_seed.unwrap_or(7)).map_err(to_napi_err)?),
    })
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

use polypack_core::model::{ChangeBatch as CoreChangeBatch, Edge as CoreEdgeModel, MemoryClass as CoreMemoryClass, Node as CoreNodeModel, VectorEntry as CoreVectorEntry};
use polypack_core::storage::{AdapterCapabilities, Durability, NodeQuery, Store as CoreStore, StoreConfig, Storage, VectorSearchCapability};
use std::path::{Path, PathBuf};

/// Deserialize an optional napi-bridged JSON field, treating a JS `null`
/// the same as an omitted (`undefined`) property.
///
/// `#[napi(object)]`'s generated getter for an `Option<T>` field only
/// treats a genuinely *missing* JS property as `None` — if the property is
/// present but explicitly `null`, it calls `T::from_napi_value` directly
/// (bypassing `Option<T>`'s own null handling), which fails for any `T`
/// that doesn't itself tolerate `null` (e.g. `Vec<f64>`, `String`, `f64`).
/// This codebase's wire contract uses explicit `null` for absent
/// `vector`/edge `data` (not omission), confirmed by grepping every
/// `SerializedNode`/`SerializedEdge` call site — so every optional field
/// here is typed `Option<serde_json::Value>` on the napi struct and routed
/// through this helper instead.
fn opt_json<T: serde::de::DeserializeOwned>(v: Option<serde_json::Value>, field: &str) -> Result<Option<T>> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => serde_json::from_value(value).map(Some).map_err(|e| Error::from_reason(format!("{field}: {e}"))),
    }
}

/// Typed mirror of `polypack_core::model::Node` for `NativeStore::apply`.
///
/// `#[napi(object)]` gives the required fields (`id`, `type`, `insertedAt`,
/// `updatedAt`) direct per-field conversion off the JS value with no
/// intermediate JSON tree; every optional field still goes through a small,
/// per-field `serde_json::Value` (see `opt_json`) rather than `data`'s old
/// behavior of JSON-bridging the *entire* node (or the whole batch) at
/// once. Confirmed via benchmarks/database-core-napi-report.md: the old
/// whole-batch double-deserialize (napi's JS->JSON walk, then
/// `serde_json::from_value` into `CoreChangeBatch`) was up to 17x slower
/// than the equivalent TypeScript write path at small batch sizes.
#[napi(object)]
pub struct NapiNode {
    pub id: String,
    #[napi(js_name = "type")]
    pub node_type: String,
    pub data: Option<serde_json::Value>,
    pub vector: Option<serde_json::Value>,
    pub inserted_at: i64,
    pub updated_at: i64,
    pub revision: Option<serde_json::Value>,
    pub activation: Option<serde_json::Value>,
    pub memory_class: Option<serde_json::Value>,
    pub confidence: Option<serde_json::Value>,
    pub source: Option<serde_json::Value>,
    pub observed_at: Option<serde_json::Value>,
    pub derived_from: Option<serde_json::Value>,
    pub supersedes: Option<serde_json::Value>,
    pub contradicts: Option<serde_json::Value>,
}

impl TryFrom<NapiNode> for CoreNodeModel {
    type Error = Error;
    fn try_from(n: NapiNode) -> Result<Self> {
        let data = match n.data {
            Some(serde_json::Value::Object(map)) => map,
            Some(serde_json::Value::Null) | None => Default::default(),
            Some(_) => return Err(Error::from_reason("node data must be an object")),
        };
        Ok(CoreNodeModel {
            id: n.id,
            node_type: n.node_type,
            data,
            vector: opt_json(n.vector, "vector")?,
            inserted_at: n.inserted_at,
            updated_at: n.updated_at,
            revision: opt_json::<u64>(n.revision, "revision")?.unwrap_or(0),
            activation: opt_json::<CoreNodeActivation>(n.activation, "activation")?,
            memory_class: opt_json::<CoreMemoryClass>(n.memory_class, "memoryClass")?,
            confidence: opt_json(n.confidence, "confidence")?,
            source: opt_json(n.source, "source")?,
            observed_at: opt_json(n.observed_at, "observedAt")?,
            derived_from: opt_json(n.derived_from, "derivedFrom")?,
            supersedes: opt_json(n.supersedes, "supersedes")?,
            contradicts: opt_json(n.contradicts, "contradicts")?,
        })
    }
}

/// Typed mirror of `polypack_core::model::Edge` for `NativeStore::apply` — see `NapiNode`.
#[napi(object)]
pub struct NapiEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    #[napi(js_name = "type")]
    pub edge_type: String,
    pub data: Option<serde_json::Value>,
    pub created_at: i64,
    pub revision: Option<serde_json::Value>,
}

impl TryFrom<NapiEdge> for CoreEdgeModel {
    type Error = Error;
    fn try_from(e: NapiEdge) -> Result<Self> {
        let data = match e.data {
            Some(serde_json::Value::Object(map)) => Some(map),
            Some(serde_json::Value::Null) | None => None,
            Some(_) => return Err(Error::from_reason("edge data must be an object")),
        };
        Ok(CoreEdgeModel {
            id: e.id,
            source: e.source,
            target: e.target,
            edge_type: e.edge_type,
            data,
            created_at: e.created_at,
            revision: opt_json::<u64>(e.revision, "revision")?.unwrap_or(0),
        })
    }
}

/// Typed mirror of `polypack_core::model::VectorEntry` for `NativeStore::apply`.
#[napi(object)]
pub struct NapiVectorEntry {
    pub id: String,
    pub vector: Vec<f64>,
}

impl From<NapiVectorEntry> for CoreVectorEntry {
    fn from(v: NapiVectorEntry) -> Self {
        CoreVectorEntry { id: v.id, vector: v.vector }
    }
}

/// Typed mirror of `polypack_core::model::ChangeBatch` for `NativeStore::apply`.
#[napi(object)]
pub struct NapiChangeBatch {
    pub put_nodes: Option<Vec<NapiNode>>,
    pub delete_node_ids: Option<Vec<String>>,
    pub put_edges: Option<Vec<NapiEdge>>,
    pub delete_edge_ids: Option<Vec<String>>,
    pub put_vectors: Option<Vec<NapiVectorEntry>>,
    pub delete_vector_ids: Option<Vec<String>>,
}

impl TryFrom<NapiChangeBatch> for CoreChangeBatch {
    type Error = Error;
    fn try_from(b: NapiChangeBatch) -> Result<Self> {
        Ok(CoreChangeBatch {
            put_nodes: b.put_nodes.unwrap_or_default().into_iter().map(TryInto::try_into).collect::<Result<_>>()?,
            delete_node_ids: b.delete_node_ids.unwrap_or_default(),
            put_edges: b.put_edges.unwrap_or_default().into_iter().map(TryInto::try_into).collect::<Result<_>>()?,
            delete_edge_ids: b.delete_edge_ids.unwrap_or_default(),
            put_vectors: b.put_vectors.unwrap_or_default().into_iter().map(Into::into).collect(),
            delete_vector_ids: b.delete_vector_ids.unwrap_or_default(),
        })
    }
}

/// Filesystem byte storage used by the native store (host adapter for Node).
struct FsStorage {
    dir: PathBuf,
    read_only: bool,
    lock_token: String,
    lock_file: Option<File>,
}

impl FsStorage {
    fn new(dir: PathBuf, read_only: bool) -> std::result::Result<Self, polypack_core::PolypackError> {
        fs::create_dir_all(&dir).map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
        if read_only {
            return Ok(Self { dir, read_only: true, lock_token: String::new(), lock_file: None });
        }
        let lock_path = dir.join("store.lock");
        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?
            .as_millis() as u64;
        let token = format!("{}-{started_at}", std::process::id());
        let metadata = format!(r#"{{"pid":{},"startedAt":{},"token":"{}"}}"#, std::process::id(), started_at, token);

        for attempt in 0..2 {
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(mut lock_file) => {
                    lock_file.write_all(metadata.as_bytes()).map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
                    lock_file.sync_all().map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
                    return Ok(Self { dir, read_only: false, lock_token: token, lock_file: Some(lock_file) });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists && attempt == 0 => {
                    let stale = fs::read_to_string(&lock_path)
                        .ok()
                        .and_then(|contents| serde_json::from_str::<serde_json::Value>(&contents).ok())
                        .and_then(|value| value.get("startedAt").and_then(serde_json::Value::as_u64))
                        .map(|old| started_at.saturating_sub(old) > 24 * 60 * 60 * 1000)
                        .unwrap_or(false);
                    if stale {
                        let _ = fs::remove_file(&lock_path);
                        continue;
                    }
                    return Err(polypack_core::PolypackError::Storage(format!("store is already locked: {}", lock_path.display())));
                }
                Err(error) => return Err(polypack_core::PolypackError::Storage(error.to_string())),
            }
        }
        Err(polypack_core::PolypackError::Storage(format!("store is already locked: {}", lock_path.display())))
    }
}

impl Drop for FsStorage {
    fn drop(&mut self) {
        if self.lock_file.take().is_none() {
            return;
        }
        // Re-read the lock file by path rather than through the retained file
        // descriptor: on POSIX, unlinking a path leaves an open fd pointing at
        // the old (now-detached) inode, so reading via the fd can observe our
        // own stale content even after another process has since replaced the
        // lock at that path. A fresh path-based read reflects whichever lock
        // is actually current, so we only ever delete a lock we still own.
        release_store_lock(&self.dir, &self.lock_token);
    }
}

fn release_store_lock(dir: &Path, token: &str) {
    let lock_path = dir.join("store.lock");
    let owns_lock = fs::read_to_string(&lock_path)
        .map(|contents| contents.contains(&format!(r#""token":"{}""#, token)))
        .unwrap_or(false);
    if owns_lock {
        let _ = fs::remove_file(lock_path);
    }
}

impl Storage for FsStorage {
    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            atomic_batches: true,
            fsync: true,
            snapshots: true,
            change_feed: true,
            secondary_indexes: true,
            vector_search: VectorSearchCapability::Exact,
            ..Default::default()
        }
    }

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
        if self.read_only { return Err(polypack_core::PolypackError::Storage("store was opened read-only".into())); }
        use std::io::Write;
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
        // Write-then-rename: a crash mid-write leaves the previous snapshot
        // intact instead of a torn file. Fsyncing the tmp file before the
        // rename is `Store`'s call via the explicit `sync()`/`sync_dir()` it
        // issues only under `Durability::Fsync` — doing it unconditionally
        // here silently upgraded `Durability::Process` ("written to the OS,
        // not fsynced") into an always-fsync adapter.
        let tmp_path = self.dir.join(format!("{name}.tmp"));
        let mut file = std::fs::File::create(&tmp_path)
            .map_err(|e| polypack_core::PolypackError::Storage(e.to_string()))?;
        file.write_all(data)
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
        if self.read_only { return Err(polypack_core::PolypackError::Storage("store was opened read-only".into())); }
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
        if self.read_only { return Err(polypack_core::PolypackError::Storage("store was opened read-only".into())); }
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
    lock_dir: PathBuf,
    lock_token: String,
}

#[napi]
impl NativeStore {
    #[napi(constructor)]
    pub fn new(dir: String, compact_threshold: Option<u32>, read_only: Option<bool>) -> Result<Self> {
        let config = StoreConfig {
            compact_threshold: compact_threshold.unwrap_or(10_000) as usize,
            durability: Durability::Process,
        };
        let lock_dir = PathBuf::from(dir);
        let storage = FsStorage::new(lock_dir.clone(), read_only.unwrap_or(false)).map_err(to_napi_err)?;
        let lock_token = storage.lock_token.clone();
        Ok(NativeStore { inner: RefCell::new(CoreStore::new(Box::new(storage), config)), lock_dir, lock_token })
    }

    /// Apply a change batch: `{ putNodes, deleteNodeIds, putEdges,
    /// deleteEdgeIds, putVectors, deleteVectorIds }`.
    ///
    /// Takes the typed `NapiChangeBatch` (per-field napi conversion, no
    /// intermediate JSON tree for the batch as a whole) rather than
    /// `serde_json::Value` — see `NapiNode`'s docs. Confirmed as a real fix
    /// via benchmarks/database-core-napi-report.md, which measured the old
    /// double-deserialize as up to 17x slower than the TS lane at small
    /// batch sizes.
    #[napi]
    pub fn apply(&self, changes: NapiChangeBatch) -> Result<()> {
        let batch: CoreChangeBatch = changes.try_into()?;
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

    /// Define or replace a persisted node-data index.
    #[napi]
    pub fn define_index(&self, definition: serde_json::Value) -> Result<()> {
        let definition: polypack_core::storage::SecondaryIndexDefinition = serde_json::from_value(definition)
            .map_err(|error| Error::from_reason(format!("invalid_argument: {error}")))?;
        self.inner.borrow_mut().define_index(definition).map_err(to_napi_err)
    }

    /// Drop a persisted node-data index.
    #[napi]
    pub fn drop_index(&self, name: String) -> Result<bool> {
        self.inner.borrow_mut().drop_index(&name).map_err(to_napi_err)
    }

    /// Return persisted index definitions.
    #[napi]
    pub fn index_definitions(&self) -> Result<Vec<serde_json::Value>> {
        self.inner
            .borrow_mut()
            .index_definitions()
            .map_err(to_napi_err)?
            .into_iter()
            .map(|definition| serde_json::to_value(definition).map_err(|error| Error::from_reason(error.to_string())))
            .collect()
    }

    /// Register or replace a node-type schema definition.
    #[napi]
    pub fn register_node_type(&self, definition: serde_json::Value) -> Result<()> {
        let definition: polypack_core::storage::NodeTypeDefinition = serde_json::from_value(definition)
            .map_err(|error| Error::from_reason(format!("invalid_argument: {error}")))?;
        self.inner.borrow_mut().register_node_type(definition).map_err(to_napi_err)
    }

    /// Register or replace an edge-type schema definition.
    #[napi]
    pub fn register_edge_type(&self, definition: serde_json::Value) -> Result<()> {
        let definition: polypack_core::storage::EdgeTypeDefinition = serde_json::from_value(definition)
            .map_err(|error| Error::from_reason(format!("invalid_argument: {error}")))?;
        self.inner.borrow_mut().register_edge_type(definition).map_err(to_napi_err)
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

    /// Force a durable checkpoint and compact the recovery WAL.
    #[napi]
    pub fn checkpoint(&self) -> Result<()> {
        self.inner.borrow_mut().compact().map_err(to_napi_err)
    }

    /// Validate persisted framing, records, references, vectors, and indexes.
    #[napi]
    pub fn verify(&self) -> Result<serde_json::Value> {
        let report = self.inner.borrow_mut().verify().map_err(to_napi_err)?;
        Ok(serde_json::json!({
            "ok": report.ok,
            "errors": report.errors,
            "nodeCount": report.node_count,
            "edgeCount": report.edge_count,
            "vectorCount": report.vector_count,
            "mutationCount": report.mutation_count,
        }))
    }

    /// Report the guarantees provided by the native filesystem adapter.
    #[napi]
    pub fn capabilities(&self) -> serde_json::Value {
        let capabilities = self.inner.borrow().capabilities();
        serde_json::json!({
            "atomicBatches": capabilities.atomic_batches,
            "transactions": capabilities.transactions,
            "fsync": capabilities.fsync,
            "secondaryIndexes": capabilities.secondary_indexes,
            "snapshots": capabilities.snapshots,
            "changeFeed": capabilities.change_feed,
            "concurrentWriters": capabilities.concurrent_writers,
            "vectorSearch": match capabilities.vector_search {
                VectorSearchCapability::None => "none",
                VectorSearchCapability::Exact => "exact",
                VectorSearchCapability::Ann => "ann",
            },
        })
    }

    /// Return operational counters for the native persistence store.
    #[napi]
    pub fn stats(&self) -> Result<serde_json::Value> {
        let mut inner = self.inner.borrow_mut();
        let node_count = inner.node_count().map_err(to_napi_err)?;
        let edge_count = inner.edges_snapshot().map_err(to_napi_err)?.len();
        let vector_count = inner.vectors_snapshot().map_err(to_napi_err)?.len();
        let mutation_count = inner.mutation_log().map_err(to_napi_err)?.len();
        let wal_bytes = fs::metadata(self.lock_dir.join("wal.msgpack"))
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        Ok(serde_json::json!({
            "persistedNodeCount": node_count,
            "edgeCount": edge_count,
            "vectorCount": vector_count,
            "mutationCount": mutation_count,
            "walBytes": wal_bytes,
        }))
    }

    /// Read durable logical mutations after a sequence cursor.
    #[napi]
    pub fn mutation_log_since(&self, sequence: String, limit: Option<u32>) -> Result<Vec<serde_json::Value>> {
        let sequence = sequence
            .parse::<u64>()
            .map_err(|error| Error::from_reason(format!("invalid_argument: sequence must be an unsigned integer: {error}")))?;
        let records = if let Some(limit) = limit {
            self.inner.borrow_mut().mutation_log_page(sequence, limit as usize).map_err(to_napi_err)?
        } else {
            self.inner.borrow_mut().mutation_log_since(sequence).map_err(to_napi_err)?
        };
        records
            .into_iter()
            .map(|record| serde_json::to_value(record).map_err(|error| Error::from_reason(error.to_string())))
            .collect()
    }

    /// Return the highest durable logical mutation sequence.
    #[napi]
    pub fn latest_mutation_sequence(&self) -> Result<String> {
        self.inner
            .borrow_mut()
            .latest_mutation_sequence()
            .map(|sequence| sequence.to_string())
            .map_err(to_napi_err)
    }

    /// Create a consistent checkpointed directory backup.
    #[napi]
    pub fn backup(&self, destination: String) -> Result<()> {
        let mut destination_storage = FsStorage::new(PathBuf::from(destination), false).map_err(to_napi_err)?;
        self.inner
            .borrow_mut()
            .backup(&mut destination_storage)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn close(&self) -> Result<()> {
        self.inner.borrow_mut().close().map_err(to_napi_err)?;
        release_store_lock(&self.lock_dir, &self.lock_token);
        Ok(())
    }
}

/// Restore a native store from a directory backup and validate the result.
#[napi]
pub fn restore_store(source: String, destination: String, compact_threshold: Option<u32>) -> Result<NativeStore> {
    let source_storage = FsStorage::new(PathBuf::from(source), true).map_err(to_napi_err)?;
    let destination_dir = PathBuf::from(destination);
    let destination_storage = FsStorage::new(destination_dir.clone(), false).map_err(to_napi_err)?;
    let lock_token = destination_storage.lock_token.clone();
    let config = StoreConfig {
        compact_threshold: compact_threshold.unwrap_or(10_000) as usize,
        durability: Durability::Process,
    };
    let mut inner = CoreStore::restore(&source_storage, Box::new(destination_storage), config).map_err(to_napi_err)?;
    let report = inner.verify().map_err(to_napi_err)?;
    if !report.ok {
        return Err(Error::from_reason(format!("storage: restored backup failed verification: {}", report.errors.join("; "))));
    }
    Ok(NativeStore { inner: RefCell::new(inner), lock_dir: destination_dir, lock_token })
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
  /// Suppression, subtracted from `score` at read time only. Absent is equivalent to 0.
  pub inhibition: Option<f64>,
  /// Epoch-ms anchor for `inhibition`'s decay. Absent iff `inhibition` is absent.
  pub last_inhibited_at: Option<i64>,
  /// Per-context activation (`{ [context]: { score, lastMeaningfulActivation } }`),
  /// additional to (not a replacement for) the global `score` above. Passed
  /// through as plain JSON — no dedicated napi type — so it round-trips
  /// whatever shape `merge_activation`/`reinforce_activation` produce.
  pub context: Option<serde_json::Value>,
}

fn to_napi_activation(a: &CoreNodeActivation) -> NodeActivation {
  NodeActivation {
    score: a.score,
    importance: a.importance,
    reinforcement_count: a.reinforcement_count as f64,
    last_meaningful_activation: a.last_meaningful_activation,
    inhibition: a.inhibition,
    last_inhibited_at: a.last_inhibited_at,
    context: a.context.as_ref().map(|c| serde_json::to_value(c).expect("context serializes")),
  }
}

fn from_napi_activation(a: &NodeActivation) -> CoreNodeActivation {
  CoreNodeActivation {
    score: a.score,
    importance: a.importance,
    reinforcement_count: a.reinforcement_count as u64,
    last_meaningful_activation: a.last_meaningful_activation,
    inhibition: a.inhibition,
    last_inhibited_at: a.last_inhibited_at,
    context: a.context.as_ref().and_then(|v| serde_json::from_value(v.clone()).ok()),
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

/// Apply a reinforcement delta to a durable activation record. When `context`
/// is given, the same delta additionally reinforces `activation.context[context]`
/// — an independently-decaying, additional lens, not a replacement for the
/// global score.
#[napi]
pub fn reinforce_activation(previous: Option<NodeActivation>, delta: f64, now: i64, context: Option<String>) -> NodeActivation {
  to_napi_activation(&polypack_core::reinforce_activation(
    previous.as_ref().map(from_napi_activation).as_ref(),
    delta,
    now,
    &polypack_core::DEFAULT_ACTIVATION,
    context.as_deref(),
  ))
}

/// Apply a suppression delta to a durable activation record's `inhibition`
/// (mirrors `reinforce_activation` but for the inhibition axis, which decays
/// on its own, shorter-by-default half-life). A negative `delta` releases
/// suppression.
#[napi]
pub fn suppress_activation(previous: Option<NodeActivation>, delta: f64, now: i64) -> NodeActivation {
  to_napi_activation(&polypack_core::suppress_activation(
    previous.as_ref().map(from_napi_activation).as_ref(),
    delta,
    now,
    polypack_core::DEFAULT_ACTIVATION.inhibition_half_life_ms,
  ))
}

/// Current decayed activation score for a stored `score`/anchor pair (0 when
/// the node has no activation — pass `score` and a missing flag via the
/// `Option<f64>` form instead by calling with `score: 0`).
#[napi]
pub fn activation_score_of(score: f64, last_meaningful_activation: i64, now: i64, half_life_ms: i64) -> f64 {
  polypack_core::clamp01(score * polypack_core::decay_factor(now - last_meaningful_activation, half_life_ms))
}

/// Conservative token estimate for a serialized memory payload. Native
/// callers can use this as the default `workingMemory` cost function when the
/// application does not provide a model-specific tokenizer.
#[napi]
pub fn estimate_node_tokens(serialized_memory: String) -> u32 {
  ((serialized_memory.len() as f64 / 4.0).ceil().max(1.0)) as u32
}

#[napi(object)]
pub struct ActivationScoreBreakdown {
  pub semantic: f64,
  pub graph: f64,
  pub recency: f64,
  pub usage: f64,
  pub weighted_semantic: f64,
  pub weighted_graph: f64,
  pub weighted_recency: f64,
  pub weighted_usage: f64,
  pub total: f64,
}

/// Explain a composite activation score using explicit signal and weight
/// values. The stateful graph-level implementation lives in polypack-graph.
#[napi]
#[allow(clippy::too_many_arguments)]
pub fn score_breakdown(
  semantic: f64,
  graph: f64,
  recency: f64,
  usage: f64,
  semantic_weight: Option<f64>,
  graph_weight: Option<f64>,
  recency_weight: Option<f64>,
  usage_weight: Option<f64>,
) -> ActivationScoreBreakdown {
  let ws = semantic_weight.unwrap_or(1.0);
  let wg = graph_weight.unwrap_or(1.0);
  let wr = recency_weight.unwrap_or(1.0);
  let wu = usage_weight.unwrap_or(1.0);
  let weighted_semantic = ws * semantic;
  let weighted_graph = wg * graph;
  let weighted_recency = wr * recency;
  let weighted_usage = wu * usage;
  ActivationScoreBreakdown { semantic, graph, recency, usage, weighted_semantic, weighted_graph, weighted_recency, weighted_usage, total: weighted_semantic + weighted_graph + weighted_recency + weighted_usage }
}
