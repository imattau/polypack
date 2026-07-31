#![cfg_attr(
  all(not(debug_assertions), target_os = "windows"),
  windows_subsystem = "windows"
)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use polypack_core::hnsw::{HnswConfig, HnswIndex};
use polypack_core::vector::{DistanceFn, ExactIndex};
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
