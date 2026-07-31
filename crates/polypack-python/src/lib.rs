//! PyO3 bindings exposing the polypack-core vector engine to Python.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1, PyArray1};
use polypack_core::error::PolypackError as CoreError;
use polypack_core::hnsw::{HnswConfig, HnswIndex as CoreHnswIndex};
use polypack_core::vector::{DistanceFn, ExactIndex as CoreExactIndex};
use pyo3::exceptions::{PyException, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

pyo3::create_exception!(polypack, PolypackError, PyException, "Base polypack error");
pyo3::create_exception!(polypack, PolypackValueError, PolypackError, "Invalid argument");
pyo3::create_exception!(polypack, PolypackDimensionError, PolypackError, "Vector dimension mismatch");
pyo3::create_exception!(polypack, PolypackClosedError, PolypackError, "Operation on a closed store");
pyo3::create_exception!(polypack, PolypackVersionError, PolypackError, "Unsupported format version");
pyo3::create_exception!(polypack, PolypackCorruptDataError, PolypackError, "Corrupt data");
pyo3::create_exception!(polypack, PolypackStorageError, PolypackError, "Storage failure");

fn to_pyerr(e: CoreError) -> PyErr {
    match e {
        CoreError::InvalidArgument(m) => PolypackValueError::new_err(m),
        CoreError::DimensionMismatch { .. } => PolypackDimensionError::new_err(e.to_string()),
        CoreError::RangeOutOfBounds(m) => PolypackValueError::new_err(m),
        CoreError::Closed => PolypackClosedError::new_err(e.to_string()),
        CoreError::FormatVersion(v) => PolypackVersionError::new_err(v.to_string()),
        CoreError::CorruptData(m) => PolypackCorruptDataError::new_err(m),
        CoreError::Storage(m) => PolypackStorageError::new_err(m),
        CoreError::NotImplemented(m) => PyNotImplementedError::new_err(m),
    }
}

/// Convert a 1-D sequence (list/tuple/numpy array) into a `Vec<f64>`.
fn to_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(arr.as_slice()?.to_vec());
    }
    obj.extract::<Vec<f64>>()
        .map_err(|_| PyValueError::new_err("vector must be a sequence of floats"))
}

fn vector_to_py<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(v).into_pyarray(py)
}

fn pair<'py>(py: Python<'py>, a: String, b: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let items: Vec<Bound<'py, PyAny>> = vec![a.into_pyobject(py)?.into_any(), b];
    PyTuple::new(py, items)
}

fn score_pair<'py>(py: Python<'py>, id: String, score: f64) -> PyResult<Bound<'py, PyTuple>> {
    let items: Vec<Bound<'py, PyAny>> = vec![
        id.into_pyobject(py)?.into_any(),
        score.into_pyobject(py)?.into_any(),
    ];
    PyTuple::new(py, items)
}

#[pyclass(module = "polypack")]
struct ExactIndex {
    inner: CoreExactIndex,
}

#[pymethods]
impl ExactIndex {
    #[new]
    #[pyo3(signature = (distance="cosine".to_string()))]
    fn new(distance: String) -> Self {
        let distance = match distance.as_str() {
            "euclidean" => DistanceFn::Euclidean,
            _ => DistanceFn::Cosine,
        };
        ExactIndex {
            inner: CoreExactIndex::new(distance),
        }
    }

    fn add(&mut self, id: String, vector: Bound<'_, PyAny>) -> PyResult<()> {
        let v = to_vec(&vector)?;
        self.inner.add(&id, &v).map_err(to_pyerr)
    }

    #[pyo3(signature = (ids, vectors))]
    fn add_many(&mut self, ids: Vec<String>, vectors: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        if ids.len() != vectors.len() {
            return Err(PolypackValueError::new_err(format!(
                "ids and vectors length mismatch: {} != {}",
                ids.len(),
                vectors.len()
            )));
        }
        let mut rows = Vec::with_capacity(ids.len());
        for v in vectors {
            rows.push(to_vec(&v)?);
        }
        for (id, v) in ids.iter().zip(rows.iter()) {
            self.inner.add(id, v).map_err(to_pyerr)?;
        }
        Ok(())
    }

    fn remove(&mut self, id: String) {
        self.inner.remove(&id)
    }

    fn remove_many(&mut self, ids: Vec<String>) {
        for id in ids {
            self.inner.remove(&id);
        }
    }

    #[pyo3(signature = (vector, top_k, threshold=0.0))]
    fn query(
        &self,
        py: Python<'_>,
        vector: Bound<'_, PyAny>,
        top_k: usize,
        threshold: f64,
    ) -> PyResult<Py<PyList>> {
        let v = to_vec(&vector)?;
        let results = py
            .allow_threads(|| self.inner.query(&v, top_k, threshold))
            .map_err(to_pyerr)?;
        let list = PyList::empty(py);
        for s in results {
            list.append(score_pair(py, s.id, s.score)?)?;
        }
        Ok(list.unbind())
    }

    fn clear(&mut self) {
        self.inner.clear()
    }

    fn has(&self, id: String) -> bool {
        self.inner.has(&id)
    }

    fn get<'py>(&self, py: Python<'py>, id: String) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.get(&id).map(|v| vector_to_py(py, v.to_vec()))
    }

    fn entries(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for (id, v) in self.inner.entries() {
            list.append(pair(py, id, vector_to_py(py, v).into_any())?)?;
        }
        Ok(list.unbind())
    }

    fn __len__(&self) -> usize {
        self.inner.size()
    }
}

#[pyclass(module = "polypack")]
struct HnswIndex {
    inner: CoreHnswIndex,
}

#[pymethods]
impl HnswIndex {
    #[new]
    #[pyo3(signature = (m=16, mmax0=32, ef_construction=200, ef_search=200, level_seed=7))]
    fn new(
        m: usize,
        mmax0: usize,
        ef_construction: usize,
        ef_search: usize,
        level_seed: u32,
    ) -> Self {
        let config = HnswConfig {
            m,
            mmax0,
            ef_construction,
            ef_search,
        };
        HnswIndex {
            inner: CoreHnswIndex::new(config, level_seed),
        }
    }

    fn add(&mut self, id: String, vector: Bound<'_, PyAny>) -> PyResult<()> {
        let v = to_vec(&vector)?;
        self.inner.add(&id, &v).map_err(to_pyerr)
    }

    fn update(&mut self, id: String, vector: Bound<'_, PyAny>) -> PyResult<()> {
        let v = to_vec(&vector)?;
        self.inner.update(&id, &v).map_err(to_pyerr)
    }

    #[pyo3(signature = (ids, vectors))]
    fn add_many(&mut self, ids: Vec<String>, vectors: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        if ids.len() != vectors.len() {
            return Err(PolypackValueError::new_err(format!(
                "ids and vectors length mismatch: {} != {}",
                ids.len(),
                vectors.len()
            )));
        }
        let mut rows = Vec::with_capacity(ids.len());
        for v in vectors {
            rows.push(to_vec(&v)?);
        }
        for (id, v) in ids.iter().zip(rows.iter()) {
            self.inner.add(id, v).map_err(to_pyerr)?;
        }
        Ok(())
    }

    fn remove(&mut self, id: String) {
        self.inner.remove(&id)
    }

    fn remove_many(&mut self, ids: Vec<String>) {
        for id in ids {
            self.inner.remove(&id);
        }
    }

    #[pyo3(signature = (vector, top_k, threshold=0.0))]
    fn query(
        &self,
        py: Python<'_>,
        vector: Bound<'_, PyAny>,
        top_k: usize,
        threshold: f64,
    ) -> PyResult<Py<PyList>> {
        let v = to_vec(&vector)?;
        let results = py
            .allow_threads(|| self.inner.query(&v, top_k, threshold))
            .map_err(to_pyerr)?;
        let list = PyList::empty(py);
        for s in results {
            list.append(score_pair(py, s.id, s.score)?)?;
        }
        Ok(list.unbind())
    }

    fn clear(&mut self) {
        self.inner.clear()
    }

    fn has(&self, id: String) -> bool {
        self.inner.has(&id)
    }

    fn get<'py>(&self, py: Python<'py>, id: String) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.get(&id).map(|v| vector_to_py(py, v.to_vec()))
    }

    fn entries(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let mut rows: Vec<(String, Vec<f64>)> = self
            .inner
            .nodes()
            .iter()
            .map(|(id, v)| (id.clone(), v.clone()))
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        let list = PyList::empty(py);
        for (id, v) in rows {
            list.append(pair(py, id, vector_to_py(py, v).into_any())?)?;
        }
        Ok(list.unbind())
    }

    fn __len__(&self) -> usize {
        self.inner.size()
    }
}

#[pyfunction]
fn engine_info() -> (String, String, String) {
    ("python".to_string(), "rust-native".to_string(), "host".to_string())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ExactIndex>()?;
    m.add_class::<HnswIndex>()?;
    m.add_class::<NativeStore>()?;
    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    m.add("PolypackError", m.py().get_type::<PolypackError>())?;
    m.add("PolypackValueError", m.py().get_type::<PolypackValueError>())?;
    m.add("PolypackDimensionError", m.py().get_type::<PolypackDimensionError>())?;
    m.add("PolypackClosedError", m.py().get_type::<PolypackClosedError>())?;
    m.add("PolypackVersionError", m.py().get_type::<PolypackVersionError>())?;
    m.add("PolypackCorruptDataError", m.py().get_type::<PolypackCorruptDataError>())?;
    m.add("PolypackStorageError", m.py().get_type::<PolypackStorageError>())?;
    Ok(())
}

// ── Storage / NativeStore ──

use polypack_core::storage::{Durability, Store as CoreStore, StoreConfig, Storage};
use std::sync::Mutex;

/// Host storage implemented by a Python object exposing `read(name) -> bytes|None`,
/// `write(name, bytes)`, `append(name, bytes)`, `delete(name)`, `exists(name) -> bool`.
struct PythonStorage(Py<PyAny>);

impl Storage for PythonStorage {
    fn read(&self, name: &str) -> polypack_core::Result<Option<Vec<u8>>> {
        Python::with_gil(|py| {
            let obj = self.0.bind(py);
            let result = obj
                .call_method1("read", (name,))
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            if result.is_none() {
                return Ok(None);
            }
            let bytes: Vec<u8> = result
                .extract()
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            Ok(Some(bytes))
        })
    }
    fn write(&mut self, name: &str, data: &[u8]) -> polypack_core::Result<()> {
        Python::with_gil(|py| {
            self.0
                .bind(py)
                .call_method1("write", (name, data))
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            Ok(())
        })
    }
    fn append(&mut self, name: &str, data: &[u8]) -> polypack_core::Result<()> {
        Python::with_gil(|py| {
            self.0
                .bind(py)
                .call_method1("append", (name, data))
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            Ok(())
        })
    }
    fn delete(&mut self, name: &str) -> polypack_core::Result<()> {
        Python::with_gil(|py| {
            self.0
                .bind(py)
                .call_method1("delete", (name,))
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            Ok(())
        })
    }
    fn exists(&self, name: &str) -> polypack_core::Result<bool> {
        Python::with_gil(|py| {
            let obj = self.0.bind(py);
            let result = obj
                .call_method1("exists", (name,))
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            result
                .extract()
                .map_err(|e| CoreError::Storage(e.to_string()))
        })
    }
}

#[pyclass(module = "polypack")]
struct NativeStore {
    inner: Mutex<CoreStore>,
}

fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if let Ok(dict) = obj.downcast::<pyo3::types::PyDict>() {
        let mut out = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            out.insert(key, py_to_json(&v)?);
        }
        return Ok(serde_json::Value::Object(out));
    }
    if let Ok(arr) = obj.downcast::<PyList>() {
        let mut out = Vec::new();
        for item in arr.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Array(out));
    }
    if let Ok(tup) = obj.downcast::<PyTuple>() {
        let mut out = Vec::new();
        for item in tup.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Array(out));
    }
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(serde_json::Value::Number(i.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::Number::from_f64(f).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    Err(PolypackValueError::new_err("value is not JSON-compatible"))
}

fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(pyo3::types::PyBool::new(py, *b).to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = pyo3::types::PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

fn py_to_polypack<T: serde::de::DeserializeOwned>(value: &Bound<'_, PyAny>) -> PyResult<T> {
    let json = py_to_json(value)?;
    serde_json::from_value(json).map_err(|e| PolypackValueError::new_err(e.to_string()))
}

#[pymethods]
impl NativeStore {
    #[new]
    #[pyo3(signature = (storage, compact_threshold=10000))]
    fn new(storage: Py<PyAny>, compact_threshold: usize) -> Self {
        let config = StoreConfig {
            compact_threshold,
            durability: Durability::Process,
        };
        NativeStore {
            inner: Mutex::new(CoreStore::new(Box::new(PythonStorage(storage)), config)),
        }
    }

    #[pyo3(signature = (put_nodes=vec![], delete_node_ids=vec![], put_edges=vec![], delete_edge_ids=vec![], put_vectors=vec![], delete_vector_ids=vec![]))]
    fn apply(
        &self,
        put_nodes: Vec<Bound<'_, PyAny>>,
        delete_node_ids: Vec<String>,
        put_edges: Vec<Bound<'_, PyAny>>,
        delete_edge_ids: Vec<String>,
        put_vectors: Vec<Bound<'_, PyAny>>,
        delete_vector_ids: Vec<String>,
    ) -> PyResult<()> {
        let mut batch = polypack_core::model::ChangeBatch {
            delete_node_ids,
            delete_edge_ids,
            delete_vector_ids,
            ..Default::default()
        };
        for n in put_nodes {
            batch.put_nodes.push(py_to_polypack(&n)?);
        }
        for e in put_edges {
            batch.put_edges.push(py_to_polypack(&e)?);
        }
        for v in put_vectors {
            let entry: polypack_core::model::VectorEntry = py_to_polypack(&v)?;
            batch.put_vectors.push(entry);
        }
        self.inner.lock().unwrap().apply(&batch).map_err(to_pyerr)
    }

    fn all_node_ids(&self) -> PyResult<Vec<String>> {
        self.inner.lock().unwrap().node_ids().map_err(to_pyerr)
    }

    fn get_node(&self, py: Python<'_>, id: String) -> PyResult<Option<PyObject>> {
        match self.inner.lock().unwrap().get_node(&id).map_err(to_pyerr)? {
            Some(node) => {
                let json = serde_json::to_value(node).map_err(|e| PolypackStorageError::new_err(e.to_string()))?;
                Ok(Some(json_to_py(py, &json)?))
            }
            None => Ok(None),
        }
    }

    fn all_edges(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let edges = self.inner.lock().unwrap().edges_snapshot().map_err(to_pyerr)?;
        let list = PyList::empty(py);
        for (_id, edge) in edges {
            let json = serde_json::to_value(edge).map_err(|e| PolypackStorageError::new_err(e.to_string()))?;
            list.append(json_to_py(py, &json)?)?;
        }
        Ok(list.unbind())
    }

    fn all_vectors(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let vectors = self.inner.lock().unwrap().vectors_snapshot().map_err(to_pyerr)?;
        let list = PyList::empty(py);
        for (id, v) in vectors {
            let arr = numpy::ndarray::Array1::from_vec(v).into_pyarray(py);
            list.append(pair(py, id, arr.into_any())?)?;
        }
        Ok(list.unbind())
    }

    fn compact(&self) -> PyResult<()> {
        self.inner.lock().unwrap().compact().map_err(to_pyerr)
    }

    fn close(&self) -> PyResult<()> {
        self.inner.lock().unwrap().close().map_err(to_pyerr)
    }
}
