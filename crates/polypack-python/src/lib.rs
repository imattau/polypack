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
