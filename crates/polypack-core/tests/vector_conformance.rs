//! Shared exact/ANN vector conformance against the language-neutral fixtures.

use polypack_core::error::PolypackError;
use polypack_core::hnsw::{HnswConfig, HnswIndex};
use polypack_core::vector::{DistanceFn, ExactIndex};
use serde_json::Value;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/conformance").join(name)
}

fn read_fixture(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture_path(name)).unwrap()).unwrap()
}

#[test]
fn exact_vector_fixture_passes() {
    let fixture = read_fixture("exact-vector.json");
    let mut index = ExactIndex::new(DistanceFn::Cosine);
    for node in fixture["setup"]["nodes"].as_array().unwrap() {
        index.add(node["id"].as_str().unwrap(), node["vector"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>().as_slice()).unwrap();
    }
    for search in fixture["expect"]["exactSearches"].as_array().unwrap() {
        let vector = search["vector"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
        let result = index.query(&vector, search["topK"].as_u64().unwrap() as usize, 0.0);
        if search.get("expectError").is_some() {
            assert!(matches!(result, Err(PolypackError::DimensionMismatch { .. })));
            continue;
        }
        let ids: Vec<String> = result.unwrap().into_iter().map(|row| row.id).collect();
        let expected: Vec<String> = search["resultIds"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        assert_eq!(ids, expected);
    }
}

#[test]
fn ann_fixtures_pass_with_declared_overlap_tolerance() {
    for name in ["hnsw-basic.json", "hnsw-churn.json"] {
        let fixture = read_fixture(name);
        let mut index = HnswIndex::new(HnswConfig { ef_search: 300, ..Default::default() }, 42).unwrap();
        for node in fixture["setup"]["nodes"].as_array().unwrap() {
            let vector = node["vector"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
            index.add(node["id"].as_str().unwrap(), &vector).unwrap();
        }
        for operation in fixture["operations"].as_array().unwrap() {
            let vector = operation.get("vector").and_then(Value::as_array).map(|values| values.iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>());
            match operation["op"].as_str().unwrap() {
                "hnswAdd" => index.add(operation["id"].as_str().unwrap(), &vector.unwrap()).unwrap(),
                "hnswRemove" => index.remove(operation["id"].as_str().unwrap()),
                "hnswUpdate" => index.update(operation["id"].as_str().unwrap(), &vector.unwrap()).unwrap(),
                other => panic!("{name}: unsupported operation {other}"),
            }
        }
        for search in fixture["expect"]["hnswSearches"].as_array().unwrap() {
            let vector = search["vector"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
            let got: Vec<String> = index.query(&vector, search["topK"].as_u64().unwrap() as usize, 0.0).unwrap().into_iter().map(|row| row.id).collect();
            let expected: Vec<String> = search["resultIds"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
            let overlap = got.iter().filter(|id| expected.contains(id)).count();
            assert!(overlap >= search["minOverlap"].as_u64().unwrap() as usize, "{name}: ANN overlap {overlap} < minimum");
        }
    }
}
