//! Schema and index guarantees against the shared fixture.

use polypack_core::{InMemoryStorage, Node, PolypackError, StoreConfig};
use polypack_graph::{Graph, GraphConfig, IndexDefinition, NodeTypeDefinition};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn schema_and_unique_index_fixture_passes() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/database-core/schema-and-indexes.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut graph = Graph::open(
        Box::new(InMemoryStorage::new()),
        StoreConfig::default(),
        GraphConfig::default(),
    )
    .unwrap();
    let node_type = &fixture["nodeType"];
    graph
        .register_node_type(
            node_type["name"].as_str().unwrap(),
            NodeTypeDefinition {
                required_fields: node_type["requiredFields"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect(),
                data_types: node_type["dataTypes"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                    .collect::<HashMap<_, _>>(),
            },
        )
        .unwrap();
    let index = &fixture["index"];
    graph
        .define_index(IndexDefinition {
            name: index["name"].as_str().unwrap().to_string(),
            node_type: Some(index["nodeType"].as_str().unwrap().to_string()),
            fields: index["fields"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect(),
            unique: index["unique"].as_bool().unwrap(),
            sparse: false,
        })
        .unwrap();
    graph
        .add_node(serde_json::from_value::<Node>(fixture["validNode"].clone()).unwrap())
        .unwrap();
    let invalid =
        graph.add_node(serde_json::from_value::<Node>(fixture["invalidNode"].clone()).unwrap());
    assert!(matches!(invalid, Err(PolypackError::InvalidArgument(_))));
    let duplicate =
        graph.add_node(serde_json::from_value::<Node>(fixture["duplicateNode"].clone()).unwrap());
    assert!(matches!(duplicate, Err(PolypackError::InvalidArgument(_))));
    assert_eq!(
        graph.size(),
        fixture["expect"]["nodeCount"].as_u64().unwrap() as usize
    );
    assert!(graph
        .get_node(fixture["expect"]["presentId"].as_str().unwrap())
        .is_some());
}
