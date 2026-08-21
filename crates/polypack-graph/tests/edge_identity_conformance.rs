//! Independent edge identity guarantees against the shared fixture.

use polypack_core::{InMemoryStorage, Node, StoreConfig};
use polypack_graph::{EdgeOwnership, Graph, GraphConfig};
use serde_json::Value;
use std::path::PathBuf;

#[test]
fn parallel_edges_fixture_passes() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/database-core/parallel-edges.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut graph = Graph::open(
        Box::new(InMemoryStorage::new()),
        StoreConfig::default(),
        GraphConfig::default(),
    )
    .unwrap();
    for node in fixture["nodes"].as_array().unwrap() {
        graph
            .add_node(serde_json::from_value::<Node>(node.clone()).unwrap())
            .unwrap();
    }
    for edge in fixture["edges"].as_array().unwrap() {
        graph
            .add_edge_with_id(
                edge["id"].as_str().unwrap(),
                edge["source"].as_str().unwrap(),
                edge["type"].as_str().unwrap(),
                edge["target"].as_str().unwrap(),
                edge["data"].as_object().cloned(),
                EdgeOwnership::Reference,
            )
            .unwrap();
    }
    let update = &fixture["update"];
    graph
        .update_edge_if_revision(
            update["id"].as_str().unwrap(),
            update["data"].as_object().cloned(),
            None,
            update["expectedRevision"].as_u64().unwrap(),
        )
        .unwrap();
    let remove = &fixture["remove"];
    assert!(graph
        .remove_edge_if_revision(
            remove["id"].as_str().unwrap(),
            remove["expectedRevision"].as_u64().unwrap()
        )
        .unwrap());
    let edges = graph.get_edges("a", Some("RELATED"));
    assert_eq!(
        edges.iter().map(|edge| edge.id.clone()).collect::<Vec<_>>(),
        vec!["claim-1"]
    );
    assert_eq!(
        edges[0].revision,
        fixture["expect"]["revision"].as_u64().unwrap()
    );
    assert_eq!(
        edges[0].data.as_ref().unwrap()["confidence"].as_f64(),
        Some(0.9)
    );
}
