//! Snapshot isolation guarantees against the shared fixture.

use polypack_core::{execute, InMemoryStorage, Node, QueryPlan, StoreConfig};
use polypack_graph::{Graph, GraphConfig};
use serde_json::Value;
use std::path::PathBuf;

#[test]
fn snapshot_fixture_passes() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/database-core/snapshot-isolation.json");
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
    let snapshot = graph.snapshot();
    graph
        .add_node(serde_json::from_value::<Node>(fixture["mutation"]["add"].clone()).unwrap())
        .unwrap();
    graph
        .remove_node(fixture["mutation"]["remove"].as_str().unwrap())
        .unwrap();
    let mut snapshot_ids = execute(&snapshot, &QueryPlan::default(), None).unwrap();
    snapshot_ids.sort();
    let mut live_ids = graph.query().ids();
    live_ids.sort();
    assert_eq!(
        snapshot_ids,
        fixture["expect"]["snapshotIds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        live_ids,
        fixture["expect"]["liveIds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>()
    );
}
