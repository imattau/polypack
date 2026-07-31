//! Run the shared query-plan conformance fixtures against the Rust executor.

use polypack_core::model::{Edge, Node};
use polypack_core::query::QueryPlan;
use polypack_core::query_exec::{aggregate, execute, GraphSnapshot};
use serde_json::Value;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/conformance")
}

fn fixture_files() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(fixtures_dir())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().map(|e| e == "json").unwrap_or(false))
        .collect();
    files.sort();
    files
}

fn to_node(value: &Value) -> Node {
    let mut node: Node = serde_json::from_value(value.clone()).unwrap();
    if let Some(v) = node.vector.as_mut() {
        // vectors stay as-is; validate nothing here
        let _ = v;
    }
    node
}

fn to_edge(value: &Value) -> Edge {
    let source = value["source"].as_str().unwrap().to_string();
    let target = value["target"].as_str().unwrap().to_string();
    let edge_type = value["type"].as_str().unwrap().to_string();
    let data = value
        .get("data")
        .map(|d| d.as_object().cloned().unwrap_or_default());
    Edge {
        id: format!("{source}::{edge_type}::{target}"),
        source,
        target,
        edge_type,
        data,
        created_at: 0,
    }
}

fn run_query_fixture(path: &PathBuf) {
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let name = fixture["name"].as_str().unwrap();

    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    if let Some(setup) = fixture.get("setup") {
        if let Some(nodes_json) = setup.get("nodes").and_then(|v| v.as_array()) {
            nodes = nodes_json.iter().map(to_node).collect();
        }
        if let Some(edges_json) = setup.get("edges").and_then(|v| v.as_array()) {
            edges = edges_json.iter().map(to_edge).collect();
        }
    }
    let snap = GraphSnapshot::new(nodes, edges);
    let expect = &fixture["expect"];

    if let Some(queries) = expect.get("queries").and_then(|v| v.as_array()) {
        for spec in queries {
            let plan: QueryPlan = serde_json::from_value(spec["plan"].clone()).unwrap_or_else(|e| {
                panic!("{name}: failed to deserialize plan: {e}");
            });
            let got = execute(&snap, &plan, None).unwrap();
            let want: Vec<String> = spec["resultIds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            assert_eq!(got, want, "{name}: resultIds mismatch");
        }
    }

    if let Some(agg) = expect.get("aggregate") {
        let plan: QueryPlan = agg
            .get("plan")
            .map(|p| serde_json::from_value(p.clone()).unwrap())
            .unwrap_or_default();
        let field = agg["field"].as_str().unwrap();
        let op = agg["op"].as_str().unwrap();
        let (value, count) = aggregate(&snap, &plan, field, op).unwrap();
        assert_eq!((value, count), (agg["value"].as_f64().unwrap(), agg["count"].as_u64().unwrap() as usize), "{name}: aggregate mismatch");
    }
}

#[test]
fn query_plan_fixtures_pass() {
    let mut ran = 0;
    for file in fixture_files() {
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        let expect = &fixture["expect"];
        if expect.get("queries").is_some() || expect.get("aggregate").is_some() {
            run_query_fixture(&file);
            ran += 1;
        }
    }
    assert!(ran >= 4, "expected at least 4 query fixtures, ran {ran}");
}
