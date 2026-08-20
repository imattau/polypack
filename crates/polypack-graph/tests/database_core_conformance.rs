//! Database-core conformance cases shared with the TypeScript and Python runners.

use polypack_core::{InMemoryStorage, Node, PolypackError, StoreConfig};
use polypack_graph::{Graph, GraphConfig};
use serde_json::{Map, Value};
use std::path::PathBuf;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/conformance/revisions-and-patches.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn object(value: &Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

#[test]
fn revisions_and_patches_fixture_passes() {
    let fixture = fixture();
    assert_eq!(fixture["schemaVersion"], 1);
    assert_eq!(fixture["name"], "revisions-and-patches");
    let mut graph = Graph::open(
        Box::new(InMemoryStorage::new()),
        StoreConfig::default(),
        GraphConfig::default(),
    )
    .unwrap();

    for node in fixture["setup"]["nodes"].as_array().unwrap() {
        graph
            .add_node(serde_json::from_value::<Node>(node.clone()).unwrap())
            .unwrap();
    }

    for operation in fixture["operations"].as_array().unwrap() {
        let id = operation["id"].as_str().unwrap();
        let expected_revision = operation.get("expectedRevision").and_then(Value::as_u64);
        let result = match operation["op"].as_str().unwrap() {
            "patchNode" => {
                let patch = &operation["patch"];
                graph
                    .patch_node(
                        id,
                        object(patch.get("set").unwrap_or(&Value::Null)),
                        patch
                            .get("unset")
                            .and_then(Value::as_array)
                            .map(|paths| {
                                paths
                                    .iter()
                                    .map(|path| path.as_str().unwrap().to_string())
                                    .collect()
                            })
                            .unwrap_or_default(),
                        object(patch.get("increment").unwrap_or(&Value::Null)),
                        object(patch.get("compareAndSet").unwrap_or(&Value::Null)),
                        expected_revision,
                    )
                    .map(|_| ())
            }
            "updateNode" => graph
                .update_node_if_revision(
                    id,
                    expected_revision.unwrap(),
                    object(&operation["data"]),
                    None,
                    None,
                )
                .map(|_| ()),
            other => panic!("unsupported database-core operation {other}"),
        };

        if operation.get("expectError").and_then(Value::as_str) == Some("conflict") {
            assert!(
                matches!(result, Err(PolypackError::Conflict { .. })),
                "expected conflict, got {result:?}"
            );
        } else {
            result.unwrap();
        }
    }

    let node = graph.get_node("person-1").unwrap();
    assert_eq!(
        node.revision,
        fixture["expect"]["nodeRevision"]["person-1"]
            .as_u64()
            .unwrap()
    );
    assert_eq!(node.data["viewCount"].as_f64(), Some(3.0));
    assert_eq!(node.data["profile"]["displayName"], "M. Smith");
    assert!(!node.data.contains_key("temporary"));
    assert!(!node.data.contains_key("stale"));
}
