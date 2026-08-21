//! Database-core conformance cases shared with the TypeScript and Python runners.

use polypack_core::{InMemoryStorage, Node, PolypackError, StoreConfig};
use polypack_graph::{Graph, GraphConfig, MigrationDefinition, MigrationOptions};
use serde_json::{Map, Value};
use std::path::PathBuf;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/conformance/revisions-and-patches.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn mutation_log_fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/database-core/durable-mutation-log.json");
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

#[test]
fn durable_mutation_log_fixture_passes() {
    let fixture = mutation_log_fixture();
    assert_eq!(fixture["schemaVersion"], 1);
    let mut graph = Graph::open(
        Box::new(InMemoryStorage::new()),
        StoreConfig::default(),
        GraphConfig::default(),
    )
    .unwrap();
    let transaction = &fixture["transaction"];
    let node = serde_json::from_value::<Node>(transaction["node"].clone()).unwrap();
    graph
        .transaction_with_identity(
            Some(transaction["operationId"].as_str().unwrap().to_string()),
            |tx| tx.add_node(node),
        )
        .unwrap();

    let records = graph.mutation_log().unwrap();
    let record = records.last().unwrap();
    assert_eq!(graph.latest_mutation_sequence().unwrap(), fixture["expect"]["latestSequence"].as_u64().unwrap());
    assert_eq!(record.operation_id, fixture["expect"]["operationId"]);
    assert_eq!(record.operations[0].operation_type, "putNode");
    assert_eq!(record.operations[0].payload["id"], fixture["expect"]["nodeId"]);
}

#[test]
fn migration_fixture_passes() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/database-core/migration.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut graph = Graph::open(
        Box::new(InMemoryStorage::new()),
        StoreConfig::default(),
        GraphConfig::default(),
    )
    .unwrap();
    for node in fixture["nodes"].as_array().unwrap() {
        graph.add_node(serde_json::from_value::<Node>(node.clone()).unwrap()).unwrap();
    }
    graph.register_migration(MigrationDefinition::new(1, 2, |mut node| {
        if let Some(name) = node.data.get("name").cloned() {
            node.data.insert("displayName".into(), name);
        }
        Ok(node)
    })).unwrap();
    let report = graph.migrate(1, 2, MigrationOptions { batch_size: 1, ..Default::default() }).unwrap();
    assert_eq!(report.migrated_nodes, fixture["expect"]["migrated"].as_u64().unwrap() as usize);
    let mut ids = graph.query().ids();
    ids.sort();
    assert_eq!(ids, fixture["expect"]["ids"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect::<Vec<_>>());
    for id in fixture["expect"]["ids"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()) {
        assert_eq!(graph.get_node(id).unwrap().data["displayName"], fixture["expect"]["displayNames"][id]);
    }
}

#[test]
fn resource_limits_fixture_passes() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/database-core/resource-limits.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let limits = &fixture["limits"];
    let mut graph = Graph::open(
        Box::new(InMemoryStorage::new()),
        StoreConfig::default(),
        GraphConfig {
            resource_limits: polypack_graph::GraphResourceLimits {
                max_vector_dimensions: limits["maxVectorDimensions"].as_u64().map(|value| value as usize),
                max_node_payload_bytes: limits["maxNodePayloadBytes"].as_u64().map(|value| value as usize),
                max_batch_size: limits["maxBatchSize"].as_u64().map(|value| value as usize),
            },
            ..GraphConfig::default()
        },
    )
    .unwrap();

    let payload = serde_json::from_value::<Node>(fixture["payloadNode"].clone()).unwrap();
    assert!(matches!(graph.add_node(payload), Err(PolypackError::ResourceLimit { name, .. }) if name == "maxNodePayloadBytes"));
    let vector = serde_json::from_value::<Node>(fixture["vectorNode"].clone()).unwrap();
    assert!(matches!(graph.add_node(vector), Err(PolypackError::ResourceLimit { name, .. }) if name == "maxVectorDimensions"));
    let batch = fixture["batchNodes"]
        .as_array()
        .unwrap()
        .iter()
        .cloned()
        .map(|value| serde_json::from_value::<Node>(value).unwrap())
        .collect();
    assert!(matches!(graph.add_nodes(batch), Err(PolypackError::ResourceLimit { name, .. }) if name == "maxBatchSize"));

    let transaction_nodes = &fixture["transactionNodes"];
    let result: polypack_core::Result<()> = graph.transaction(|tx| {
        tx.add_node(serde_json::from_value::<Node>(transaction_nodes[0].clone()).unwrap())?;
        tx.add_node(serde_json::from_value::<Node>(transaction_nodes[1].clone()).unwrap())?;
        Ok(())
    });
    assert!(matches!(result, Err(PolypackError::ResourceLimit { name, .. }) if name == "maxBatchSize"));
    for id in fixture["expect"]["absentNodeIds"].as_array().unwrap().iter().map(|value| value.as_str().unwrap()) {
        assert!(graph.get_node(id).is_none());
    }

    graph
        .transaction(|tx| tx.add_node(serde_json::from_value::<Node>(fixture["afterRollbackNode"].clone()).unwrap()))
        .unwrap();
    assert!(matches!(graph.patch_node(
        fixture["afterRollbackNode"]["id"].as_str().unwrap(),
        object(&fixture["oversizedPatch"]["set"]),
        Vec::new(),
        Map::new(),
        Map::new(),
        None,
    ), Err(PolypackError::ResourceLimit { name, .. }) if name == "maxNodePayloadBytes"));
    assert!(graph.get_node(fixture["afterRollbackNode"]["id"].as_str().unwrap()).unwrap().data.is_empty());
    for id in fixture["expect"]["presentNodeIds"].as_array().unwrap().iter().map(|value| value.as_str().unwrap()) {
        assert!(graph.get_node(id).is_some());
    }
}
