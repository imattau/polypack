//! Transaction semantics against the shared language-neutral fixture.

use polypack_core::{InMemoryStorage, Node, StoreConfig};
use polypack_graph::{Graph, GraphConfig};
use serde_json::Value;
use std::path::PathBuf;

#[test]
fn transaction_fixture_passes() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/database-core/transaction.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut graph = Graph::open(
        Box::new(InMemoryStorage::new()),
        StoreConfig::default(),
        GraphConfig::default(),
    )
    .unwrap();
    graph
        .add_node(serde_json::from_value::<Node>(fixture["setup"]["nodes"][0].clone()).unwrap())
        .unwrap();

    let tx = &fixture["transaction"];
    graph
        .transaction(|graph| {
            let patch = &tx["patch"];
            graph.patch_node(
                patch["id"].as_str().unwrap(),
                patch["set"].as_object().cloned().unwrap_or_default(),
                Vec::new(),
                patch["increment"].as_object().cloned().unwrap_or_default(),
                serde_json::Map::new(),
                patch["expectedRevision"].as_u64(),
            )?;
            graph.add_node(serde_json::from_value::<Node>(tx["addNode"].clone()).unwrap())?;
            let edge = &tx["addEdge"];
            graph.add_edge(
                edge["source"].as_str().unwrap(),
                edge["type"].as_str().unwrap(),
                edge["target"].as_str().unwrap(),
                None,
                polypack_graph::EdgeOwnership::Reference,
            )?;
            assert_eq!(
                graph
                    .get_node(tx["readYourWrites"]["id"].as_str().unwrap())
                    .unwrap()
                    .data["count"]
                    .as_f64(),
                Some(tx["readYourWrites"]["count"].as_f64().unwrap()),
            );
            Ok(())
        })
        .unwrap();

    assert_eq!(
        graph.size(),
        fixture["expect"]["nodeCount"].as_u64().unwrap() as usize
    );
    assert_eq!(
        graph.get_node("person-1").unwrap().data["count"].as_f64(),
        Some(2.0)
    );
    assert_eq!(
        graph.get_node("person-1").unwrap().revision,
        fixture["expect"]["person1Revision"].as_u64().unwrap()
    );
    assert_eq!(
        graph.get_edge_targets("person-1", "RELATED_TO"),
        vec!["person-2"]
    );

    let rollback = &fixture["rollback"];
    let result: Result<(), _> = graph.transaction(|graph| {
        let patch = &rollback["patch"];
        graph.patch_node(
            patch["id"].as_str().unwrap(),
            patch["set"].as_object().cloned().unwrap_or_default(),
            Vec::new(),
            serde_json::Map::new(),
            serde_json::Map::new(),
            patch["expectedRevision"].as_u64(),
        )?;
        graph.add_node(serde_json::from_value::<Node>(rollback["addNode"].clone()).unwrap())?;
        Err(polypack_core::PolypackError::InvalidArgument(
            "rollback fixture failure".into(),
        ))
    });
    assert!(result.is_err());
    assert_eq!(
        graph.size(),
        fixture["expect"]["rollbackCount"].as_u64().unwrap() as usize
    );
    assert!(graph.get_node("temporary").is_none());
    assert_eq!(
        graph.get_node("person-1").unwrap().data["count"].as_f64(),
        Some(2.0)
    );
    assert_eq!(
        graph.get_node("person-1").unwrap().revision,
        fixture["expect"]["rollbackRevision"].as_u64().unwrap()
    );
}
