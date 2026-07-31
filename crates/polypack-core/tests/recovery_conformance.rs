//! Run the shared recovery fixtures (`fixtures/recovery/*.json`) against the
//! Rust storage state machine with an in-memory byte store.

use polypack_core::model::{Edge, Node};
use polypack_core::storage::format::{encode_snapshot, encode_wal};
use polypack_core::storage::wal::WalEntry;
use polypack_core::storage::{InMemoryStorage, Store, StoreConfig, Storage, SNAPSHOT_FILE, WAL_FILE};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/recovery")
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

fn wal_entries(value: &Value) -> Vec<WalEntry> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let kind = entry["kind"].as_str().unwrap();
            match kind {
                "putNode" => WalEntry::PutNode(serde_json::from_value(entry["node"].clone()).unwrap()),
                "deleteNode" => WalEntry::DeleteNode(entry["id"].as_str().unwrap().to_string()),
                "putEdge" => WalEntry::PutEdge(serde_json::from_value(entry["edge"].clone()).unwrap()),
                "deleteEdge" => WalEntry::DeleteEdge(entry["id"].as_str().unwrap().to_string()),
                "putVector" => WalEntry::PutVector {
                    id: entry["id"].as_str().unwrap().to_string(),
                    vector: entry["vector"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect(),
                },
                "deleteVector" => WalEntry::DeleteVector(entry["id"].as_str().unwrap().to_string()),
                "clearAll" => WalEntry::ClearAll,
                other => panic!("unknown wal kind {other}"),
            }
        })
        .collect()
}

fn run_fixture(path: &PathBuf) {
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let store_json = &fixture["store"];

    let storage = Arc::new(Mutex::new(InMemoryStorage::new()));

    if let Some(snapshot) = store_json.get("snapshot") {
        let nodes: Vec<(String, Node)> = snapshot["nodes"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|n| {
                        let node: Node = serde_json::from_value(n.clone()).unwrap();
                        (node.id.clone(), node)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let edges: Vec<(String, Edge)> = snapshot["edges"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|e| {
                        let edge: Edge = serde_json::from_value(e.clone()).unwrap();
                        (edge.id.clone(), edge)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let vectors: Vec<(String, Vec<f64>)> = snapshot["vectors"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|pair| {
                        let id = pair[0].as_str().unwrap().to_string();
                        let vector = pair[1].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
                        (id, vector)
                    })
                    .collect()
            })
            .unwrap_or_default();
        storage.lock().unwrap().write(SNAPSHOT_FILE, &encode_snapshot(&nodes, &edges, &vectors)).unwrap();
    }

    if let Some(wal) = store_json.get("wal") {
        let mut data = encode_wal(&wal_entries(wal));
        if let Some(hex_tail) = store_json.get("corruptTailHex").and_then(|v| v.as_str()) {
            let tail = hex::decode(hex_tail).unwrap();
            data.extend_from_slice(&tail);
        }
        storage.lock().unwrap().write(WAL_FILE, &data).unwrap();
    }

    let expect = &fixture["expect"];
    let mut store = Store::new(Box::new(storage.clone()), StoreConfig::default());
    let mut ids = store.node_ids().unwrap();
    ids.sort();
    let mut expected: Vec<String> = expect["presentNodeIds"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    expected.sort();
    assert_eq!(ids, expected, "presentNodeIds mismatch in {:?}", path.file_name().unwrap());
    if let Some(absent) = expect.get("absentNodeIds") {
        for id in absent.as_array().unwrap() {
            let id = id.as_str().unwrap();
            assert!(!ids.contains(&id.to_string()), "expected {id} absent in {:?}", path.file_name().unwrap());
        }
    }
    if let Some(vectors) = expect.get("vectors").and_then(|v| v.as_object()) {
        for (id, vector) in vectors {
            let got = store.get_vector(id).unwrap();
            assert_eq!(got, Some(vector.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()), "vector {id} mismatch");
        }
    }
    store.close().unwrap();

    let storage = storage.lock().unwrap();
    let snap_present = storage.exists(SNAPSHOT_FILE).unwrap();
    assert_eq!(snap_present, expect["snapshotPresentAfterRecovery"].as_bool().unwrap());
    let wal = storage.read(WAL_FILE).unwrap();
    let wal_empty = wal.map(|w| w.is_empty()).unwrap_or(true);
    assert_eq!(wal_empty, expect["walRemovedAfterRecovery"].as_bool().unwrap());
}

#[test]
fn recovery_fixtures_pass() {
    let files = fixture_files();
    assert!(files.len() >= 4, "expected recovery fixtures, found {}", files.len());
    for file in files {
        run_fixture(&file);
    }
}

mod hex {
    pub fn decode(s: &str) -> Result<Vec<u8>, ()> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ()))
            .collect()
    }
}
