use polypack_core::model::{ChangeBatch, Node, VectorEntry};
use polypack_core::storage::{migrate_vector_precision, FileStorage, Store, StoreConfig, VectorPrecision};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(label: &str) -> PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("polypack-{label}-{}-{stamp}", std::process::id()))
}

#[test]
fn filesystem_precision_migration_keeps_backup_and_history() {
    let root = temp_path("migration-test");
    let store_dir = root.join("store");
    let backup_dir = root.join("backup");
    fs::create_dir_all(&store_dir).unwrap();
    let mut store = Store::new(Box::new(FileStorage::open(&store_dir, false).unwrap()), StoreConfig::default());
    let node = Node {
        id: "n".into(),
        node_type: "doc".into(),
        vector: Some(vec![1.0 / 3.0; 384]),
        inserted_at: 1,
        updated_at: 1,
        ..Default::default()
    };
    store.apply(&ChangeBatch {
        put_nodes: vec![node],
        put_vectors: vec![VectorEntry { id: "n".into(), vector: vec![1.0 / 3.0; 384] }],
        ..Default::default()
    }).unwrap();
    let sequence = store.latest_mutation_sequence().unwrap();
    store.close().unwrap();
    drop(store);

    let result = migrate_vector_precision(&store_dir, VectorPrecision::Float32, Some(&backup_dir)).unwrap();
    assert_eq!(result.from, VectorPrecision::Float64);
    assert_eq!(result.to, VectorPrecision::Float32);

    let mut converted = Store::new(Box::new(FileStorage::open(&store_dir, false).unwrap()), StoreConfig::default());
    let expected = (1.0f64 / 3.0) as f32 as f64;
    assert_eq!(converted.get_node("n").unwrap().unwrap().vector.unwrap()[0], expected);
    assert_eq!(converted.latest_mutation_sequence().unwrap(), sequence);
    assert!(converted.verify().unwrap().ok);
    converted.close().unwrap();
    drop(converted);

    let mut backup = Store::new(Box::new(FileStorage::open(&backup_dir, false).unwrap()), StoreConfig::default());
    assert_eq!(backup.vector_precision().unwrap(), VectorPrecision::Float64);
    assert_eq!(backup.get_node("n").unwrap().unwrap().vector.unwrap()[0], 1.0 / 3.0);
    assert_eq!(backup.latest_mutation_sequence().unwrap(), sequence);
    backup.close().unwrap();
    drop(backup);
    fs::remove_dir_all(root).unwrap();
}
