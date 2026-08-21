use polypack_core::sync::{sync_checksum, sync_identity_checksum, validate_sync_batch};
use serde_json::Value;
use std::path::PathBuf;

#[test]
fn shared_sync_protocol_fixture_passes() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/sync/protocol.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let operations = fixture["operations"].as_array().unwrap();
    assert_eq!(validate_sync_batch(operations).unwrap(), fixture["checksum"].as_str().unwrap());
    assert_eq!(sync_checksum(operations), fixture["checksum"].as_str().unwrap());
    let operation_ids = fixture["operationIds"].as_array().unwrap().iter().map(|value| value.as_str().unwrap().to_string()).collect::<Vec<_>>();
    let transaction_ids = fixture["transactionIds"].as_array().unwrap().iter().map(|value| value.as_str().unwrap().to_string()).collect::<Vec<_>>();
    assert_eq!(sync_identity_checksum(&operation_ids, &transaction_ids), fixture["identityChecksum"].as_str().unwrap());
}
