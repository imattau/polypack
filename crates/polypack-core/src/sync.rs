//! Portable sync-envelope validation and checksums.
//!
//! This module deliberately stops at the wire contract. Transports, server
//! storage, authorization, and conflict resolution remain application-layer
//! concerns, matching the TypeScript sync implementation.

use crate::error::{PolypackError, Result};
use serde_json::{Map, Value};

/// Compute the TypeScript-compatible FNV-1a checksum for an ordered batch.
pub fn sync_checksum(operations: &[Value]) -> String {
    let input = serde_json::to_string(operations).expect("JSON sync values must serialize");
    let mut hash: u32 = 2_166_136_261;
    for unit in input.encode_utf16() {
        hash ^= unit as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    format!("{hash:08x}")
}

/// Compute the checksum used for retained operation/transaction identities.
pub fn sync_identity_checksum(operation_ids: &[String], transaction_ids: &[String]) -> String {
    let mut payload = Map::new();
    let mut operations = operation_ids.to_vec();
    let mut transactions = transaction_ids.to_vec();
    operations.sort();
    transactions.sort();
    payload.insert("operationIds".into(), Value::Array(operations.into_iter().map(Value::String).collect()));
    payload.insert("transactionIds".into(), Value::Array(transactions.into_iter().map(Value::String).collect()));
    let mut operation = Map::new();
    operation.insert("seq".into(), Value::from(0));
    operation.insert("timestamp".into(), Value::from(0));
    operation.insert("clientId".into(), Value::String("sync-identities".into()));
    operation.insert("kind".into(), Value::String("addNode".into()));
    operation.insert("payload".into(), Value::Object(payload));
    sync_checksum(&[Value::Object(operation)])
}

/// Validate the portable fields required by every sync operation.
pub fn validate_sync_operation(operation: &Value) -> Result<()> {
    let object = operation.as_object().ok_or_else(|| PolypackError::InvalidArgument("sync operation must be an object".into()))?;
    let seq = object.get("seq").and_then(Value::as_u64).ok_or_else(|| PolypackError::InvalidArgument("sync sequence must be a positive integer".into()))?;
    if seq == 0 { return Err(PolypackError::InvalidArgument("sync sequence must be a positive integer".into())); }
    if !object.get("timestamp").and_then(Value::as_f64).is_some_and(f64::is_finite) { return Err(PolypackError::InvalidArgument("sync timestamp must be finite".into())); }
    for field in ["clientId", "kind"] {
        if object.get(field).and_then(Value::as_str).is_none_or(str::is_empty) { return Err(PolypackError::InvalidArgument(format!("sync {field} must be non-empty"))); }
    }
    if !object.get("payload").is_some_and(Value::is_object) { return Err(PolypackError::InvalidArgument("sync payload must be an object".into())); }
    for field in ["operationId", "transactionId"] {
        if object.get(field).is_some() && object.get(field).and_then(Value::as_str).is_none_or(str::is_empty) { return Err(PolypackError::InvalidArgument(format!("sync {field} must be non-empty"))); }
    }
    if let Some(revision) = object.get("baseRevision") {
        if revision.as_u64().is_none() { return Err(PolypackError::InvalidArgument("sync baseRevision must be a non-negative integer".into())); }
    }
    Ok(())
}

/// Validate a batch and return its deterministic checksum.
pub fn validate_sync_batch(operations: &[Value]) -> Result<String> {
    for operation in operations { validate_sync_operation(operation)?; }
    Ok(sync_checksum(operations))
}
