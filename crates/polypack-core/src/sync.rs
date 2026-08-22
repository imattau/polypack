//! Portable sync-envelope validation and checksums.
//!
//! This module deliberately stops at the wire contract. Transports, server
//! storage, authorization, and conflict resolution remain application-layer
//! concerns, matching the TypeScript sync implementation.

use crate::error::{PolypackError, Result};
use serde_json::{Map, Value};
use std::collections::HashSet;

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

/// Synchronous, transport-neutral server state machine for native Rust hosts.
/// Durable storage and authorization are supplied by the host around this
/// state machine; the TypeScript and Python servers expose the same semantics.
pub struct SyncServer {
    protocol_version: u64,
    max_ops: Option<usize>,
    max_batch_ops: Option<usize>,
    base_cursor: u64,
    ops: Vec<Value>,
    operation_ids: HashSet<String>,
    transaction_ids: HashSet<String>,
    seen_ops: HashSet<String>,
}

impl SyncServer {
    pub fn new(protocol_version: u64, max_ops: Option<usize>, max_batch_ops: Option<usize>) -> Result<Self> {
        if protocol_version == 0 || max_ops == Some(0) || max_batch_ops == Some(0) { return Err(PolypackError::InvalidArgument("invalid sync server limits".into())); }
        Ok(Self { protocol_version, max_ops, max_batch_ops, base_cursor: 0, ops: Vec::new(), operation_ids: HashSet::new(), transaction_ids: HashSet::new(), seen_ops: HashSet::new() })
    }

    pub fn cursor(&self) -> u64 { self.base_cursor + self.ops.len() as u64 }
    pub fn operations(&self) -> &[Value] { &self.ops }

    pub fn submit(&mut self, operations: &[Value]) -> Result<Vec<Value>> {
        if self.max_batch_ops.is_some_and(|limit| operations.len() > limit) { return Err(PolypackError::ResourceLimit { name: "maxBatchOps".into(), limit: self.max_batch_ops.unwrap() }); }
        // `validate_sync_batch` also computes a whole-batch checksum (a full
        // JSON serialize + UTF-16 hash pass) that submit() never uses —
        // validate each operation directly instead of paying for a checksum
        // that's immediately discarded.
        for operation in operations { validate_sync_operation(operation)?; }
        let mut accepted = Vec::new();
        let mut accepted_transactions = HashSet::new();
        for operation in operations {
            let object = operation.as_object().unwrap();
            let client_id = object["clientId"].as_str().unwrap();
            let seq_key = format!("{client_id}:{}", object["seq"].as_u64().unwrap());
            let operation_key = object.get("operationId").and_then(Value::as_str).map(|id| format!("{client_id}:{id}"));
            let transaction_key = object.get("transactionId").and_then(Value::as_str).map(|id| format!("{client_id}:{id}"));
            if self.seen_ops.contains(&seq_key)
                || operation_key.as_ref().is_some_and(|key| self.operation_ids.contains(key))
                || transaction_key.as_ref().is_some_and(|key| self.transaction_ids.contains(key) && !accepted_transactions.contains(key))
            { continue; }
            self.seen_ops.insert(seq_key);
            if let Some(key) = &operation_key { self.operation_ids.insert(key.clone()); }
            if let Some(key) = &transaction_key { self.transaction_ids.insert(key.clone()); accepted_transactions.insert(key.clone()); }
            // One clone stays in `self.ops`, the other is returned to the
            // caller — both are needed, but doing them together here (rather
            // than cloning into `accepted` and then cloning `accepted` again
            // into `self.ops` afterwards) avoids a second full pass over the
            // accepted operations.
            self.ops.push(operation.clone());
            accepted.push(operation.clone());
        }
        if let Some(limit) = self.max_ops {
            if self.ops.len() > limit {
                let removed = self.ops.len() - limit;
                // Only the seq-keyed entry is bounded by the ring buffer;
                // operationId/transactionId identities are retained forever
                // so a delayed retry after compaction is still deduped
                // instead of being re-accepted. Matches the TS/Python servers.
                for op in self.ops.drain(..removed) {
                    let object = op.as_object().unwrap();
                    let key = format!("{}:{}", object["clientId"].as_str().unwrap(), object["seq"].as_u64().unwrap());
                    self.seen_ops.remove(&key);
                }
                self.base_cursor += removed as u64;
            }
        }
        Ok(accepted)
    }

    pub fn recover(&self, from_cursor: u64, limit: usize) -> Result<Value> {
        let valid = from_cursor >= self.base_cursor && from_cursor <= self.cursor();
        let requested = if valid { from_cursor } else { 0 };
        let offset = if requested == 0 { 0 } else { (requested - self.base_cursor) as usize };
        let page: Vec<Value> = self.ops.iter().skip(offset).take(limit).cloned().collect();
        let cursor = (if requested == 0 { self.base_cursor } else { requested }) + page.len() as u64;
        let checksum = sync_checksum(&page);
        let mut response = serde_json::json!({ "type": if requested == 0 { "snapshot" } else { "delta" }, "clientId": "server", "fromSeq": requested, "cursor": cursor, "more": offset + page.len() < self.ops.len(), "ops": page, "protocolVersion": self.protocol_version, "checksum": checksum });
        if !valid { response["errors"] = serde_json::json!([{ "code": "cursor_expired", "message": "requested cursor is no longer available" }]); }
        Ok(response)
    }
}
