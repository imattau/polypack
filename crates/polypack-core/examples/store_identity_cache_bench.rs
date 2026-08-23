//! Isolates `Store::apply_with_identity`'s idempotency-check cost as the
//! mutation log grows.
//!
//! Before the identity-cache fix, every identity-bearing `apply()` (which
//! `Graph::transaction()` always triggers, since it synthesizes a
//! transaction_id) re-read and re-parsed the *entire* `mutations.jsonl` log
//! to check for a previously-seen operation/transaction id — O(total
//! historical mutation count) per call, O(n^2) over n transactions. This
//! benchmark isolates that cost from storage I/O (in-memory backend) and
//! from WAL/snapshot compaction (disabled) by measuring wall time per
//! single-node transaction as the log grows.

use polypack_core::{ChangeBatch, InMemoryStorage, Node, Store, StoreConfig};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const NO_AUTO_COMPACT: usize = 1_000_000_000;

struct Args { transactions: usize, sample_every: usize, out: PathBuf }

fn args() -> Args {
    let raw: Vec<String> = env::args().collect();
    let mut transactions = 20_000usize;
    let mut sample_every = 1_000usize;
    let mut out = PathBuf::from("benchmarks/results/database-core-identity-cache-rust.json");
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--transactions" => { transactions = raw[i + 1].parse().unwrap(); i += 2; }
            "--sample-every" => { sample_every = raw[i + 1].parse().unwrap(); i += 2; }
            "--out" => { out = PathBuf::from(&raw[i + 1]); i += 2; }
            _ => { i += 1; }
        }
    }
    Args { transactions, sample_every, out }
}

fn node(i: usize) -> Node {
    Node { id: format!("n{i}"), node_type: "n".into(), data: serde_json::Map::new(), vector: None, inserted_at: i as i64, updated_at: i as i64, revision: 0, activation: None, ..Default::default() }
}

fn main() {
    let a = args();
    let config = StoreConfig { compact_threshold: NO_AUTO_COMPACT, ..Default::default() };
    let mut store = Store::new(Box::new(InMemoryStorage::new()), config);

    let mut samples = Vec::new();
    for i in 0..a.transactions {
        let changes = ChangeBatch { put_nodes: vec![node(i)], ..Default::default() };
        let transaction_id = format!("tx-{i}");
        let start = Instant::now();
        store.apply_with_transaction(&changes, Some(&transaction_id)).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if i % a.sample_every == 0 || i == a.transactions - 1 {
            samples.push(json!({ "transactionIndex": i, "applyMs": elapsed_ms }));
        }
    }

    let result = json!({
        "schemaVersion": 1,
        "lang": "rust",
        "transactions": a.transactions,
        "samples": samples,
    });
    if let Some(parent) = a.out.parent() { fs::create_dir_all(parent).unwrap(); }
    fs::write(&a.out, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!("Wrote {}", a.out.display());
    let first = samples[0]["applyMs"].as_f64().unwrap();
    let last = samples[samples.len() - 1]["applyMs"].as_f64().unwrap();
    println!(
        "first transaction: {first:.4}ms, last transaction: {last:.4}ms (ratio {:.1}x over {} transactions)",
        last / first.max(0.0001),
        a.transactions,
    );
}
