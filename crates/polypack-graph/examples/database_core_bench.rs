//! database-core benchmark — Rust lane.
//!
//! Companion to `benchmarks/database-core-ts.ts` and
//! `python/polypack/bench_db.py`: measures the same three things against a
//! real on-disk store so `benchmarks/database-core-compare.ts` can merge all
//! three into one report.
//!
//!   1. durable write throughput — `Graph::add_node` + `Graph::flush()`
//!      against a `FileStorage`-backed store.
//!   2. mutation-log replay + recovery — reopen the store cold and time
//!      `Graph::mutation_log()` and `Graph::warm()`.
//!   3. sync throughput — `polypack_core::sync::SyncServer::submit()`
//!      in-process op ingestion.
//!
//! Run:
//!   cargo run --release --example database_core_bench --manifest-path crates/Cargo.toml -- --count 20000 --sync-ops 5000

use polypack_core::storage::FileStorage;
use polypack_core::sync::SyncServer;
use polypack_core::{Node, StoreConfig};
use polypack_graph::{Graph, GraphConfig};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    count: usize,
    sync_ops: usize,
    dir: PathBuf,
    out: PathBuf,
    seed: u32,
}

fn parse_args() -> Args {
    let mut count = 20_000usize;
    let mut sync_ops = 5_000usize;
    let mut dir = env::temp_dir().join(format!("polypack-bench-rust-{}", std::process::id()));
    let mut out = PathBuf::from("benchmarks/results/database-core-rust.json");
    let mut seed = 42u32;
    let raw: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--count" => { count = raw[i + 1].parse().expect("--count must be a number"); i += 2; }
            "--sync-ops" => { sync_ops = raw[i + 1].parse().expect("--sync-ops must be a number"); i += 2; }
            "--dir" => { dir = PathBuf::from(&raw[i + 1]); i += 2; }
            "--out" => { out = PathBuf::from(&raw[i + 1]); i += 2; }
            "--seed" => { seed = raw[i + 1].parse().expect("--seed must be a number"); i += 2; }
            _ => { i += 1; }
        }
    }
    Args { count, sync_ops, dir, out, seed }
}

/// Same PRNG as the TS/Python benchmarks: bit-for-bit mulberry32.
struct Mulberry32(u32);
impl Mulberry32 {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b79f5);
        let mut t = self.0;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

fn make_node(i: usize, rng: &mut Mulberry32) -> Node {
    let node_type = ["user", "post", "comment"][i % 3].to_string();
    let mut data = serde_json::Map::new();
    data.insert("idx".into(), json!(i));
    data.insert("value".into(), json!(rng.next()));
    data.insert("tag".into(), json!(format!("tag_{}", i % 50)));
    Node { id: format!("n{i}"), node_type, data, vector: None, inserted_at: i as i64, updated_at: i as i64, revision: 0, activation: None, ..Default::default() }
}

fn bench_durable_writes(args: &Args) -> (f64, f64) {
    let _ = fs::remove_dir_all(&args.dir);
    let storage = FileStorage::open(&args.dir, false).expect("open store");
    let mut graph = Graph::open(Box::new(storage), StoreConfig::default(), GraphConfig::default()).expect("open graph");
    let mut rng = Mulberry32(args.seed);
    const FLUSH_EVERY: usize = 500;
    let t0 = Instant::now();
    let mut batch = Vec::with_capacity(FLUSH_EVERY);
    for i in 0..args.count {
        batch.push(make_node(i, &mut rng));
        if batch.len() >= FLUSH_EVERY {
            graph.add_nodes(std::mem::take(&mut batch)).expect("add_nodes");
            graph.flush().expect("flush");
        }
    }
    if !batch.is_empty() {
        graph.add_nodes(batch).expect("add_nodes");
        graph.flush().expect("flush");
    }
    let write_ms = t0.elapsed().as_secs_f64() * 1000.0;
    graph.dispose().expect("dispose");
    (write_ms, args.count as f64 / (write_ms / 1000.0))
}

fn bench_mutation_log_and_recovery(args: &Args) -> (usize, f64, f64) {
    let open_start = Instant::now();
    let storage = FileStorage::open(&args.dir, false).expect("reopen store");
    let mut graph = Graph::open(Box::new(storage), StoreConfig::default(), GraphConfig::default()).expect("reopen graph");
    graph.warm().expect("warm");
    let recovery_ms = open_start.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let mutations = graph.mutation_log().expect("mutation_log");
    let replay_ms = t1.elapsed().as_secs_f64() * 1000.0;

    graph.dispose().expect("dispose");
    (mutations.len(), replay_ms, recovery_ms)
}

fn bench_sync_throughput(args: &Args) -> (f64, f64) {
    let mut server = SyncServer::new(1, None, None).expect("sync server");
    const BATCH_SIZE: usize = 100;
    let mut seq: u64 = 0;
    let t0 = Instant::now();
    let mut offset = 0;
    while offset < args.sync_ops {
        let mut ops: Vec<Value> = Vec::new();
        let end = (offset + BATCH_SIZE).min(args.sync_ops);
        for _ in offset..end {
            seq += 1;
            ops.push(json!({
                "seq": seq,
                "timestamp": seq,
                "clientId": "bench-client",
                "kind": "addNode",
                "payload": { "id": format!("s{seq}") },
                "operationId": format!("bench-client:{seq}"),
            }));
        }
        server.submit(&ops).expect("submit");
        offset = end;
    }
    let submit_ms = t0.elapsed().as_secs_f64() * 1000.0;
    (submit_ms, args.sync_ops as f64 / (submit_ms / 1000.0))
}

fn main() {
    let args = parse_args();
    println!("polypack-graph — database-core benchmark");
    println!("  count={} sync_ops={} dir={}", args.count, args.sync_ops, args.dir.display());

    let (write_ms, write_ops_per_sec) = bench_durable_writes(&args);
    println!("  durable writes: {:.1}ms ({:.0} ops/sec)", write_ms, write_ops_per_sec);

    let (mutation_count, mutation_replay_ms, recovery_ms) = bench_mutation_log_and_recovery(&args);
    println!("  mutation log: {} records, replay {:.2}ms, recovery (warm) {:.2}ms", mutation_count, mutation_replay_ms, recovery_ms);

    let (sync_submit_ms, sync_ops_per_sec) = bench_sync_throughput(&args);
    println!("  sync throughput: {:.1}ms ({:.0} ops/sec)", sync_submit_ms, sync_ops_per_sec);

    let _ = fs::remove_dir_all(&args.dir);

    let result = json!({
        "results": [{
            "lang": "rust",
            "count": args.count,
            "syncOps": args.sync_ops,
            "writeMs": write_ms,
            "writeOpsPerSec": write_ops_per_sec,
            "mutationCount": mutation_count,
            "mutationReplayMs": mutation_replay_ms,
            "recoveryMs": recovery_ms,
            "syncSubmitMs": sync_submit_ms,
            "syncOpsPerSec": sync_ops_per_sec,
        }]
    });
    if let Some(parent) = args.out.parent() { let _ = fs::create_dir_all(parent); }
    fs::write(&args.out, serde_json::to_string_pretty(&result).unwrap()).expect("write results");
    println!("Wrote {}", args.out.display());
}
