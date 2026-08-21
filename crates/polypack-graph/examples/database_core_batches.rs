//! Cross-binding durable batch-size benchmark — Rust lane.

use polypack_core::storage::FileStorage;
use polypack_core::{Node, StoreConfig};
use polypack_graph::{Graph, GraphConfig};
use serde_json::json;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const BATCHES: [usize; 4] = [1, 100, 500, 5_000];
const NO_AUTO_COMPACT: usize = 1_000_000_000;

struct Args { count: usize, seed: u32, out: PathBuf }

fn args() -> Args {
    let raw: Vec<String> = env::args().collect();
    let mut count = 5_000usize;
    let mut seed = 42u32;
    let mut out = PathBuf::from("benchmarks/results/database-core-batches-rust.json");
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--count" => { count = raw[i + 1].parse().expect("--count must be positive"); i += 2; }
            "--seed" => { seed = raw[i + 1].parse().expect("--seed must be a number"); i += 2; }
            "--out" => { out = PathBuf::from(&raw[i + 1]); i += 2; }
            _ => { i += 1; }
        }
    }
    Args { count, seed, out }
}

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b79f5);
        let mut t = (self.0 ^ (self.0 >> 15)).wrapping_mul(self.0 | 1);
        t = (t + (t ^ (t >> 7)).wrapping_mul(self.0 | 61)) ^ t;
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

fn node(i: usize, rng: &mut Rng) -> Node {
    let mut data = serde_json::Map::new();
    data.insert("idx".into(), json!(i));
    data.insert("value".into(), json!(rng.next()));
    data.insert("tag".into(), json!(format!("tag_{}", i % 50)));
    Node { id: format!("n{i}"), node_type: ["user", "post", "comment"][i % 3].into(), data, vector: None, inserted_at: i as i64, updated_at: i as i64, revision: 0, activation: None }
}

fn bytes(dir: &Path, name: &str) -> u64 {
    fs::metadata(dir.join(name)).map(|m| m.len()).unwrap_or(0)
}

fn run_case(root: &Path, args: &Args, batch_size: usize) -> serde_json::Value {
    let dir = root.join(format!("batch-{batch_size}"));
    fs::create_dir_all(&dir).unwrap();
    let storage = FileStorage::open(&dir, false).unwrap();
    let config = StoreConfig { compact_threshold: NO_AUTO_COMPACT, ..Default::default() };
    let mut graph = Graph::open(Box::new(storage), config, GraphConfig::default()).unwrap();
    let mut rng = Rng(args.seed);
    let start = Instant::now();
    for i in 0..args.count {
        graph.add_node(node(i, &mut rng)).unwrap();
        if (i + 1) % batch_size == 0 { graph.flush().unwrap(); }
    }
    if args.count % batch_size != 0 { graph.flush().unwrap(); }
    let write_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mutations = graph.mutation_log().unwrap().len();
    let before = json!({
        "walBytes": bytes(&dir, "wal.msgpack"),
        "snapshotBytes": bytes(&dir, "snapshot.msgpack"),
        "mutationLogBytes": bytes(&dir, "mutations.jsonl"),
    });
    let start = Instant::now();
    graph.checkpoint().unwrap();
    let compact_ms = start.elapsed().as_secs_f64() * 1000.0;
    let after = json!({
        "walBytes": bytes(&dir, "wal.msgpack"),
        "snapshotBytes": bytes(&dir, "snapshot.msgpack"),
        "mutationLogBytes": bytes(&dir, "mutations.jsonl"),
    });
    let verified = graph.verify().unwrap();
    graph.dispose().unwrap();
    json!({
        "batchSize": batch_size, "count": args.count, "seed": args.seed,
        "writeMs": write_ms, "writeOpsPerSec": args.count as f64 / (write_ms / 1000.0),
        "mutationCount": mutations, "compactMs": compact_ms,
        "verified": verified.ok, "nodeCount": verified.node_count,
        "before": before, "after": after,
    })
}

fn main() {
    let args = args();
    assert!(args.count > 0, "--count must be positive");
    let root = env::temp_dir().join(format!("polypack-batches-rust-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let cases: Vec<_> = BATCHES.iter().map(|batch| run_case(&root, &args, *batch)).collect();
    let result = json!({ "schemaVersion": 1, "lang": "rust", "count": args.count, "seed": args.seed, "batchSizes": BATCHES, "cases": cases });
    if let Some(parent) = args.out.parent() { fs::create_dir_all(parent).unwrap(); }
    fs::write(&args.out, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    fs::remove_dir_all(&root).ok();
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    println!("Wrote {}", args.out.display());
}
