//! Isolates `Graph::flush()`'s per-edge cost as total edge count grows.
//!
//! `flush()` resolves each dirty edge id by scanning every source's edge map
//! (`self.edges.iter().find_map(...)`) instead of using an id index. That
//! predicts flush time per fixed-size batch to grow ~linearly with the
//! total edges already inserted (quadratic overall). This benchmark isolates
//! that cost from storage I/O (in-memory backend) and from node writes (all
//! nodes are created up front, unflushed edges only).

use polypack_core::{InMemoryStorage, Node, StoreConfig};
use polypack_graph::{EdgeOwnership, Graph, GraphConfig};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// Matches database_core_batches.rs: disable auto-compaction so periodic
// Store::compact() calls (whose cost scales with total store size) don't
// confound the per-edge flush-cost measurement under test here.
const NO_AUTO_COMPACT: usize = 1_000_000_000;

struct Args { nodes: usize, batch_size: usize, batches: usize, out: PathBuf }

fn args() -> Args {
    let raw: Vec<String> = env::args().collect();
    let mut nodes = 2_000usize;
    let mut batch_size = 200usize;
    let mut batches = 100usize;
    let mut out = PathBuf::from("benchmarks/results/database-core-edge-flush-rust.json");
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--nodes" => { nodes = raw[i + 1].parse().unwrap(); i += 2; }
            "--batch-size" => { batch_size = raw[i + 1].parse().unwrap(); i += 2; }
            "--batches" => { batches = raw[i + 1].parse().unwrap(); i += 2; }
            "--out" => { out = PathBuf::from(&raw[i + 1]); i += 2; }
            _ => { i += 1; }
        }
    }
    Args { nodes, batch_size, batches, out }
}

fn node(i: usize) -> Node {
    Node { id: format!("n{i}"), node_type: "n".into(), data: serde_json::Map::new(), vector: None, inserted_at: i as i64, updated_at: i as i64, revision: 0, activation: None, ..Default::default() }
}

fn main() {
    let a = args();
    let config = StoreConfig { compact_threshold: NO_AUTO_COMPACT, ..Default::default() };
    let mut g = Graph::open(Box::new(InMemoryStorage::new()), config, GraphConfig::default()).unwrap();
    g.add_nodes((0..a.nodes).map(node).collect()).unwrap();
    g.flush().unwrap();

    // Every edge points into a small ring of targets so `add_edge` stays
    // O(1); only the flush-time id resolution is under test. Sources and
    // edge types are varied so distinct edges accumulate across sources.
    let mut edge_seq = 0usize;
    let mut per_batch = Vec::with_capacity(a.batches);
    for batch in 0..a.batches {
        let start = Instant::now();
        for _ in 0..a.batch_size {
            let source = format!("n{}", edge_seq % a.nodes);
            let target = format!("n{}", (edge_seq + 1) % a.nodes);
            // edge_type varies per wrap around `nodes` so the (source, type,
            // target) triple stays unique — otherwise add_edge would treat
            // a repeat as a no-op duplicate once edge_seq exceeds `nodes`.
            let edge_type = format!("rel{}", edge_seq / a.nodes);
            g.add_edge(&source, &edge_type, &target, None, EdgeOwnership::Reference).unwrap();
            edge_seq += 1;
        }
        g.flush().unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        per_batch.push(json!({
            "batch": batch,
            "totalEdgesBefore": batch * a.batch_size,
            "flushMs": elapsed_ms,
            "msPerEdgeInBatch": elapsed_ms / a.batch_size as f64,
        }));
    }

    let result = json!({
        "schemaVersion": 1,
        "lang": "rust",
        "nodes": a.nodes,
        "batchSize": a.batch_size,
        "batches": a.batches,
        "totalEdges": edge_seq,
        "perBatch": per_batch,
    });
    if let Some(parent) = a.out.parent() { fs::create_dir_all(parent).unwrap(); }
    fs::write(&a.out, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!("Wrote {}", a.out.display());
    println!(
        "first batch: {:.4}ms/edge, last batch: {:.4}ms/edge (ratio {:.1}x)",
        per_batch[0]["msPerEdgeInBatch"].as_f64().unwrap(),
        per_batch[a.batches - 1]["msPerEdgeInBatch"].as_f64().unwrap(),
        per_batch[a.batches - 1]["msPerEdgeInBatch"].as_f64().unwrap() / per_batch[0]["msPerEdgeInBatch"].as_f64().unwrap(),
    );
}
